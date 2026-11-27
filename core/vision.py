import base64
from pathlib import Path

from PIL import Image

from core.media import extract_keyframes


def _build_prompt(duration: float, style_hint: str = "") -> str:
    target_chars = max(12, int(round(duration * 4.0)))
    min_chars = max(8, int(target_chars * 0.85))
    max_chars = max(16, int(target_chars * 1.12))
    extra = f"\n风格要求：{style_hint.strip()}" if style_hint.strip() else ""
    return (
        f"你是一位专业的短视频口播文案策划师。请按时间顺序观察这组来自同一个 {duration:.1f} 秒视频的关键帧画面，"
        f"为这段视频写一段自然、流畅、吸引人的中文口播解说词。\n"
        f"严格要求：\n"
        f"1. 总字数严格控制在 {min_chars} 到 {max_chars} 个汉字之间（按正常口播每秒约4字计算，确保能与视频时长精准同步）。\n"
        f"2. 只输出纯口播正文，不要输出任何标题、分镜编号、时间戳、括号动作提示或 Markdown 符号。\n"
        f"3. 用自然的短句表达，每句用中文标点（，。！？）分隔，方便后续自动断句生成字幕。{extra}"
    )


def generate_script_from_video(
    video_path: Path,
    duration: float,
    work_dir: Path,
    provider: str,
    api_key: str,
    model_name: str,
    base_url: str = "",
    style_hint: str = "",
) -> str:
    """Extract keyframes from a silent video and ask a multimodal LLM to write a timed oral script."""
    if not api_key.strip():
        raise ValueError("请先填写视觉大模型的 API Key，或者直接在下方手动输入口播文案。")

    frames_dir = work_dir / "keyframes"
    max_frames = max(4, min(10, int(duration // 3) + 2))
    frames = extract_keyframes(video_path, duration, frames_dir, max_frames=max_frames)
    if not frames:
        raise RuntimeError("未能从视频中抽取关键帧。")

    prompt = _build_prompt(duration, style_hint=style_hint)

    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=api_key.strip())
        contents: list = [prompt]
        for frame in frames:
            contents.append(Image.open(frame))
        resp = client.models.generate_content(
            model=model_name.strip() or "gemini-2.5-flash",
            contents=contents,
        )
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("Gemini 返回了空文案。")
        return text

    # OpenAI-compatible multimodal endpoint (OpenAI, Qwen-VL / DashScope, SiliconFlow, OpenRouter, etc.)
    from openai import OpenAI

    client_kwargs = {"api_key": api_key.strip()}
    if base_url.strip():
        client_kwargs["base_url"] = base_url.strip()
    client = OpenAI(**client_kwargs)

    user_content: list[dict] = [{"type": "text", "text": prompt}]
    for frame in frames:
        b64 = base64.b64encode(frame.read_bytes()).decode("ascii")
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    resp = client.chat.completions.create(
        model=model_name.strip() or "gpt-4o-mini",
        messages=[{"role": "user", "content": user_content}],
        temperature=0.7,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("模型返回了空文案。")
    return text
