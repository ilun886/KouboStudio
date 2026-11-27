"""KouboStudio MCP Server (stdio).

Exposes the one-click talking-head / voiceover video editing pipeline as MCP tools
so any MCP-compatible AI Agent (Antigravity, Claude Desktop, Cursor, etc.) can
analyze videos, generate scripts, dub with audio-video sync, burn subtitles, and mix BGM.
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure repo root is importable regardless of working directory
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Important: redirect loguru to stderr so stdout stays 100% clean for MCP JSON-RPC!
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

from core.asr import detect_and_transcribe
from core.composer import compose_final_video
from core.media import probe_video
from core.models import RenderOptions, Segment, SubtitleStyle
from core.tts import (
    ZH_VOICES,
    build_synced_voice_for_script,
    build_synced_voice_for_segments,
)
from core.vision import generate_script_from_video

WORKSPACE_DIR = ROOT_DIR / "workspace"
BGM_DIR = ROOT_DIR / "assets" / "bgm"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

mcp = MCPServer("koubo-studio")


def _resolve_bgm(bgm: str | None) -> Path | None:
    if not bgm or bgm.lower() in ("none", "off", ""):
        return None
    p = Path(bgm)
    if p.is_file():
        return p
    cand = BGM_DIR / bgm
    if cand.is_file():
        return cand
    if bgm == "default":
        files = sorted(BGM_DIR.glob("*.mp3"))
        return files[0] if files else None
    raise FileNotFoundError(f"BGM not found: {bgm}")


@mcp.tool()
def list_koubo_assets() -> dict[str, Any]:
    """List available Chinese TTS voices and built-in BGM tracks for KouboStudio."""
    bgms = sorted(p.name for p in BGM_DIR.glob("*.mp3")) if BGM_DIR.exists() else []
    return {
        "voices": ZH_VOICES,
        "builtin_bgms": bgms,
        "default_voice": "zh-CN-XiaoxiaoNeural",
        "default_bgm": bgms[0] if bgms else None,
    }


@mcp.tool()
def analyze_video(
    video_path: str,
    whisper_model: str = "small",
    language: str = "zh",
) -> dict[str, Any]:
    """
    Analyze a video file: probe metadata, detect whether it has valid human speech
    (voiced vs silent mode), and transcribe timestamped segments if voiced.
    """
    vp = Path(video_path).expanduser().resolve()
    if not vp.is_file():
        raise FileNotFoundError(f"Video file not found: {vp}")

    info = probe_video(vp)
    work_dir = WORKSPACE_DIR / f"mcp_analyze_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    has_speech, segments, detected_lang = detect_and_transcribe(
        video_path=vp,
        work_dir=work_dir,
        model_size=whisper_model,
        language=language,
    )
    return {
        "video_path": str(vp),
        "mode": "voiced" if has_speech else "silent",
        "has_speech": has_speech,
        "detected_language": detected_lang,
        "video_info": {
            "duration": round(info.duration, 3),
            "width": info.width,
            "height": info.height,
            "fps": round(info.fps, 2),
            "has_audio_stream": info.has_audio_stream,
        },
        "recommended_chars_for_silent": max(12, int(round(info.duration * 4.0))),
        "segments": [
            {"index": s.index, "start": s.start, "end": s.end, "text": s.text}
            for s in segments
        ],
    }


@mcp.tool()
def generate_silent_script(
    video_path: str,
    provider: str = "gemini",
    api_key: str = "",
    model_name: str = "",
    base_url: str = "",
    style_hint: str = "",
) -> dict[str, Any]:
    """
    Extract keyframes from a silent video and generate a duration-matched Chinese
    oral-broadcast script using a multimodal LLM (Gemini or OpenAI-compatible).
    Falls back to GEMINI_API_KEY / OPENAI_API_KEY environment variables if api_key is empty.
    """
    vp = Path(video_path).expanduser().resolve()
    if not vp.is_file():
        raise FileNotFoundError(f"Video file not found: {vp}")

    resolved_key = api_key.strip() or (
        os.environ.get("GEMINI_API_KEY", "")
        if provider == "gemini"
        else os.environ.get("OPENAI_API_KEY", "")
    )
    info = probe_video(vp)
    work_dir = WORKSPACE_DIR / f"mcp_vision_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    script = generate_script_from_video(
        video_path=vp,
        duration=info.duration,
        work_dir=work_dir,
        provider=provider,
        api_key=resolved_key,
        model_name=model_name,
        base_url=base_url,
        style_hint=style_hint,
    )
    return {
        "video_path": str(vp),
        "duration": round(info.duration, 3),
        "script": script,
        "char_count": len(script),
    }


@mcp.tool()
def produce_koubo_video(
    video_path: str,
    output_path: str = "",
    script_text: str = "",
    segments: list[dict[str, Any]] | None = None,
    keep_original_voice: bool = False,
    voice_name: str = "zh-CN-XiaoxiaoNeural",
    voice_rate: str = "+0%",
    bgm: str = "default",
    bgm_volume: float = 0.22,
    bgm_ducking: bool = True,
    subtitle_enabled: bool = True,
    subtitle_position: str = "bottom",
    subtitle_font_size: int = 56,
    subtitle_color: str = "#FFFFFF",
    whisper_model: str = "small",
    vision_provider: str = "gemini",
    vision_api_key: str = "",
    vision_model: str = "",
    vision_base_url: str = "",
    style_hint: str = "",
) -> dict[str, Any]:
    """
    One-click talking-head / voiceover video production:
    - Automatically detects voiced vs silent video (unless segments or script_text is provided).
    - Voiced mode: transcribes speech -> keeps original voice OR re-dubs per segment with audio-video sync.
    - Silent mode: uses script_text (or auto-generates via vision LLM if API key is available) -> TTS + duration sync.
    - Burns crisp Chinese ASS subtitles and mixes BGM with optional audio ducking.
    """
    vp = Path(video_path).expanduser().resolve()
    if not vp.is_file():
        raise FileNotFoundError(f"Video file not found: {vp}")

    info = probe_video(vp)
    job_id = uuid.uuid4().hex[:10]
    work_dir = WORKSPACE_DIR / f"mcp_job_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    out_mp4 = (
        Path(output_path).expanduser().resolve()
        if output_path.strip()
        else (work_dir / f"{vp.stem}_koubo.mp4")
    )
    bgm_path = _resolve_bgm(bgm)

    voice_wav: Path | None = None
    final_segments: list[Segment] = []
    mode = ""

    if segments:
        mode = "voiced"
        seg_objs = [
            Segment(
                index=i + 1,
                start=float(s["start"]),
                end=float(s["end"]),
                text=str(s["text"]).strip(),
            )
            for i, s in enumerate(segments)
            if str(s.get("text", "")).strip()
        ]
        if keep_original_voice and info.has_audio_stream:
            final_segments = seg_objs
        else:
            voice_wav, final_segments = build_synced_voice_for_segments(
                segments=seg_objs,
                video_duration=info.duration,
                work_dir=work_dir,
                voice=voice_name,
                base_rate=voice_rate,
            )
    elif script_text.strip():
        mode = "silent"
        voice_wav, final_segments = build_synced_voice_for_script(
            script_text=script_text.strip(),
            video_duration=info.duration,
            work_dir=work_dir,
            voice=voice_name,
            base_rate=voice_rate,
        )
    else:
        has_speech, detected_segs, _ = detect_and_transcribe(
            video_path=vp,
            work_dir=work_dir,
            model_size=whisper_model,
            language="zh",
        )
        if has_speech:
            mode = "voiced"
            if keep_original_voice:
                final_segments = detected_segs
            else:
                voice_wav, final_segments = build_synced_voice_for_segments(
                    segments=detected_segs,
                    video_duration=info.duration,
                    work_dir=work_dir,
                    voice=voice_name,
                    base_rate=voice_rate,
                )
        else:
            mode = "silent"
            resolved_key = vision_api_key.strip() or (
                os.environ.get("GEMINI_API_KEY", "")
                if vision_provider == "gemini"
                else os.environ.get("OPENAI_API_KEY", "")
            )
            if not resolved_key:
                raise ValueError(
                    "检测到这是无声视频，请传入 script_text 口播文案，或提供 vision_api_key 让视觉模型自动看画面写文案。"
                )
            auto_script = generate_script_from_video(
                video_path=vp,
                duration=info.duration,
                work_dir=work_dir,
                provider=vision_provider,
                api_key=resolved_key,
                model_name=vision_model,
                base_url=vision_base_url,
                style_hint=style_hint,
            )
            voice_wav, final_segments = build_synced_voice_for_script(
                script_text=auto_script,
                video_duration=info.duration,
                work_dir=work_dir,
                voice=voice_name,
                base_rate=voice_rate,
            )

    pos = subtitle_position if subtitle_position in ("bottom", "middle", "top") else "bottom"
    options = RenderOptions(
        keep_original_voice=keep_original_voice and mode == "voiced",
        voice_name=voice_name,
        voice_rate=voice_rate,
        bgm_path=bgm_path,
        bgm_volume=bgm_volume,
        bgm_ducking=bgm_ducking,
        subtitle_enabled=subtitle_enabled,
        subtitle_style=SubtitleStyle(
            font_name="Microsoft YaHei",
            font_size=subtitle_font_size,
            primary_color=subtitle_color,
            position=pos,
        ),
    )

    mp4_path, ass_path = compose_final_video(
        video_path=vp,
        voice_wav_path=voice_wav,
        segments=final_segments,
        output_mp4=out_mp4,
        work_dir=work_dir,
        options=options,
    )
    out_info = probe_video(mp4_path)
    return {
        "status": "ok",
        "mode": mode,
        "output_mp4": str(mp4_path),
        "subtitle_ass": str(ass_path) if ass_path else None,
        "duration": round(out_info.duration, 3),
        "segments_count": len(final_segments),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
