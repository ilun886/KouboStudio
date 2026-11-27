import asyncio
import re
from pathlib import Path

import edge_tts
from pydub import AudioSegment

from core.media import change_audio_speed, get_audio_duration
from core.models import Segment

ZH_VOICES: dict[str, str] = {
    "zh-CN-XiaoxiaoNeural": "晓晓（女声 · 温暖自然 · 推荐口播）",
    "zh-CN-YunxiNeural": "云希（男声 · 阳光生动 · 推荐口播）",
    "zh-CN-YunjianNeural": "云健（男声 · 沉稳有力 · 知识解说）",
    "zh-CN-XiaoyiNeural": "晓伊（女声 · 清脆活泼 · 探店种草）",
    "zh-CN-YunyangNeural": "云扬（男声 · 新闻播报 · 专业干练）",
    "zh-CN-liaoning-XiaobeiNeural": "晓北（女声 · 东北幽默）",
    "zh-CN-shaanxi-XiaoniNeural": "晓妮（女声 · 陕西特色）",
}


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Fallback when called inside an already running loop
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


async def _edge_tts_generate(
    text: str,
    output_mp3: Path,
    voice: str,
    rate: str = "+0%",
) -> list[Segment]:
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        boundary="SentenceBoundary",
    )
    submaker = edge_tts.SubMaker()
    boundaries: list[Segment] = []

    with open(output_mp3, "wb") as f:
        async for chunk in communicate.stream():
            ctype = chunk.get("type")
            if ctype == "audio":
                f.write(chunk["data"])
            elif ctype in ("SentenceBoundary", "WordBoundary"):
                submaker.feed(chunk)
                if ctype == "SentenceBoundary":
                    start_s = chunk["offset"] / 10_000_000.0
                    dur_s = chunk["duration"] / 10_000_000.0
                    seg_text = str(chunk.get("text", "")).strip()
                    if seg_text:
                        boundaries.append(
                            Segment(
                                index=len(boundaries) + 1,
                                start=round(start_s, 3),
                                end=round(start_s + dur_s, 3),
                                text=seg_text,
                            )
                        )
    return boundaries


def _split_script_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[。！？!?；;\n])", cleaned)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else ([cleaned] if cleaned else [])


def _fallback_proportional_segments(text: str, total_duration: float) -> list[Segment]:
    sentences = _split_script_sentences(text)
    if not sentences:
        return []
    total_len = sum(max(1, len(s)) for s in sentences)
    cursor = 0.0
    result: list[Segment] = []
    for idx, sent in enumerate(sentences, start=1):
        dur = total_duration * (max(1, len(sent)) / total_len)
        result.append(
            Segment(
                index=idx,
                start=round(cursor, 3),
                end=round(cursor + dur, 3),
                text=sent,
            )
        )
        cursor += dur
    return result


def build_synced_voice_for_segments(
    segments: list[Segment],
    video_duration: float,
    work_dir: Path,
    voice: str = "zh-CN-XiaoxiaoNeural",
    base_rate: str = "+0%",
) -> tuple[Path, list[Segment]]:
    """
    Voiced-video re-dubbing with segment-level audio-video synchronization.
    Each segment is synthesized independently, gently tempo-adjusted to fit its original
    time slot without overlapping the next segment, and placed onto a unified timeline.
    """
    tts_dir = work_dir / "tts_segments"
    tts_dir.mkdir(parents=True, exist_ok=True)

    valid_segs = [s for s in segments if s.text.strip()]
    if not valid_segs:
        raise ValueError("没有可用于配音的字幕内容。")

    placed_clips: list[tuple[int, AudioSegment]] = []
    synced_segments: list[Segment] = []
    prev_end = 0.0

    for i, seg in enumerate(valid_segs):
        raw_mp3 = tts_dir / f"seg_{i:04d}_raw.mp3"
        synced_mp3 = tts_dir / f"seg_{i:04d}_sync.mp3"

        _run_async(_edge_tts_generate(seg.text.strip(), raw_mp3, voice=voice, rate=base_rate))
        tts_dur = get_audio_duration(raw_mp3)
        if tts_dur <= 0.01:
            continue

        actual_start = max(seg.start, prev_end)
        orig_dur = max(0.25, seg.end - seg.start)
        next_start = valid_segs[i + 1].start if i + 1 < len(valid_segs) else video_duration
        slot_dur = max(0.30, next_start - actual_start)

        speed = 1.0
        if tts_dur > slot_dur:
            speed = min(1.38, tts_dur / slot_dur)
        elif tts_dur > orig_dur * 1.08:
            speed = min(1.18, tts_dur / orig_dur)

        if speed > 1.02:
            change_audio_speed(raw_mp3, synced_mp3, speed)
            final_clip_path = synced_mp3
            final_dur = get_audio_duration(synced_mp3)
        else:
            final_clip_path = raw_mp3
            final_dur = tts_dur

        clip_audio = AudioSegment.from_file(str(final_clip_path))
        start_ms = int(round(actual_start * 1000))
        placed_clips.append((start_ms, clip_audio))

        actual_end = actual_start + final_dur
        prev_end = actual_end + 0.03
        synced_segments.append(
            Segment(
                index=len(synced_segments) + 1,
                start=round(actual_start, 3),
                end=round(actual_end, 3),
                text=seg.text.strip(),
            )
        )

    total_ms = int(round(max(video_duration, prev_end) * 1000))
    canvas = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    for start_ms, clip in placed_clips:
        canvas = canvas.overlay(clip.set_frame_rate(44100).set_channels(2), position=start_ms)

    out_wav = work_dir / "dubbed_voice.wav"
    canvas.export(str(out_wav), format="wav")
    return out_wav, synced_segments


def build_synced_voice_for_script(
    script_text: str,
    video_duration: float,
    work_dir: Path,
    voice: str = "zh-CN-XiaoxiaoNeural",
    base_rate: str = "+0%",
) -> tuple[Path, list[Segment]]:
    """
    Silent-video narration synthesis with whole-video duration synchronization.
    Automatically adjusts tempo gently if the narration exceeds video duration,
    keeping subtitles and voice locked to the video timeline.
    """
    cleaned = script_text.strip()
    if not cleaned:
        raise ValueError("口播文案不能为空。")

    raw_mp3 = work_dir / "narration_raw.mp3"
    boundaries = _run_async(_edge_tts_generate(cleaned, raw_mp3, voice=voice, rate=base_rate))
    tts_dur = get_audio_duration(raw_mp3)
    if tts_dur <= 0.05:
        raise RuntimeError("配音生成失败，音频时长为 0。")

    if not boundaries:
        boundaries = _fallback_proportional_segments(cleaned, tts_dur)

    speed = 1.0
    if video_duration > 0.5 and tts_dur > video_duration * 0.98:
        speed = min(1.40, tts_dur / (video_duration * 0.96))

    synced_mp3 = work_dir / "narration_synced.mp3"
    if speed > 1.02:
        change_audio_speed(raw_mp3, synced_mp3, speed)
        final_src = synced_mp3
        scale = 1.0 / speed
    else:
        final_src = raw_mp3
        scale = 1.0

    final_dur = get_audio_duration(final_src)
    lead_in = 0.2 if (video_duration - final_dur) > 0.5 else 0.0

    synced_segments = [
        Segment(
            index=i,
            start=round(seg.start * scale + lead_in, 3),
            end=round(min(video_duration, seg.end * scale + lead_in), 3),
            text=seg.text,
        )
        for i, seg in enumerate(boundaries, start=1)
    ]

    clip = AudioSegment.from_file(str(final_src)).set_frame_rate(44100).set_channels(2)
    total_ms = int(round(max(video_duration, final_dur + lead_in) * 1000))
    canvas = AudioSegment.silent(duration=total_ms, frame_rate=44100)
    canvas = canvas.overlay(clip, position=int(round(lead_in * 1000)))

    out_wav = work_dir / "narration_voice.wav"
    canvas.export(str(out_wav), format="wav")
    return out_wav, synced_segments
