import json
import subprocess
from pathlib import Path

from core.models import VideoInfo


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-800:]}")
    return proc


def probe_video(video_path: Path) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    proc = _run_cmd(cmd)
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v_stream:
        raise ValueError(f"No video stream found in {video_path}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    width = int(v_stream.get("width", 1080))
    height = int(v_stream.get("height", 1920))

    # Handle rotation metadata (common for phone vertical videos)
    rotate = 0
    tags = v_stream.get("tags") or {}
    if "rotate" in tags:
        try:
            rotate = abs(int(tags["rotate"])) % 360
        except ValueError:
            rotate = 0
    for side in v_stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                rotate = abs(int(side["rotation"])) % 360
            except ValueError:
                pass
    if rotate in (90, 270):
        width, height = height, width

    fps_str = v_stream.get("r_frame_rate") or v_stream.get("avg_frame_rate") or "30/1"
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 30.0
    except Exception:
        fps = 30.0

    duration = float(fmt.get("duration") or v_stream.get("duration") or 0.0)
    return VideoInfo(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio_stream=has_audio,
    )


def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    proc = _run_cmd(cmd)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio(
    video_path: Path,
    output_wav: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_wav),
    ]
    _run_cmd(cmd)
    return output_wav


def extract_keyframes(
    video_path: Path,
    duration: float,
    output_dir: Path,
    max_frames: int = 8,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = max(1, min(max_frames, int(duration // 2) + 1)) if duration > 0 else 1
    timestamps = (
        [duration * (i + 0.5) / count for i in range(count)]
        if duration > 0.5
        else [0.0]
    )
    frames: list[Path] = []
    for idx, ts in enumerate(timestamps):
        out_img = output_dir / f"frame_{idx:02d}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{ts:.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale='min(768,iw)':-2",
            "-q:v",
            "3",
            str(out_img),
        ]
        try:
            _run_cmd(cmd)
            if out_img.exists() and out_img.stat().st_size > 0:
                frames.append(out_img)
        except Exception:
            continue
    return frames


def change_audio_speed(input_path: Path, output_path: Path, speed: float) -> Path:
    """Adjust audio tempo cleanly while preserving pitch using FFmpeg atempo."""
    speed = max(0.5, min(2.0, speed))
    if abs(speed - 1.0) < 0.015:
        if input_path != output_path:
            output_path.write_bytes(input_path.read_bytes())
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-filter:a",
        f"atempo={speed:.4f}",
        "-vn",
        str(output_path),
    ]
    _run_cmd(cmd)
    return output_path
