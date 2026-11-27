import subprocess
from pathlib import Path

from core.media import extract_audio, get_audio_duration, probe_video
from core.models import RenderOptions, Segment
from core.subtitle import generate_ass_subtitle


def compose_final_video(
    video_path: Path,
    voice_wav_path: Path | None,
    segments: list[Segment],
    output_mp4: Path,
    work_dir: Path,
    options: RenderOptions,
) -> tuple[Path, Path | None]:
    """
    Compose the final MP4:
    - Original video (extending last frame slightly if synced voice is marginally longer)
    - Synced voice track (or original voice if keep_original_voice=True)
    - Optional BGM with dynamic audio ducking (sidechaincompress)
    - Optional crisp ASS burned-in subtitles
    """
    info = probe_video(video_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Determine the voice track to mix
    if options.keep_original_voice and info.has_audio_stream:
        active_voice = work_dir / "original_voice_44k.wav"
        extract_audio(video_path, active_voice, sample_rate=44100, channels=2)
    else:
        if voice_wav_path is None or not voice_wav_path.exists():
            raise ValueError("缺少配音音轨，无法合成视频。")
        active_voice = voice_wav_path

    voice_dur = get_audio_duration(active_voice)
    extra_pad = max(0.0, voice_dur - info.duration)

    ass_path: Path | None = None
    if options.subtitle_enabled and segments:
        ass_path = work_dir / "subtitles.ass"
        generate_ass_subtitle(
            segments=segments,
            output_ass=ass_path,
            video_width=info.width,
            video_height=info.height,
            style=options.subtitle_style,
        )

    # Build video filter chain
    v_filters: list[str] = []
    if extra_pad > 0.08:
        v_filters.append(f"tpad=stop_mode=clone:stop_duration={extra_pad:.3f}")
    if ass_path is not None:
        v_filters.append(f"ass='{ass_path.name}'")

    # Build FFmpeg command
    cmd: list[str] = ["ffmpeg", "-y", "-i", str(video_path.resolve()), "-i", str(active_voice.resolve())]
    has_bgm = options.bgm_path is not None and options.bgm_path.exists() and options.bgm_volume > 0.0
    if has_bgm:
        cmd.extend(["-stream_loop", "-1", "-i", str(options.bgm_path.resolve())])

    filter_parts: list[str] = []
    if v_filters:
        filter_parts.append(f"[0:v]{','.join(v_filters)}[vout]")
        v_map = "[vout]"
    else:
        v_map = "0:v"

    if has_bgm:
        vol = max(0.01, min(1.0, options.bgm_volume))
        if options.bgm_ducking:
            filter_parts.append(
                "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                "asplit=2[voice_main][voice_sc]"
            )
            filter_parts.append(
                f"[2:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"volume={vol:.3f}[bgm_vol]"
            )
            filter_parts.append(
                "[bgm_vol][voice_sc]sidechaincompress="
                "threshold=0.03:ratio=5:attack=40:release=350[bgm_ducked]"
            )
            filter_parts.append(
                "[voice_main][bgm_ducked]amix=inputs=2:duration=first:"
                "dropout_transition=2:normalize=0[aout]"
            )
        else:
            filter_parts.append(
                "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[voice_main]"
            )
            filter_parts.append(
                f"[2:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"volume={vol:.3f}[bgm_vol]"
            )
            filter_parts.append(
                "[voice_main][bgm_vol]amix=inputs=2:duration=first:"
                "dropout_transition=2:normalize=0[aout]"
            )
        a_map = "[aout]"
    else:
        a_map = "1:a"

    if filter_parts:
        cmd.extend(["-filter_complex", ";".join(filter_parts)])

    cmd.extend(
        [
            "-map",
            v_map,
            "-map",
            a_map,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_mp4.resolve()),
        ]
    )

    proc = subprocess.run(
        cmd,
        cwd=str(work_dir.resolve()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg synthesis failed:\n{proc.stderr[-1200:]}")

    return output_mp4, ass_path
