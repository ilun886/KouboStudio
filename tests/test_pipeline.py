"""End-to-end smoke test for KouboStudio V1:
1. Silent video -> script -> Edge-TTS + ASS subtitles + BGM ducking -> MP4
2. Voiced video (synthesized Chinese speech) -> Faster-Whisper detection/ASR -> re-dub synced -> MP4
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from core.asr import detect_and_transcribe
from core.composer import compose_final_video
from core.media import probe_video
from core.models import RenderOptions, SubtitleStyle
from core.tts import build_synced_voice_for_script, build_synced_voice_for_segments

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_BGM = ROOT / "assets" / "bgm" / "output000.mp3"


def _make_silent_video(out_mp4: Path, duration: float = 5.0):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x1e293b:s=720x1280:d={duration}:r=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        # --- Test 1: Silent video -> Script -> TTS + Sync + ASS + BGM Ducking ---
        silent_video = work / "silent.mp4"
        _make_silent_video(silent_video, duration=5.0)
        info = probe_video(silent_video)
        assert not info.has_audio_stream, "Expected silent video"

        has_speech, segs, _ = detect_and_transcribe(silent_video, work / "asr_silent", model_size="tiny")
        assert not has_speech and len(segs) == 0, "Silent video should be detected as silent"

        script = "大家好，这是一段自动生成的口播测试视频。音画同步、中文字幕和背景音乐已经全部就绪！"
        voice_wav, synced_segs = build_synced_voice_for_script(
            script_text=script,
            video_duration=info.duration,
            work_dir=work / "silent_tts",
        )
        assert voice_wav.exists() and len(synced_segs) >= 1

        out_silent = work / "out_silent.mp4"
        mp4_path, ass_path = compose_final_video(
            video_path=silent_video,
            voice_wav_path=voice_wav,
            segments=synced_segs,
            output_mp4=out_silent,
            work_dir=work / "silent_compose",
            options=RenderOptions(
                bgm_path=SAMPLE_BGM if SAMPLE_BGM.exists() else None,
                bgm_volume=0.2,
                bgm_ducking=True,
                subtitle_enabled=True,
                subtitle_style=SubtitleStyle(),
            ),
        )
        out_info = probe_video(mp4_path)
        assert out_info.has_audio_stream and out_info.duration >= 4.5
        assert ass_path is not None and ass_path.exists()
        print(f"[PASS] Silent mode E2E -> {mp4_path.name} ({out_info.duration:.2f}s, {len(synced_segs)} subtitle segments)")

        # --- Test 2: Voiced video (use out_silent without BGM as voiced input) -> ASR -> Re-dub Sync ---
        voiced_input = work / "voiced_input.mp4"
        compose_final_video(
            video_path=silent_video,
            voice_wav_path=voice_wav,
            segments=[],
            output_mp4=voiced_input,
            work_dir=work / "voiced_prep",
            options=RenderOptions(bgm_path=None, subtitle_enabled=False),
        )
        has_speech2, asr_segs, lang = detect_and_transcribe(
            voiced_input, work / "asr_voiced", model_size="tiny", language="zh"
        )
        assert has_speech2 and len(asr_segs) >= 1, f"Expected speech detected, got {has_speech2}, {asr_segs}"
        print(f"[PASS] Voiced detection & ASR -> lang={lang}, segments={[(s.start, s.end, s.text) for s in asr_segs]}")

        redub_wav, redub_segs = build_synced_voice_for_segments(
            segments=asr_segs,
            video_duration=probe_video(voiced_input).duration,
            work_dir=work / "redub_tts",
            voice="zh-CN-YunxiNeural",
        )
        out_redub = work / "out_redub.mp4"
        compose_final_video(
            video_path=voiced_input,
            voice_wav_path=redub_wav,
            segments=redub_segs,
            output_mp4=out_redub,
            work_dir=work / "redub_compose",
            options=RenderOptions(
                bgm_path=SAMPLE_BGM if SAMPLE_BGM.exists() else None,
                bgm_volume=0.22,
                bgm_ducking=True,
                subtitle_enabled=True,
            ),
        )
        redub_info = probe_video(out_redub)
        assert redub_info.has_audio_stream and redub_info.duration >= 4.5
        print(f"[PASS] Voiced re-dub E2E -> {out_redub.name} ({redub_info.duration:.2f}s, {len(redub_segs)} synced segments)")


if __name__ == "__main__":
    main()
