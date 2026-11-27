from pathlib import Path
from typing import Optional

from loguru import logger
from pydub import AudioSegment

from core.media import extract_audio, probe_video
from core.models import Segment

_MODEL_CACHE: dict[tuple[str, str], object] = {}
_CUDA_UNAVAILABLE = False


def _load_whisper(model_size: str = "small", force_cpu: bool = False):
    """Load Faster-Whisper, preferring CUDA when usable and falling back to CPU int8."""
    global _CUDA_UNAVAILABLE
    from faster_whisper import WhisperModel

    candidates = [("cpu", "int8")] if (force_cpu or _CUDA_UNAVAILABLE) else [("cuda", "float16"), ("cpu", "int8")]
    for device, compute_type in candidates:
        key = (model_size, device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key], device
        if device == "cuda":
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() <= 0:
                    _CUDA_UNAVAILABLE = True
                    continue
            except Exception:
                _CUDA_UNAVAILABLE = True
                continue
        try:
            logger.info(f"Loading Faster-Whisper model={model_size}, device={device}, compute={compute_type}")
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            _MODEL_CACHE[key] = model
            return model, device
        except Exception as exc:
            logger.warning(f"Failed to init Whisper on {device}: {exc}")
            if device == "cuda":
                _CUDA_UNAVAILABLE = True
            continue
    raise RuntimeError("Unable to initialize Faster-Whisper on either CUDA or CPU.")


def _run_transcribe(wav_path: Path, model_size: str, language: Optional[str]):
    global _CUDA_UNAVAILABLE
    model, device = _load_whisper(model_size=model_size)
    lang_arg = language if language and language != "auto" else None
    prompt_arg = "以下是简体中文口播内容，请使用简体中文和标点符号。" if lang_arg == "zh" else None
    transcribe_kwargs = dict(
        language=lang_arg,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )
    if prompt_arg:
        transcribe_kwargs["initial_prompt"] = prompt_arg
    try:
        segments_iter, trans_info = model.transcribe(str(wav_path), **transcribe_kwargs)
        return list(segments_iter), trans_info
    except RuntimeError as exc:
        if device == "cuda" and ("cublas" in str(exc).lower() or "cudnn" in str(exc).lower() or "cuda" in str(exc).lower()):
            logger.warning(f"CUDA runtime missing ({exc}), falling back to CPU int8.")
            _CUDA_UNAVAILABLE = True
            model, _ = _load_whisper(model_size=model_size, force_cpu=True)
            segments_iter, trans_info = model.transcribe(str(wav_path), **transcribe_kwargs)
            return list(segments_iter), trans_info
        raise


def detect_and_transcribe(
    video_path: Path,
    work_dir: Path,
    model_size: str = "small",
    language: Optional[str] = "zh",
) -> tuple[bool, list[Segment], str]:
    """
    Detect whether a video contains valid human speech and transcribe it with timestamps.

    Returns:
        (has_speech, segments, detected_language)
    """
    info = probe_video(video_path)
    if not info.has_audio_stream:
        logger.info("Video has no audio stream -> Silent mode.")
        return False, [], ""

    wav_path = work_dir / "extracted_16k.wav"
    extract_audio(video_path, wav_path, sample_rate=16000, channels=1)

    # Fast check for completely silent audio tracks
    try:
        audio = AudioSegment.from_wav(str(wav_path))
        if len(audio) < 300 or audio.dBFS == float("-inf") or audio.dBFS < -55.0:
            logger.info(f"Audio track is effectively silent (dBFS={audio.dBFS}) -> Silent mode.")
            return False, [], ""
    except Exception as exc:
        logger.warning(f"Audio level check failed, continuing to Whisper VAD: {exc}")

    raw_segments, trans_info = _run_transcribe(wav_path, model_size=model_size, language=language)

    segments: list[Segment] = []
    total_chars = 0
    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue
        total_chars += len(text)
        segments.append(
            Segment(
                index=len(segments) + 1,
                start=round(float(seg.start), 3),
                end=round(float(seg.end), 3),
                text=text,
            )
        )

    has_speech = len(segments) > 0 and total_chars >= 2
    detected_lang = getattr(trans_info, "language", "") or ""
    logger.info(
        f"ASR result: has_speech={has_speech}, segments={len(segments)}, lang={detected_lang}"
    )
    return has_speech, segments, detected_lang
