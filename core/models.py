from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio_stream: bool


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str


@dataclass
class SubtitleStyle:
    font_name: str = "Microsoft YaHei"
    font_size: int = 56
    primary_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: float = 3.0
    position: Literal["bottom", "middle", "top"] = "bottom"
    margin_v_ratio: float = 0.12


@dataclass
class RenderOptions:
    keep_original_voice: bool = False
    voice_name: str = "zh-CN-XiaoxiaoNeural"
    voice_rate: str = "+0%"
    bgm_path: Path | None = None
    bgm_volume: float = 0.22
    bgm_ducking: bool = True
    subtitle_enabled: bool = True
    subtitle_style: SubtitleStyle = field(default_factory=SubtitleStyle)
