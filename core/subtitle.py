from pathlib import Path

from core.models import Segment, SubtitleStyle


def _hex_to_ass_color(hex_color: str, alpha: int = 0) -> str:
    """Convert #RRGGBB to ASS &HAABBGGRR format."""
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _wrap_text(text: str, max_chars_per_line: int) -> str:
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    # Remove trailing Chinese punctuation for cleaner short-video look
    text = text.rstrip("，。、,.")
    if len(text) <= max_chars_per_line:
        return text

    lines: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if len(buf) >= max_chars_per_line:
            lines.append(buf)
            buf = ""
        elif ch in "，。！？；,!?" and len(buf) >= int(max_chars_per_line * 0.6):
            lines.append(buf.rstrip("，。、,."))
            buf = ""
    if buf.strip():
        lines.append(buf.strip())
    return r"\N".join(lines[:3])


def _escape_ass(text: str) -> str:
    return text.replace("{", r"\{").replace("}", r"\}")


def generate_ass_subtitle(
    segments: list[Segment],
    output_ass: Path,
    video_width: int,
    video_height: int,
    style: SubtitleStyle,
) -> Path:
    """Generate a crisp, short-video-ready ASS subtitle file adapted to video aspect ratio."""
    output_ass.parent.mkdir(parents=True, exist_ok=True)

    is_vertical = video_height >= video_width
    max_chars = 14 if is_vertical else 24

    # Scale font size proportionally to 1080p baseline so subtitles look consistent across resolutions
    base_ref = min(video_width, video_height)
    scaled_font_size = max(24, int(round(style.font_size * (base_ref / 1080.0))))
    margin_v = int(round(video_height * style.margin_v_ratio))
    margin_lr = int(round(video_width * 0.06))

    alignment = {"bottom": 2, "middle": 5, "top": 8}.get(style.position, 2)
    primary = _hex_to_ass_color(style.primary_color, alpha=0)
    outline = _hex_to_ass_color(style.outline_color, alpha=0)
    shadow = _hex_to_ass_color("#000000", alpha=0x60)

    header = f"""[Script Info]
Title: KouboStudio Subtitles
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{scaled_font_size},{primary},{primary},{outline},{shadow},-1,0,0,0,100,100,1,0,1,{style.outline_width:.1f},1.5,{alignment},{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for seg in segments:
        txt = seg.text.strip()
        if not txt or seg.end <= seg.start:
            continue
        start_str = _format_ass_time(seg.start)
        end_str = _format_ass_time(seg.end)
        wrapped = _wrap_text(_escape_ass(txt), max_chars_per_line=max_chars)
        events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{wrapped}")

    output_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")
    return output_ass
