# KouboStudio — 口播一键成片 MCP Server (V1 极简版)

专为 AI Agent 打造的轻量级**口播一键成片** MCP 服务：

- **自动判断有声 / 无声**：FFmpeg + Faster-Whisper VAD 人声检测
- **有声视频**：识别带时间戳的简体中文台词 → 保留原声或 Edge-TTS 逐句音画同步重配音
- **无声视频**：抽帧交给视觉大模型（Gemini / OpenAI 兼容接口）写口播词（或由 Agent 直接传入文案）→ 配音并自动对齐视频时长
- **成品合成**：ASS 高对比度中文字幕 + BGM 人声闪避（说话时自动压低背景音）→ 输出 MP4

## 暴露的 MCP 工具（`mcp_server.py`）

- `list_koubo_assets`：列出可用中文音色与内置 BGM
- `analyze_video`：检测视频有声/无声并识别带时间戳台词
- `generate_silent_script`：无声视频看画面生成口播文案
- `produce_koubo_video`：一键合成音画同步成品 MP4

## MCP 接入配置

```json
{
  "mcpServers": {
    "koubo-studio": {
      "command": "d:/Antigravity/KouboStudio/.venv/Scripts/python.exe",
      "args": ["d:/Antigravity/KouboStudio/mcp_server.py"],
      "env": {
        "HF_ENDPOINT": "https://hf-mirror.com"
      }
    }
  }
}
```

## 致谢与来源

本项目受开源项目 [harry0703/MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)（MIT License）启发并参考其音频/字幕/BGM 自动化工作流思路精简改编，专注于**口播一键成片与 MCP Agent 接入**场景。

