# THE TRANSLATER SERVER

本地实时翻译：抓取电脑正在播放的声音（视频、会议、直播……），边听边出原文和译文。全部在本机显卡上运行，不联网、不收费。

- 语音识别：[faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo`
- 翻译：[TranslateGemma](https://ollama.com/library/translategemma)（Google 的翻译专用模型，55 种语言），通过 [Ollama](https://ollama.com) 运行

## 效果

仿 [Soniox](https://soniox.com) 的流式思路：

- 每 0.5 秒重新识别当前句子，连续两次一致的部分定稿，其余显示为临时字
- 句子没说完就先出临时译文，随原文更新
- 按句末（。？/ ます・です・か / 英文大写开头的下一句）定稿，避免碎句
- 自动识别语种，中途换语言也能跟上

在 RTX 4070 Ti SUPER 上，短句译文约 0.2–0.4 秒，说话后 0.5–1 秒出临时译文。

## 配置要求

| | 最低 | 推荐 |
|---|---|---|
| 系统 | Windows 10/11 | |
| 显卡 | NVIDIA，8GB 显存（自动用 4B 翻译模型，质量差一些） | NVIDIA，12GB 以上（用 12B 翻译模型） |
| 硬盘 | 约 15GB | |

RTX 3060 12GB 能跑 12B，但流式刷新会慢一些。

## 安装

1. 下载本仓库（`git clone` 或 Code → Download ZIP 解压）
2. 双击 **`install.bat`**

安装脚本会自动：检查显卡、按显存选翻译模型、装 Python 3.12 和 Ollama（如果没有）、下载翻译模型和 Whisper 模型、创建虚拟环境。首次大约需要下载 10GB。

## 使用

双击 **`start.bat`**，会出现一个置顶的「实时翻译」窗口，然后播放任何有声音的内容即可。

- 灰色小字：识别的原文（斜体为临时结果，可能还会变）
- 白色大字：译文（斜体为临时译文，粗体为定稿）

捕获的是 **Windows 默认播放设备** 的声音。换了耳机/音箱需要重启程序。

### 参数

```
start.bat --target en          翻译成英文（可选 zh-Hans / zh-Hant / en / ja / ko，默认简体中文）
start.bat --threshold 0.01     背景噪音大时调高语音能量阈值（默认 0.006）
start.bat --model translategemma:4b   手动指定翻译模型
```

## 常见问题

- **窗口没有字**：确认声音是从 Windows 默认播放设备出来的；命令行窗口里会显示正在捕获的设备名。
- **识别出奇怪的重复句子**：一般是背景音乐/噪音，试试调高 `--threshold`。
- **翻译慢**：用 `nvidia-smi` 看显存是否被其他程序占满，Ollama 里其他大模型可以 `ollama stop <模型名>` 释放。

## 计划

- [ ] 服务端 / 客户端拆分：显卡机器做识别和翻译，其他电脑只负责抓声音和显示字幕
- [ ] 开机自启、断线重连
