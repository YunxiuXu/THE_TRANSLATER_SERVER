# THE TRANSLATER SERVER

本地实时翻译：抓取电脑正在播放的声音，边听边在置顶窗口里显示原文和译文。全部在本机 NVIDIA 显卡上运行。

- 语音识别：faster-whisper `large-v3-turbo`
- 翻译：Ollama 上的 `translategemma:12b`（显存不足 12GB 可用 `translategemma:4b`）

## 运行

```
ollama pull translategemma:12b
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python live_translate.py [--target zh-Hans|zh-Hant|en|ja|ko] [--model translategemma:12b] [--threshold 0.006]
```

参考环境：Windows 11、Python 3.12、RTX 4070 Ti SUPER 16GB，占用约 11GB 显存，短句译文 0.2–0.4 秒。

## 工作方式

仿 Soniox 的流式思路：每 0.5 秒重新识别当前句子，连续两次一致的词定稿（LocalAgreement），其余显示为临时字；句子没说完就先出临时译文；按句末定稿而不是按停顿切。

## 踩过的坑（适配时注意）

- **WASAPI loopback 必须在同一个线程里创建 PyAudio 和打开流**，否则报 `OSError -9999 Unanticipated host error`。只能捕获 Windows 默认播放设备，换设备要重启。
- **远程桌面（RDP）会话里抓不到声音**：默认设备变成「远程音频」，它的 loopback 永远是静音（实测在它上面播放测试音，loopback 录到 0）。要在本机控制台会话里运行。
- **CUDA DLL**：cuBLAS / cuDNN 来自 pip 包 `nvidia-cublas-cu12`、`nvidia-cudnn-cu12`，脚本开头用 `os.add_dll_directory` 加载，不需要单独装 CUDA Toolkit。
- **Whisper 不能锁语言、不能喂前文 `initial_prompt`**：锁语言会在中途换语种时把英文"听"成日文；喂前文会让一次识别错误被无限复读。
- **Whisper 在停顿处会乱加句号**：英文用"下一句大写开头"判断真句末，日文常不打「。」，改用 ます/です/ください 等句末形式切句（后接 が/けど/ので 的不切）。
- **TranslateGemma 有固定的提示词模板**（见 `Translator._translate`），原文前要空两行；对不完整的句子它会自己加"（此处省略……）"之类的注释，已用正则去掉。
- 翻译模型和 Whisper 要同时放进显存；如果 Ollama 里还加载着别的大模型，先 `ollama stop` 掉，否则会挤到内存里变慢。
