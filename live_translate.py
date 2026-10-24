"""实时翻译（流式版）：系统声音 -> Whisper 流式识别 -> TranslateGemma 边说边翻 -> 置顶字幕窗口。

仿 Soniox 的思路：
- 每 0.5s 重新识别当前句子，连续两次一致的词定稿（LocalAgreement），其余显示为临时字
- 句子没说完就先出临时译文，随原文更新
- 按句末（。？/ます/です/か…）定稿，而不是按停顿切，避免碎句

用法: python live_translate.py [--target zh-Hans] [--model translategemma:12b]
"""
import argparse
import os
import queue
import re
import site
import threading
import time

# 让 CTranslate2 找到 pip 装的 cuBLAS / cuDNN DLL
for sp in site.getsitepackages():
    for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
        p = os.path.join(sp, "Lib", "site-packages", sub) if not sp.endswith("site-packages") else os.path.join(sp, sub)
        if os.path.isdir(p):
            os.add_dll_directory(p)
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

import numpy as np
import pyaudiowpatch as pyaudio
import requests
import tkinter as tk
from faster_whisper import WhisperModel

SR = 16000
FRAME = 480  # 30ms @ 16k
STEP = 0.5  # 每隔多久重新识别一次
ENDPOINT = 0.7  # 静音多久算一段话结束
SENTENCE_TIMEOUT = 2.0  # 句子没说完但静音这么久，也强制定稿
MAX_BUF = 15.0  # 识别缓冲区上限（秒）
MAX_SENT = {"cjk": 100, "other": 300}  # 句子超过这么长就在逗号处强制切（字符）
MIN_SENT = {"cjk": 12, "other": 8}  # 句子至少这么长才单独成句（中日韩按字，其他按词）
CJK = ("ja", "zh", "ko")

LANGS = {
    "en": ("English", "en"), "ja": ("Japanese", "ja"), "zh": ("Chinese", "zh"),
    "ko": ("Korean", "ko"), "fr": ("French", "fr"), "de": ("German", "de"),
    "es": ("Spanish", "es"), "ru": ("Russian", "ru"), "it": ("Italian", "it"),
    "pt": ("Portuguese", "pt"), "vi": ("Vietnamese", "vi"), "th": ("Thai", "th"),
    "ar": ("Arabic", "ar"), "id": ("Indonesian", "id"), "hi": ("Hindi", "hi"),
}
TARGETS = {
    "zh-Hans": "Chinese (Simplified)", "zh-Hant": "Chinese (Traditional)",
    "en": "English", "ja": "Japanese", "ko": "Korean",
}
# Whisper 在静音/音乐上常见的幻觉
HALLUCINATIONS = (
    "thank you for watching", "thanks for watching", "ご視聴ありがとうございました",
    "字幕由", "请不吝点赞", "subtitles by", "amara.org",
)
SENT_END = re.compile(r"[。！？!?]+|(?<!\.)\.(?!\.)(?=\s)")  # 省略号不算句末
# 日语 Whisper 常不打句号，用句末形式切；后面接 が/けど/ので 等的是从句，不切
JA_END = re.compile(
    r"(ませんでした|でしょうか|ましたか|ますか|ですか|ました|でした|ません|ください|でしょう|ましょう|ます|です)"
    r"(?!\s*(が|けど|けれど|ので|から|し|と|ね|よ|か|も|は|ま|で|、|。))")
COMPLETE_PAUSE = {"cjk": 0.7, "other": 1.2}  # 句子看起来说完了，静音这么久就定稿
# 句子看起来已经说完（用于停顿时判断要不要定稿）
LOOKS_COMPLETE = re.compile(
    r"(ます|ました|です|でした|ません|ください|でしょう|ましょう|か|ね|よ|[。！？!?.])\s*$")
# 同一段文字（6 字以上）连续出现 3 次，基本是 Whisper 复读幻觉
REPEAT = re.compile(r"(.{6,}?)\1{2,}")
# 模型对不完整句子加的注释
NOTE = re.compile(r"[（(][^）)]*(省略|上下文|原文|未完|不完整|补充|注[:：]|待翻译|为空)[^）)]*[）)]")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="zh-Hans", choices=list(TARGETS))
    ap.add_argument("--model", default="translategemma:12b")
    ap.add_argument("--whisper", default="large-v3-turbo")
    ap.add_argument("--threshold", type=float, default=0.006, help="语音能量阈值")
    return ap.parse_args()


class AudioCapture(threading.Thread):
    """WASAPI loopback 捕获默认扬声器，输出 16k 单声道 float32。"""

    def __init__(self, out_q):
        super().__init__(daemon=True)
        self.out_q = out_q
        pa = pyaudio.PyAudio()
        self.dev = pa.get_default_wasapi_loopback()
        pa.terminate()
        self.rate = int(self.dev["defaultSampleRate"])
        self.ch = self.dev["maxInputChannels"]

    def run(self):
        # WASAPI 要求 PyAudio 和流在同一个线程里创建
        pa = pyaudio.PyAudio()
        chunk = 1024
        stream = pa.open(format=pyaudio.paInt16, channels=self.ch, rate=self.rate,
                         input=True, input_device_index=self.dev["index"], frames_per_buffer=chunk)
        while True:
            data = stream.read(chunk, exception_on_overflow=False)
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            x = x.reshape(-1, self.ch).mean(axis=1)
            n = int(len(x) * SR / self.rate)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
            self.out_q.put(x)


class Translator(threading.Thread):
    """定稿句子优先按顺序翻；空闲时翻当前未完成句子的最新版本（旧的直接丢弃）。"""

    def __init__(self, args, ui_q):
        super().__init__(daemon=True)
        self.args, self.ui_q = args, ui_q
        self.finals = queue.Queue()
        self.lock = threading.Lock()
        self.prov = None  # (sid, text, lang)
        self.wake = threading.Event()

    def final(self, sid, text, lang):
        self.finals.put((sid, text, lang))
        self.wake.set()

    def provisional(self, sid, text, lang):
        with self.lock:
            self.prov = (sid, text, lang)
        self.wake.set()

    def _translate(self, text, lang):
        tgt_code = self.args.target
        if tgt_code.split("-")[0] == lang:
            return text
        tgt_name = TARGETS[tgt_code]
        src_name, src_code = LANGS.get(lang, (lang, lang))
        prompt = (
            f"You are a professional {src_name} ({src_code}) to {tgt_name} ({tgt_code}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the original {src_name} text "
            f"while adhering to {tgt_name} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {tgt_name} translation, without any additional explanations or commentary. "
            f"Please translate the following {src_name} text into {tgt_name}:\n\n\n{text}"
        )
        r = requests.post("http://127.0.0.1:11434/api/chat", timeout=60, json={
            "model": self.args.model, "stream": False, "keep_alive": "30m",
            "messages": [{"role": "user", "content": prompt}],
        })
        return NOTE.sub("", r.json()["message"]["content"]).strip()

    def run(self):
        while True:
            self.wake.wait()
            try:
                sid, text, lang = self.finals.get_nowait()
                t0 = time.time()
                try:
                    tr = self._translate(text, lang)
                except Exception as e:
                    tr = f"[翻译失败: {e}]"
                self.ui_q.put(("final_tr", sid, tr, time.time() - t0))
                continue
            except queue.Empty:
                pass
            with self.lock:
                job, self.prov = self.prov, None
                if job is None and self.finals.empty():
                    self.wake.clear()
            if job:
                try:
                    self.ui_q.put(("prov_tr", job[0], self._translate(job[1], job[2])))
                except Exception:
                    pass


class Streamer(threading.Thread):
    def __init__(self, args, audio_q, ui_q, translator):
        super().__init__(daemon=True)
        self.args, self.audio_q, self.ui_q, self.tr = args, audio_q, ui_q, translator
        self.whisper = WhisperModel(args.whisper, device="cuda", compute_type="float16")
        self.history = ""  # 已定稿的原文，给 Whisper 当提示
        self.sent = ""  # 当前句子已定稿的部分
        self.sent_lang = None
        self.sid = 0  # 当前句子编号
        self.last_emitted = ""
        self.lang_vote = None

    def _asr(self, buf, t0):
        # 不锁语言、不喂前文提示：锁语言会在切换语种时把英文“听”成日文，
        # 喂前文会让一次识别错误被反复复读
        segs, info = self.whisper.transcribe(
            buf, beam_size=1, vad_filter=True, word_timestamps=True,
            condition_on_previous_text=False)
        words = [(w.word, w.start + t0, w.end + t0) for s in segs for w in (s.words or [])]
        text = "".join(w[0] for w in words).lower()
        if any(h in text for h in HALLUCINATIONS) or REPEAT.search(text):
            return [], info.language, 0.0
        return words, info.language, info.language_probability

    def _emit_sentence(self, text):
        text = text.strip()
        if not text or text == self.last_emitted:
            return
        self.last_emitted = text
        lang = self.sent_lang or "en"
        self.ui_q.put(("final_src", self.sid, text, lang))
        self.tr.final(self.sid, text, lang)
        self.history = (self.history + text)[-200:]
        self.sid += 1

    def _kind(self):
        return "cjk" if self.sent_lang in CJK else "other"

    def long_enough(self, text):
        n = len(text.strip()) if self._kind() == "cjk" else len(text.split())
        return n >= MIN_SENT[self._kind()]

    def _is_boundary(self, m):
        # Whisper 在停顿处常自作主张加句号：后面得接上新内容、
        # 西文的下一句得是大写开头、切出去的部分不能太短，才算真的句末
        rest = self.sent[m.end():].lstrip()
        if not rest or not self.long_enough(self.sent[:m.end()]):
            return False
        return self._kind() == "cjk" or rest[0].isupper() or rest[0].isdigit() or rest[0] in "\"'“"

    def _split_sentences(self, finish=False):
        """把 self.sent 里已经结束的句子切出去定稿。"""
        while True:
            ends = list(SENT_END.finditer(self.sent))
            if self.sent_lang == "ja":
                ends = sorted(ends + list(JA_END.finditer(self.sent)), key=lambda m: m.end())
            m = next((m for m in ends if self._is_boundary(m)), None)
            if not m:
                break
            self._emit_sentence(self.sent[:m.end()])
            self.sent = self.sent[m.end():].lstrip()
        if len(self.sent) > MAX_SENT[self._kind()]:
            cut = max(self.sent.rfind("、"), self.sent.rfind(","), self.sent.rfind("，"))
            if cut > 20:
                self._emit_sentence(self.sent[:cut + 1])
                self.sent = self.sent[cut + 1:].lstrip()
        if finish and self.sent:
            self._emit_sentence(self.sent)
            self.sent = ""
        if not self.sent:
            self.sent_lang = None

    def run(self):
        buf = np.zeros(0, np.float32)
        t_end = 0.0  # 已收到音频的绝对时间
        committed_end = 0.0  # 最后一个定稿词的结束时间
        prev = []  # 上一次识别中尚未定稿的词
        heard, silence, since_step = False, 0.0, 0.0
        pending = np.zeros(0, np.float32)
        while True:
            chunks = [self.audio_q.get()]
            while not self.audio_q.empty():
                chunks.append(self.audio_q.get_nowait())
            pending = np.concatenate([pending] + chunks)
            n = len(pending) // FRAME * FRAME
            frames, pending = pending[:n], pending[n:]
            for i in range(0, n, FRAME):
                rms = float(np.sqrt(np.mean(frames[i:i + FRAME] ** 2)))
                if rms > self.args.threshold:
                    heard, silence = True, 0.0
                else:
                    silence += FRAME / SR
            buf = np.concatenate([buf, frames])
            t_end += n / SR
            since_step += n / SR

            if not heard:
                buf = buf[-int(0.3 * SR):]  # 没人说话时只留一点前导音
                if self.sent and (silence > SENTENCE_TIMEOUT or
                                  (silence > COMPLETE_PAUSE[self._kind()] and LOOKS_COMPLETE.search(self.sent)
                                   and self.long_enough(self.sent))):
                    self._split_sentences(finish=True)
                    self._live([])
                continue
            endpoint = silence >= ENDPOINT
            if since_step < STEP and not endpoint:
                continue
            since_step = 0.0

            t0 = t_end - len(buf) / SR
            words, lang, prob = self._asr(buf, t0)
            words = [w for w in words if (w[1] + w[2]) / 2 > committed_end]
            # 连续两次以高置信度判成另一种语言，才认为换语种了，前一句直接定稿
            switched = words and lang != self.sent_lang and prob > 0.8
            if switched and self.sent and self.lang_vote == lang:
                self._split_sentences(finish=True)
            self.lang_vote = lang if switched else None
            if words and not self.sent_lang:
                self.sent_lang = lang
            if endpoint:
                n_commit = len(words)
            else:
                n_commit = 0  # LocalAgreement：与上一次结果相同的前缀才定稿
                while (n_commit < len(words) and n_commit < len(prev)
                       and words[n_commit][0].strip().lower() == prev[n_commit][0].strip().lower()):
                    n_commit += 1
            if n_commit:
                self.sent += "".join(w[0] for w in words[:n_commit])
                if self.sent_lang not in ("ja", "zh"):
                    self.sent = self.sent.lstrip()
                committed_end = words[n_commit - 1][2]
            prev = words[n_commit:]
            self._split_sentences()

            if endpoint:
                # 一段话结束：清空缓冲；要不要定稿交给上面的静音判断，期间有新的话就接上
                buf, prev, heard = np.zeros(0, np.float32), [], False
                committed_end = t_end
            elif len(buf) / SR > MAX_BUF:
                keep = max(committed_end, t_end - MAX_BUF / 2)
                buf = buf[-int((t_end - keep) * SR):]
            self._live(prev)

    def _live(self, nonfinal_words):
        nonfinal = "".join(w[0] for w in nonfinal_words)
        self.ui_q.put(("live", self.sid, self.sent, nonfinal))
        full = (self.sent + nonfinal).strip()
        if full:
            self.tr.provisional(self.sid, full, self.sent_lang or "en")


class SubtitleWindow:
    def __init__(self, ui_q):
        self.ui_q = ui_q
        self.root = tk.Tk()
        self.root.title("实时翻译")
        self.root.geometry("900x420+200+500")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self.root.configure(bg="#111")
        self.text = tk.Text(self.root, bg="#111", fg="#eee", wrap="word", bd=0,
                            padx=16, pady=12, font=("Microsoft YaHei UI", 15))
        self.text.pack(fill="both", expand=True)
        f = "Microsoft YaHei UI"
        self.text.tag_config("orig", foreground="#9aa", font=(f, 12))
        self.text.tag_config("lang", foreground="#5a8", font=("Consolas", 10))
        self.text.tag_config("trans", foreground="#fff", font=(f, 16, "bold"))
        self.text.tag_config("meta", foreground="#566", font=("Consolas", 9))
        self.text.tag_config("live_src", foreground="#bcc", font=(f, 12))
        self.text.tag_config("live_nf", foreground="#677", font=(f, 12, "italic"))
        self.text.tag_config("live_tr", foreground="#cdd", font=(f, 16, "italic"))
        self.text.insert("end", "正在监听系统声音……播放任何视频/音频试试\n\n", "meta")
        self.text.mark_set("live_start", "end-1c")
        self.text.mark_gravity("live_start", "left")
        self.live = (0, "", "")  # (sid, 定稿部分, 临时部分)
        self.live_tr = (-1, "")
        self.tr_marks = {}
        self.root.after(50, self.poll)

    def _render_live(self):
        self.text.delete("live_start", "end-1c")
        sid, src, nf = self.live
        if not (src or nf):
            return
        self.text.insert("end-1c", src, "live_src")
        self.text.insert("end-1c", nf + "\n", "live_nf")
        if self.live_tr[0] == sid and self.live_tr[1]:
            self.text.insert("end-1c", self.live_tr[1], "live_tr")

    def poll(self):
        changed = False
        try:
            while True:
                msg = self.ui_q.get_nowait()
                kind = msg[0]
                changed = True
                if kind == "live":
                    self.live = msg[1:]
                elif kind == "prov_tr":
                    self.live_tr = (msg[1], msg[2])
                elif kind == "final_src":
                    _, sid, text, lang = msg
                    self.text.delete("live_start", "end-1c")
                    self.text.insert("end-1c", f"[{lang}] ", "lang")
                    self.text.insert("end-1c", text + "\n\n\n", "orig")
                    mark = f"tr{sid}"
                    self.text.mark_set(mark, "end-3c")
                    self.text.mark_gravity(mark, "right")
                    self.tr_marks[sid] = mark
                    self.text.mark_set("live_start", "end-1c")
                    self.text.mark_gravity("live_start", "left")
                    if self.live[0] == sid:
                        self.live = (sid + 1, "", "")
                elif kind == "final_tr":
                    _, sid, tr, dt = msg
                    mark = self.tr_marks.pop(sid, None)
                    if mark:
                        self.text.insert(mark, tr, "trans")
                        self.text.insert(mark, f"  [{dt:.1f}s]", "meta")
                        self.text.mark_unset(mark)
        except queue.Empty:
            pass
        if changed:
            self._render_live()
            self.text.see("end")
        self.root.after(50, self.poll)


def main():
    args = parse_args()
    audio_q, ui_q = queue.Queue(), queue.Queue()
    print("加载 Whisper 模型……", flush=True)
    translator = Translator(args, ui_q)
    streamer = Streamer(args, audio_q, ui_q, translator)
    # 预热翻译模型，避免第一句等加载
    requests.post("http://127.0.0.1:11434/api/generate",
                  json={"model": args.model, "prompt": "", "keep_alive": "30m"}, timeout=120)
    cap = AudioCapture(audio_q)
    print(f"捕获设备: {cap.dev['name']}  ({cap.rate}Hz, {cap.ch}ch)", flush=True)
    cap.start()
    translator.start()
    streamer.start()
    win = SubtitleWindow(ui_q)
    win.root.mainloop()


if __name__ == "__main__":
    main()
