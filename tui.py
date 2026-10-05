"""Lana TUI: status mic | transkrip | jawaban local LLM.

Run:  .venv/Scripts/python tui.py [--llm-model qwen2.5:7b] [--no-audio]
Keys: q quit · /model <nama> ganti model · /models list · /clear reset history
"""
import argparse
import threading
import time
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, Header, Input, ProgressBar, RichLog, Static

from config import Config
from live_mic import run_pipeline


class StateMsg(Message):
    def __init__(self, state: str):
        super().__init__()
        self.state = state


class TranscriptMsg(Message):
    def __init__(self, text: str):
        super().__init__()
        self.text = text


class AnswerMsg(Message):
    def __init__(self, text: str):
        super().__init__()
        self.text = text


class ErrorMsg(Message):
    def __init__(self, text: str):
        super().__init__()
        self.text = text


class TUIEvents:
    def __init__(self, app: "LanaApp"):
        self.app = app
        self._last_meter = 0.0

    def _post(self, msg: Message):
        self.app.call_from_thread(self.app.post_message, msg)

    def state(self, s: str):
        self._post(StateMsg(s))

    def meter(self, level: float):
        now = time.monotonic()
        if now - self._last_meter > 0.1:  # throttle biar pump tidak kebanjiran
            self._last_meter = now
            self.app.call_from_thread(self.app.set_meter, min(1.0, level * 8))

    def saved(self, path, dur: float):
        pass  # tercatat saat transkrip tiba

    def transcript(self, path, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._post(TranscriptMsg(f"[dim]{ts}[/]  {text or '(tidak jelas)'}"))

    def answer(self, text: str):
        self._post(AnswerMsg(text))

    def error(self, msg: str):
        self._post(ErrorMsg(msg))


class LanaApp(App):
    CSS = """
    #status { height: 3; border: solid green; }
    #panels { height: 1fr; }
    #transcript, #answer { width: 1fr; border: solid grey; }
    #cmd { height: 3; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, cfg: Config, no_audio: bool = False):
        super().__init__()
        self.cfg = cfg
        self.no_audio = no_audio
        self.history: list = []
        self.llm = None
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("starting…", id="status")
        with Horizontal(id="panels"):
            with Vertical(id="transcript"):
                yield Static("— transkrip —")
                yield RichLog(id="log_hear", markup=True)
            with Vertical(id="answer"):
                yield Static("— lana —")
                yield RichLog(id="log_say", markup=True)
        yield ProgressBar(id="meter", total=100, show_eta=False)
        yield Input(placeholder="/model <nama> · /models · /clear · q keluar", id="cmd")
        yield Footer()

    def on_mount(self):
        if self.cfg.llm_enabled:
            from llm_service import LocalLLM

            try:
                self.llm = LocalLLM(host=self.cfg.llm_host, model=self.cfg.llm_model)
                names = self.llm.list_models()
                self.log_line(f"ollama: {len(names)} model ({', '.join(names[:5])})")
            except Exception as e:
                self.log_line(f"[red]llm off: {e}[/]")
                self.llm = None
        if self.no_audio:
            self.set_status("listening (no-audio test mode)")
            return
        self.worker = threading.Thread(target=run_pipeline, kwargs={
            "cfg": self.cfg, "events": TUIEvents(self),
            "stop": self.stop, "llm": self.llm, "history": self.history,
        }, daemon=True)
        self.worker.start()

    def on_unmount(self):
        self.stop.set()

    # -- UI helpers (main thread) --
    def set_status(self, s: str):
        colors = {"speaking": "yellow", "transcribing": "cyan", "thinking": "magenta"}
        c = next((v for k, v in colors.items() if k in s), "green")
        self.query_one("#status", Static).update(f"[{c}]{s}[/]")

    def set_meter(self, level: float):
        self.query_one("#meter", ProgressBar).update(progress=int(level * 100))

    def log_line(self, text: str):
        self.query_one("#log_say", RichLog).write(text)

    # -- messages from pipeline thread --
    def on_state_msg(self, m: StateMsg):
        self.set_status(m.state)

    def on_transcript_msg(self, m: TranscriptMsg):
        self.query_one("#log_hear", RichLog).write(m.text)

    def on_answer_msg(self, m: AnswerMsg):
        self.query_one("#log_say", RichLog).write(m.text)

    def on_error_msg(self, m: ErrorMsg):
        self.query_one("#log_say", RichLog).write(f"[red]{m.text}[/]")

    # -- commands --
    def on_input_submitted(self, e: Input.Submitted):
        cmd = e.value.strip()
        e.input.clear()
        if cmd.startswith("/model "):
            name = cmd.split(maxsplit=1)[1]
            if self.llm is None:
                from llm_service import LocalLLM

                self.llm = LocalLLM(host=self.cfg.llm_host, model=name)
            else:
                self.llm.model = name
            self.cfg.llm_model = name
            self.log_line(f"model → {name}")
        elif cmd == "/models":
            if self.llm is None:
                self.log_line("[red]llm off[/]")
            else:
                try:
                    self.log_line(", ".join(self.llm.list_models()) or "(kosong)")
                except Exception as ex:
                    self.log_line(f"[red]{ex}[/]")
        elif cmd == "/clear":
            self.history.clear()
            self.log_line("(history dibersihkan)")
        elif cmd:
            self.log_line(f"(unknown: {cmd})")


def main():
    p = argparse.ArgumentParser(description="lana-ai TUI")
    Config.add_cli_args(p)
    p.add_argument("--no-audio", action="store_true", help="buka UI tanpa mic (test)")
    args = p.parse_args()
    cfg = Config.from_args(args)
    LanaApp(cfg, no_audio=args.no_audio).run()


if __name__ == "__main__":
    main()
