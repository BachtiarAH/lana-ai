"""Lana AI — single entry point.

TUI (default):   .venv/Scripts/python lana-ai.py
Headless:        .venv/Scripts/python lana-ai.py --headless
UI test:         .venv/Scripts/python lana-ai.py --no-audio
"""
import argparse

from config import Config


def main():
    p = argparse.ArgumentParser(description="lana-ai: mic -> VAD -> STT -> local LLM")
    Config.add_cli_args(p)
    p.add_argument("--headless", action="store_true", help="tanpa TUI, print ke console")
    p.add_argument("--no-audio", action="store_true", help="buka TUI tanpa mic (test)")
    args = p.parse_args()
    cfg = Config.from_args(args)

    llm = None
    if cfg.llm_enabled:
        from llm_service import LocalLLM

        llm = LocalLLM(host=cfg.llm_host, model=cfg.llm_model)

    if args.headless:
        from live_mic import run_pipeline

        if args.offline:
            import os

            os.environ["HF_HUB_OFFLINE"] = "1"
        if cfg.stt_model:
            from stt_service import get_model

            try:
                get_model(cfg.stt_model)
            except Exception as e:
                print(f"[warn] preload stt gagal: {e}")
        try:
            run_pipeline(cfg, stt_offline=args.offline, llm=llm)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        from tui import LanaApp

        LanaApp(cfg, no_audio=args.no_audio).run()


if __name__ == "__main__":
    main()
