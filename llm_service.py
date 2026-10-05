"""Local LLM via Ollama HTTP API. Stdlib only. Model swappable by name."""
import json
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "Kamu Lana, asisten suara lokal. Jawab sesuai bahasa yang dipakai user. "
    "Jawaban ringkas dan jelas karena dibaca sebagai teks di TUI."
)


class LLMOfflineError(RuntimeError):
    pass


class LocalLLM:
    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "llama3.1:8b", timeout: int = 120):
        self.host = host.rstrip("/")
        self.model = model  # ganti kapan saja: llm.model = "qwen2.5:7b"
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise LLMOfflineError(f"Ollama tidak terjangkau di {self.host}: {e}") from e

    def chat(self, messages: list) -> str:
        """messages: [{'role': 'user'|'assistant'|'system', 'content': str}]."""
        res = self._post("/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            "stream": False,
        })
        return (res.get("message") or {}).get("content", "").strip()

    def list_models(self) -> list:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as r:
                data = json.loads(r.read().decode())
            return [m["name"] for m in data.get("models", [])]
        except urllib.error.URLError as e:
            raise LLMOfflineError(f"Ollama tidak terjangkau di {self.host}: {e}") from e
