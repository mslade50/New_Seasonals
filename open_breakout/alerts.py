"""Best-effort alerts: console always, plus the repo's existing Slack webhook when configured.

Posting runs on a daemon thread with a short timeout; it never blocks or raises into the order path.
"""
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

ENV_PATH = Path(__file__).resolve().parents[1] / '.env'


def webhook_url() -> str | None:
    url = os.environ.get('SLACK_WEBHOOK_URL')
    if not url:
        try:
            from dotenv import dotenv_values
            url = dotenv_values(ENV_PATH).get('SLACK_WEBHOOK_URL')
        except Exception:
            url = None
    return url or None


class Alerts:
    def __init__(self, prefix: str, url: str | None = None):
        self.prefix = prefix
        self.url = url
        self.threads: list[threading.Thread] = []

    def flush(self, timeout: float = 5.) -> None:
        """Wait (bounded) for pending posts before the process exits; daemon threads die at exit."""
        deadline = time.monotonic() + timeout
        for thread in self.threads:
            thread.join(max(0., deadline - time.monotonic()))
        self.threads = [t for t in self.threads if t.is_alive()]

    @property
    def channel(self) -> str:
        return 'console+slack' if self.url else 'console'

    def __call__(self, text: str) -> None:
        line = f'[{self.prefix}] {text}'
        print(f'ALERT {line}', flush=True)
        if self.url:
            thread = threading.Thread(target=self._post, args=(line,), daemon=True)
            thread.start()
            self.threads = [t for t in self.threads if t.is_alive()] + [thread]

    def _post(self, text: str) -> None:
        try:
            request = urllib.request.Request(self.url, data=json.dumps({'text': text}).encode('utf-8'), method='POST',
                                             headers={'Content-Type': 'application/json; charset=utf-8'})
            urllib.request.urlopen(request, timeout=5).close()
        except Exception as exc:
            print(f'ALERT delivery failed ({type(exc).__name__}); console only', flush=True)
