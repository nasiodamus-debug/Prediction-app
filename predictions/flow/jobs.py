"""Long work that runs OFF the Telegram update queue (AD-22).

python-telegram-bot handles updates one after the other. A handler that awaits a minute of SportyBet pages, stats fetches and
analyst calls therefore holds every later button press in the queue until it is done, and by then Telegram no longer accepts
the answer ("Query is too old"). `BackgroundJobs` lets a handler start that work and return at once:

  * single-flight per key: a second request with the same key while one runs is refused (`start` returns None);
  * a failure inside a job is handed to `on_error` (and logged), it never escapes into the event loop;
  * every job logs how long it took, at INFO, with the key only (no user text, no secrets);
  * `cancel_all` stops everything cleanly at shutdown and `wait` lets tests (and shutdown) wait for the running jobs.

Standard library only; nothing here knows about Telegram, SportyBet or SQLite."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


class BackgroundJobs:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._tasks: dict[str, asyncio.Task] = {}

    def running(self, key: str) -> bool:
        task = self._tasks.get(key)
        return task is not None and not task.done()

    def active_keys(self) -> list[str]:
        return [key for key in self._tasks if self.running(key)]

    def start(self, key: str, work: Callable[[], Awaitable[None]], label: str = "",
              on_error: Callable[[Exception], Awaitable[None]] | None = None) -> asyncio.Task | None:
        """Run `work()` in the background. Returns None (and starts nothing) when `key` is already running."""
        if self.running(key):
            return None
        task = asyncio.get_running_loop().create_task(self._run(key, work, label or key, on_error))
        self._tasks[key] = task
        task.add_done_callback(lambda done, key=key: self._forget(key, done))
        return task

    def _forget(self, key: str, task: asyncio.Task) -> None:
        if self._tasks.get(key) is task:
            del self._tasks[key]

    async def _run(self, key, work, label, on_error) -> None:
        started = self._clock()
        try:
            await work()
        except asyncio.CancelledError:
            log.info("%s was cancelled after %.1f s", label, self._clock() - started)
            raise
        except Exception as error:  # never let a job crash the bot; the owner is told through on_error
            log.error("%s failed after %.1f s", label, self._clock() - started, exc_info=error)
            if on_error is not None:
                try:
                    await on_error(error)
                except Exception as second:
                    log.warning("Could not report the failure of %s: %s", label, type(second).__name__)
            return
        log.info("%s took %.1f s", label, self._clock() - started)

    async def wait(self) -> None:
        """Wait until nothing is running (jobs started by running jobs are waited for too)."""
        while True:
            pending = [t for t in self._tasks.values() if not t.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def cancel_all(self) -> None:
        pending = [t for t in self._tasks.values() if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
