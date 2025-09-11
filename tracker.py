import time
import torch

from utils import print0


class PerformanceLogger:
    def __init__(self, warmup_steps: int = 10, logging_steps: int = 10):
        self.warmup_steps = 10
        self.logging_steps = 10

        self._current_step = 0
        self._is_in_warmup = True

        self._total_tokens = 0

    def _set_warmup_state(self):
        if self._current_step == self.warmup_steps:
            self._start_time = time.perf_counter()
            self._is_in_warmup = False

    def _maybe_log(self):
        if (
            self._current_step % self.logging_steps == 0
            and self._current_step > self.warmup_steps
        ):
            self._last_time = time.perf_counter()
            metrics = self.get_metrics()
            print0(
                f"Step: {self._current_step} | Tokens per second: {metrics['tokens_per_second']} | Iterations per second: {metrics['iterations_per_second']}"
            )

    def mark_step(self, batch: torch.Tensor):
        self._current_step += 1
        self._set_warmup_state()

        if self._is_in_warmup:
            return

        self._total_tokens += batch.size(0) * batch.size(1)

        self._maybe_log()

    def get_metrics(self):
        assert self._current_step > 0, "No steps recorded"
        assert not self._is_in_warmup, "Still in warmup phase"
        return {
            "steps": self._current_step,
            "tokens_per_second": self._total_tokens
            / (self._last_time - self._start_time),
            "iterations_per_second": (self._current_step - self.warmup_steps)
            / (self._last_time - self._start_time),
        }
