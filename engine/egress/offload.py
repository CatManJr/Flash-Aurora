from __future__ import annotations

from aurora import Batch


class CpuOffloader:
    @staticmethod
    def to_cpu(batch: Batch) -> Batch:
        return batch.to("cpu")
