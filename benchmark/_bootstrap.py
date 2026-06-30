"""Import before ``torch`` so libgomp sees a valid ``OMP_NUM_THREADS``."""
from __future__ import annotations

from flash_aurora.aurora._openmp_env import sanitize_openmp_env
from flash_aurora.engine.runtime.cuda_memory import configure_pytorch_cuda_allocator

sanitize_openmp_env()
configure_pytorch_cuda_allocator()
