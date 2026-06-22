"""Import before ``torch`` so libgomp sees a valid ``OMP_NUM_THREADS``."""
from __future__ import annotations

from flash_aurora.aurora._openmp_env import sanitize_openmp_env

sanitize_openmp_env()
