# Microsoft Aurora (upstream)

The ``flash_aurora.models.aurora`` Python package is derived from
[Microsoft Aurora](https://github.com/microsoft/aurora) (MIT License).

## Upstream freeze line

This legacy optimized family is **capped** at Microsoft Aurora tag
[**v1.8.0**](https://github.com/microsoft/aurora/releases/tag/v1.8.0) -- the
last release before **v2.0.0** (Aurora 1.5). Do not sync post-v1.8.0 model
semantics into this package. Aurora 1.5 lives in
``flash_aurora.models.aurora_v1p5``.

## Required notices (MIT)

When you redistribute source or binaries that include ``flash_aurora.models.aurora``,
you must:

1. Retain the **Microsoft copyright header** on files that still carry it.
2. Include a copy of **``LICENSE.txt``** in this directory (or an equivalent
   reproduction of the MIT permission notice above).

## Provenance

| Item | Value |
|------|--------|
| Upstream repository | https://github.com/microsoft/aurora |
| Upstream baseline | v1.8.0 (freeze; pre-Aurora 2.0.0) |
| Upstream license | MIT (see ``LICENSE.txt``) |
| Copyright | Copyright (c) Microsoft Corporation |

## Modifications in Flash-Aurora

Files under ``flash_aurora/models/aurora/`` may include changes by Catman Jr. and
others for inference performance (precision wiring, local checkpoint loading,
etc.). Where a file header names **Catman Jr.**, those portions are under the
MIT license stated in that header; the remainder of Microsoft-authored logic in
the same file remains subject to Microsoft's copyright and this ``LICENSE.txt``.

Shared Triton/CuTe kernels live under ``flash_aurora/models/ops/`` (not inside
this family package). Precision presets live under
``flash_aurora/models/inference_precision.py``. See third-party notices in
individual op files (e.g. NVIDIA BSD-3-Clause in
``ops/cute/_dense_gemm_sm120.py``).

## Model weights

Checkpoint files (e.g. on Hugging Face ``microsoft/aurora``) are **not** part of
this source tree and may be subject to separate terms from Microsoft / ECMWF /
data providers. This notice covers **source code** only.
