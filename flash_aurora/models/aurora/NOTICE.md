# Microsoft Aurora (upstream)

The ``flash_aurora.aurora`` Python package is derived from
[Microsoft Aurora](https://github.com/microsoft/aurora) (MIT License).

## Required notices (MIT)

When you redistribute source or binaries that include ``flash_aurora.aurora``,
you must:

1. Retain the **Microsoft copyright header** on files that still carry it.
2. Include a copy of **``LICENSE.txt``** in this directory (or an equivalent
   reproduction of the MIT permission notice above).

## Provenance

| Item | Value |
|------|--------|
| Upstream repository | https://github.com/microsoft/aurora |
| Upstream license | MIT (see ``LICENSE.txt``) |
| Copyright | Copyright (c) Microsoft Corporation |

## Modifications in Flash-Aurora

Files under ``flash_aurora/aurora/`` may include changes by Catman Jr. and
others for inference performance (Triton/CuTe kernels, precision presets, local
checkpoint loading, etc.). Where a file header names **Catman Jr.**, those
portions are under the MIT license stated in that header; the remainder of
Microsoft-authored logic in the same file remains subject to Microsoft's
copyright and this ``LICENSE.txt``.

Custom kernel code lives primarily under ``flash_aurora/aurora/ops/``. See also
third-party notices in individual files (e.g. NVIDIA BSD-3-Clause in
``ops/cute/_dense_gemm_sm120.py``).

## Model weights

Checkpoint files (e.g. on Hugging Face ``microsoft/aurora``) are **not** part of
this source tree and may be subject to separate terms from Microsoft / ECMWF /
data providers. This notice covers **source code** only.
