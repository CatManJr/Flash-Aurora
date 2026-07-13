# Microsoft Aurora 1.5 (side path)

The ``flash_aurora.models.aurora_v1p5`` Python package is derived from
[Microsoft Aurora](https://github.com/microsoft/aurora) tag ``v2.0.0``
(MIT License). It carries Aurora 1.5 model semantics (expanded variables,
variable lead time, insolation, upstream rollout) without Flash-Aurora
CuTe/Triton/`inference_precision` kernels.

## Required notices (MIT)

When you redistribute source or binaries that include
``flash_aurora.models.aurora_v1p5``, you must:

1. Retain the **Microsoft copyright header** on files that still carry it.
2. Include a copy of **``LICENSE.txt``** in this directory.

## Provenance

| Item | Value |
|------|--------|
| Upstream repository | https://github.com/microsoft/aurora |
| Upstream tag | v2.0.0 |
| Upstream license | MIT (see ``LICENSE.txt``) |
| Copyright | Copyright (c) Microsoft Corporation |
| Shared data types | ``flash_aurora.models.aurora.batch`` (Batch / Metadata) |

## Model weights

Checkpoint files (e.g. on Hugging Face ``microsoft/aurora``) are **not**
part of this source tree and may be subject to separate terms. This notice
covers **source code** only.
