"""Deterministic synthetic job traces for the job-level scheduler benchmark.

Weather forecasting is not a request stream with random arrivals. Operational
centres run a few independent products at fixed 6-hourly analysis cycles
(00/06/12/18 UTC), each starting once its initial conditions have landed, with
ensembles fanning out into simultaneous independent members. The traces here
follow that shape, with no Poisson process.

The traces are SYNTHETIC. Cycle times, IC availability delays and deadlines
are modelled on public NWP schedules, not measured from any centre. Verify and
cite the operational figures before the paper presents them as realistic.

All times are wall-clock seconds since the trace start. ``cycle_interval_s``
compresses the real 6 h between cycles, while job service times stay real, so
it sets how hard the trace loads the workers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

CYCLE_HOURS_UTC = (0, 6, 12, 18)
_HOURS_BETWEEN_CYCLES = 6
_CYCLES_PER_DAY = len(CYCLE_HOURS_UTC)
_DEFAULT_START_DATE = date(2024, 6, 1)
AD_HOC_PRODUCT = "ad_hoc"
BURST_PRODUCT = "burst"


@dataclass(frozen=True)
class TraceJob:
    """One independent forecast job the replayer submits at ``arrival_s``."""

    job_id: str
    product: str
    preset: str
    steps: int
    arrival_s: float
    valid_time: str
    deadline_s: float | None  # relative to arrival; None means no deadline
    member: int | None = None


@dataclass(frozen=True)
class ProductSpec:
    """A forecast product launched at chosen cycles, as ``members`` parallel jobs."""

    name: str
    preset: str
    steps: int
    cycle_hours: tuple[int, ...]
    ic_ready_delay_s: float
    members: int = 1
    member_stagger_s: float = 0.0
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"product {self.name!r}: steps must be >= 1")
        if self.members < 1:
            raise ValueError(f"product {self.name!r}: members must be >= 1")
        if not self.cycle_hours or not set(self.cycle_hours) <= set(CYCLE_HOURS_UTC):
            raise ValueError(
                f"product {self.name!r}: cycle_hours must be a non-empty subset of {CYCLE_HOURS_UTC}"
            )
        if self.ic_ready_delay_s < 0 or self.member_stagger_s < 0:
            raise ValueError(f"product {self.name!r}: delays must be >= 0")


@dataclass(frozen=True)
class AdHocSpec:
    """A sparse one-off request, e.g. a researcher's on-demand run."""

    preset: str
    steps: int
    arrival_s: float
    deadline_s: float | None = None


@dataclass(frozen=True)
class CycleTraceSpec:
    products: tuple[ProductSpec, ...]
    cycle_interval_s: float
    days: int = 1
    start_date: date = _DEFAULT_START_DATE
    ad_hoc: tuple[AdHocSpec, ...] = ()
    ic_jitter_s: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.cycle_interval_s <= 0:
            raise ValueError("cycle_interval_s must be > 0")
        if self.days < 1:
            raise ValueError("days must be >= 1")
        if self.ic_jitter_s < 0:
            raise ValueError("ic_jitter_s must be >= 0")
        names = [product.name for product in self.products]
        if len(set(names)) != len(names):
            raise ValueError(f"product names must be unique, got {names}")


def generate_cycle_trace(spec: CycleTraceSpec) -> tuple[TraceJob, ...]:
    """Expand every product at every cycle it runs, plus ad-hoc requests, in arrival order."""
    rng = random.Random(spec.seed)
    jobs: list[TraceJob] = []
    for cycle_index in range(spec.days * _CYCLES_PER_DAY):
        cycle_start = _cycle_start(spec.start_date, cycle_index)
        for product in spec.products:
            if cycle_start.hour not in product.cycle_hours:
                continue
            # The IC lands once per product and cycle; every member waits for it.
            ic_ready_s = product.ic_ready_delay_s + rng.uniform(0.0, spec.ic_jitter_s)
            arrival_base_s = cycle_index * spec.cycle_interval_s + ic_ready_s
            jobs.extend(_product_jobs(product, cycle_start, arrival_base_s))
    jobs.extend(_ad_hoc_jobs(spec))
    return _in_arrival_order(jobs)


def generate_homogeneous_burst(
    preset: str,
    *,
    jobs: int,
    steps: int,
    deadline_s: float | None = None,
    start: datetime = datetime.combine(_DEFAULT_START_DATE, time(0)),
) -> tuple[TraceJob, ...]:
    """``jobs`` independent runs of one preset that all arrive at t=0, each on its own IC."""
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return tuple(
        TraceJob(
            job_id=f"{BURST_PRODUCT}-{preset}-{index:03d}",
            product=BURST_PRODUCT,
            preset=preset,
            steps=steps,
            arrival_s=0.0,
            valid_time=(start + timedelta(hours=_HOURS_BETWEEN_CYCLES * index)).isoformat(),
            deadline_s=deadline_s,
        )
        for index in range(jobs)
    )


# Production presets with a local ingress cache. ``small_pretrained`` is the
# 400x800 test grid. ``wave`` has a checkpoint but no MARS cache on this machine.
EXPENSIVE_PRESETS: tuple[str, ...] = ("hres_0.1", "aurora_v1p5_ensemble")
ROTATING_PRESETS: tuple[str, ...] = (
    "era5_pretrained",
    "hres_t0_finetuned",
    "cams",
    "tc_tracking",
    "aurora_v1p5",
)
SCHEDULER_TRACE_PRESETS: tuple[str, ...] = EXPENSIVE_PRESETS + ROTATING_PRESETS
_ROTATION_ROUNDS = 2


def default_operational_spec(
    cycle_interval_s: float,
    *,
    ensemble_members: int = 8,
    days: int = 1,
) -> CycleTraceSpec:
    """One heterogeneous day: the four presets of the 4-GPU node, one product each.

    Every product must finish before the next cycle's inputs arrive, so its
    deadline is one ``cycle_interval_s``. The 0.1 degree and air-quality products run
    only at 00/12 UTC. Ensemble size, delays and ad-hoc arrivals are stand-ins.
    """
    interval = cycle_interval_s
    products = (
        ProductSpec("ens-0.25", "era5_pretrained", 4, CYCLE_HOURS_UTC, 0.10 * interval,
                    members=ensemble_members, deadline_s=interval),
        ProductSpec("ft-0.25", "hres_t0_finetuned", 4, CYCLE_HOURS_UTC, 0.10 * interval,
                    deadline_s=interval),
        ProductSpec("hres-0.1", "hres_0.1", 2, (0, 12), 0.15 * interval, deadline_s=interval),
        ProductSpec("cams-0.4", "cams", 4, (0, 12), 0.20 * interval, deadline_s=interval),
    )
    ad_hoc = (
        AdHocSpec("era5_pretrained", 4, arrival_s=0.55 * interval),
        AdHocSpec("cams", 4, arrival_s=2.30 * interval),
    )
    return CycleTraceSpec(
        products=products,
        cycle_interval_s=cycle_interval_s,
        days=days,
        ad_hoc=ad_hoc,
    )


def group_b_trace(
    cycle_interval_s: float,
    *,
    ensemble_members: int = 8,
    rotation_rounds: int = _ROTATION_ROUNDS,
) -> tuple[TraceJob, ...]:
    """Expensive presets first, then the five smaller presets round-robin.

    ``hres_0.1`` and ``aurora_v1p5_ensemble`` are submitted at t=0 and stay on
    their own GPUs. The other five arrive at ``cycle_interval_s`` and again one
    interval later, so two pool GPUs have a queue to schedule. Each rotating
    preset appears ``rotation_rounds`` times. The trace clock is synthetic; the
    replayer still reads the local cached analysis.
    """
    if cycle_interval_s <= 0:
        raise ValueError("cycle_interval_s must be > 0")
    if ensemble_members < 1 or rotation_rounds < 1:
        raise ValueError("ensemble_members and rotation_rounds must be >= 1")
    jobs: list[TraceJob] = [
        TraceJob(
            job_id="expensive-hres_0.1",
            product="hres-0.1",
            preset="hres_0.1",
            steps=2,
            arrival_s=0.0,
            valid_time="2024-06-01T00:00:00",
            deadline_s=cycle_interval_s,
        )
    ]
    jobs.extend(
        TraceJob(
            job_id=f"expensive-ens15-m{member:02d}",
            product="ens15",
            preset="aurora_v1p5_ensemble",
            steps=4,
            arrival_s=0.0,
            valid_time="2024-06-01T00:00:00",
            deadline_s=cycle_interval_s,
            member=member,
        )
        for member in range(ensemble_members)
    )
    for round_index in range(rotation_rounds):
        round_start_s = cycle_interval_s * (1 + round_index)
        for preset in ROTATING_PRESETS:
            jobs.append(
                TraceJob(
                    job_id=f"rotate-r{round_index}-{preset}",
                    product="rotate",
                    preset=preset,
                    steps=4,
                    arrival_s=round_start_s,
                    valid_time="2024-06-01T06:00:00",
                    deadline_s=cycle_interval_s,
                )
            )
    return _in_arrival_order(jobs)


def _cycle_start(start_date: date, cycle_index: int) -> datetime:
    midnight = datetime.combine(start_date, time(0))
    return midnight + timedelta(hours=_HOURS_BETWEEN_CYCLES * cycle_index)


def _product_jobs(
    product: ProductSpec,
    cycle_start: datetime,
    arrival_base_s: float,
) -> list[TraceJob]:
    cycle_label = cycle_start.strftime("%Y%m%d%H")
    return [
        TraceJob(
            job_id=_product_job_id(product, cycle_label, member),
            product=product.name,
            preset=product.preset,
            steps=product.steps,
            arrival_s=arrival_base_s + member * product.member_stagger_s,
            valid_time=cycle_start.isoformat(),
            deadline_s=product.deadline_s,
            member=member if product.members > 1 else None,
        )
        for member in range(product.members)
    ]


def _product_job_id(product: ProductSpec, cycle_label: str, member: int) -> str:
    base = f"{product.name}-{cycle_label}"
    return base if product.members == 1 else f"{base}-m{member:02d}"


def _ad_hoc_jobs(spec: CycleTraceSpec) -> list[TraceJob]:
    valid_time = datetime.combine(spec.start_date, time(0)).isoformat()
    return [
        TraceJob(
            job_id=f"{AD_HOC_PRODUCT}-{index:03d}-{request.preset}",
            product=AD_HOC_PRODUCT,
            preset=request.preset,
            steps=request.steps,
            arrival_s=request.arrival_s,
            valid_time=valid_time,
            deadline_s=request.deadline_s,
        )
        for index, request in enumerate(spec.ad_hoc)
    ]


def _in_arrival_order(jobs: list[TraceJob]) -> tuple[TraceJob, ...]:
    return tuple(sorted(jobs, key=lambda job: (job.arrival_s, job.job_id)))
