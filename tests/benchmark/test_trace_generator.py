"""Cycle-based job trace generation."""

from __future__ import annotations

from datetime import datetime

import pytest

from _trace_generator import (
    AD_HOC_PRODUCT,
    CYCLE_HOURS_UTC,
    AdHocSpec,
    CycleTraceSpec,
    ProductSpec,
    default_operational_spec,
    generate_cycle_trace,
    generate_homogeneous_burst,
)

_INTERVAL_S = 100.0


def _product(**overrides) -> ProductSpec:
    fields = dict(
        name="p",
        preset="era5_pretrained",
        steps=4,
        cycle_hours=CYCLE_HOURS_UTC,
        ic_ready_delay_s=10.0,
    )
    return ProductSpec(**{**fields, **overrides})


def _trace(*products: ProductSpec, **spec_overrides):
    spec = CycleTraceSpec(products=products, cycle_interval_s=_INTERVAL_S, **spec_overrides)
    return generate_cycle_trace(spec)


def _hour_of(valid_time: str) -> int:
    return datetime.fromisoformat(valid_time).hour


def test_burst_arrives_all_at_once_with_distinct_initial_conditions() -> None:
    trace = generate_homogeneous_burst("era5_pretrained", jobs=5, steps=4)

    assert {job.arrival_s for job in trace} == {0.0}
    assert len({job.valid_time for job in trace}) == 5
    assert len({job.job_id for job in trace}) == 5
    assert {job.preset for job in trace} == {"era5_pretrained"}


@pytest.mark.parametrize("jobs,steps", [(0, 4), (3, 0)])
def test_burst_rejects_empty_or_stepless_jobs(jobs: int, steps: int) -> None:
    with pytest.raises(ValueError):
        generate_homogeneous_burst("era5_pretrained", jobs=jobs, steps=steps)


def test_product_runs_once_per_six_hourly_cycle() -> None:
    trace = _trace(_product())

    assert [job.arrival_s for job in trace] == [10.0, 110.0, 210.0, 310.0]
    assert [_hour_of(job.valid_time) for job in trace] == [0, 6, 12, 18]


def test_product_restricted_to_some_cycles_skips_the_others() -> None:
    trace = _trace(_product(cycle_hours=(0, 12)))

    assert [job.arrival_s for job in trace] == [10.0, 210.0]
    assert [_hour_of(job.valid_time) for job in trace] == [0, 12]


def test_ensemble_fans_out_into_staggered_independent_jobs() -> None:
    trace = _trace(_product(cycle_hours=(0,), members=3, member_stagger_s=0.5))

    assert [job.arrival_s for job in trace] == [10.0, 10.5, 11.0]
    assert [job.member for job in trace] == [0, 1, 2]
    assert len({job.job_id for job in trace}) == 3


def test_single_member_product_has_no_member_index() -> None:
    [job] = _trace(_product(cycle_hours=(0,)))

    assert job.member is None


def test_multi_day_trace_rolls_the_valid_time_over_midnight() -> None:
    trace = _trace(_product(cycle_hours=(0,)), days=2)

    dates = [datetime.fromisoformat(job.valid_time).date() for job in trace]
    assert dates[1] > dates[0]
    assert [job.arrival_s for job in trace] == [10.0, 410.0]


def test_ad_hoc_requests_are_merged_in_arrival_order() -> None:
    trace = _trace(
        _product(cycle_hours=(0, 12)),
        ad_hoc=(AdHocSpec("cams", steps=2, arrival_s=150.0),),
    )

    assert [job.arrival_s for job in trace] == [10.0, 150.0, 210.0]
    assert trace[1].product == AD_HOC_PRODUCT


def test_trace_is_sorted_by_arrival_across_products() -> None:
    trace = _trace(_product(name="late", ic_ready_delay_s=40.0), _product(name="early"))

    arrivals = [job.arrival_s for job in trace]
    assert arrivals == sorted(arrivals)


def test_same_seed_reproduces_the_trace_and_another_seed_changes_it() -> None:
    jittered = dict(ic_jitter_s=20.0)

    first = _trace(_product(), seed=1, **jittered)
    again = _trace(_product(), seed=1, **jittered)
    other = _trace(_product(), seed=2, **jittered)

    assert first == again
    assert first != other


def test_ic_jitter_is_shared_by_all_members_of_a_cycle() -> None:
    trace = _trace(_product(cycle_hours=(0,), members=4), ic_jitter_s=20.0, seed=3)

    assert len({job.arrival_s for job in trace}) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        dict(cycle_hours=(3,)),
        dict(cycle_hours=()),
        dict(steps=0),
        dict(members=0),
        dict(ic_ready_delay_s=-1.0),
    ],
)
def test_product_rejects_invalid_fields(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _product(**overrides)


def test_spec_rejects_duplicate_product_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        CycleTraceSpec(products=(_product(), _product()), cycle_interval_s=_INTERVAL_S)


@pytest.mark.parametrize("overrides", [dict(cycle_interval_s=0.0), dict(days=0), dict(ic_jitter_s=-1.0)])
def test_spec_rejects_invalid_fields(overrides: dict) -> None:
    fields = dict(products=(_product(),), cycle_interval_s=_INTERVAL_S)
    with pytest.raises(ValueError):
        CycleTraceSpec(**{**fields, **overrides})


def test_default_operational_day_covers_the_four_node_presets() -> None:
    trace = generate_cycle_trace(default_operational_spec(_INTERVAL_S, ensemble_members=8))

    presets = {job.preset for job in trace}
    assert presets == {"era5_pretrained", "hres_t0_finetuned", "hres_0.1", "cams"}


def test_default_operational_day_runs_the_heavy_model_at_two_cycles_only() -> None:
    trace = generate_cycle_trace(default_operational_spec(_INTERVAL_S))

    heavy = [job for job in trace if job.preset == "hres_0.1"]
    assert [_hour_of(job.valid_time) for job in heavy] == [0, 12]


def test_default_operational_ensemble_fans_out_at_every_cycle() -> None:
    trace = generate_cycle_trace(default_operational_spec(_INTERVAL_S, ensemble_members=8))

    ensemble = [job for job in trace if job.product == "ens-0.25"]
    assert len(ensemble) == 8 * len(CYCLE_HOURS_UTC)


def test_default_cycle_products_must_finish_before_the_next_cycle() -> None:
    trace = generate_cycle_trace(default_operational_spec(_INTERVAL_S))

    cycle_jobs = [job for job in trace if job.product != AD_HOC_PRODUCT]
    assert {job.deadline_s for job in cycle_jobs} == {_INTERVAL_S}
