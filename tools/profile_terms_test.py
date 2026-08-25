"""Tests for the per-component step profiler.

Like ``benchmark_test.py``, these guard a *methodology*: a bug in the HLO parse
or the attribution join does not crash, it silently charges a term's cost to
the wrong term. Every test here works on fixed synthetic HLO text and synthetic
trace events, so it needs neither a GPU nor a model.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import profile_terms as pt  # noqa: E402

# A miniature but structurally faithful dump: an entry computation whose
# kernels are fusions, plus the fused computations carrying the scope
# metadata. Mirrors what XLA emits for a step -- including the case that makes
# attribution hard, ``mixed_fusion``, whose two operands come from different
# scopes.
HLO = """HloModule jit__run_from_state, entry_computation_layout={()->()}

%fused_conv (param_0: f32[4]) -> f32[4] {
  %param_0 = f32[4]{0} parameter(0)
  %m = f32[4]{0} multiply(%param_0, %param_0), metadata={op_name="jit(_run_from_state)/scan/jcm:bridge_to_dynamics/jcm:tiedtke_convection/mul"}
  ROOT %a = f32[4]{0} add(%m, %m), metadata={op_name="jit(_run_from_state)/scan/jcm:bridge_to_dynamics/jcm:tiedtke_convection/add"}
}

%fused_mixed (param_0: f32[4]) -> f32[4] {
  %param_1 = f32[4]{0} parameter(0)
  %r = f32[4]{0} sine(%param_1), metadata={op_name="jit(_run_from_state)/scan/jcm:bridge_to_dynamics/jcm:rrtmgp_radiation/sin"}
  ROOT %c = f32[4]{0} cosine(%r), metadata={op_name="jit(_run_from_state)/scan/jcm:bridge_to_dynamics/jcm:tiedtke_convection/cos"}
}

%fused_dyn (param_0: f32[4]) -> f32[4] {
  %param_2 = f32[4]{0} parameter(0)
  ROOT %d = f32[4]{0} negate(%param_2), metadata={op_name="jit(_run_from_state)/scan/jcm:dynamics/neg"}
}

ENTRY %main (p: f32[4]) -> f32[4] {
  %p = f32[4]{0} parameter(0)
  %conv_fusion = f32[4]{0} fusion(%p), kind=kLoop, calls=%fused_conv, metadata={op_name="jit(_run_from_state)/scan/jcm:bridge_to_dynamics/jcm:tiedtke_convection/add"}
  %mixed_fusion = f32[4]{0} fusion(%conv_fusion), kind=kLoop, calls=%fused_mixed, metadata={op_name="jit(_run_from_state)/scan/jcm:bridge_to_dynamics/jcm:tiedtke_convection/cos"}
  %dyn_fusion = f32[4]{0} fusion(%mixed_fusion), kind=kLoop, calls=%fused_dyn, metadata={op_name="jit(_run_from_state)/scan/jcm:dynamics/neg"}
  %copy = f32[4]{0} copy(%dyn_fusion)
  ROOT %bridge = f32[4]{0} tanh(%copy), metadata={op_name="jit(_run_from_state)/scan/jcm:bridge_to_physics/tanh"}
}
"""


@pytest.fixture
def parsed():
    """Parse the fixture module into (name, instructions, members)."""
    module, instructions = pt.parse_hlo_module(HLO)
    return module, instructions, pt.parse_hlo_module_computations(HLO)


def test_parse_module_name_and_instructions(parsed):
    module, instructions, _ = parsed
    assert module == "jit__run_from_state"
    # Instructions from every computation share one namespace.
    assert {"m", "a", "r", "c", "d", "conv_fusion", "dyn_fusion",
            "copy", "bridge"} <= set(instructions)


def test_parse_extracts_calls_edges(parsed):
    _, instructions, _ = parsed
    assert instructions["conv_fusion"].calls == ("fused_conv",)
    # A plain instruction calls nothing.
    assert instructions["copy"].calls == ()


def test_computation_membership(parsed):
    _, _, members = parsed
    # Parameters are members too. They carry no metadata, so they contribute
    # no label and are harmless to the mixed-kernel check.
    assert members["fused_conv"] == ["param_0", "m", "a"]
    assert set(members["main"]) == {
        "p", "conv_fusion", "mixed_fusion", "dyn_fusion", "copy", "bridge",
    }


def test_label_extraction_is_innermost():
    """A term nested inside the enclosing physics scope wins over it."""
    from jcm import profiling
    assert profiling.label_from_op_name(
        "jit(f)/scan/jcm:bridge_to_dynamics/jcm:tiedtke_convection/mul"
    ) == "tiedtke_convection"
    assert profiling.label_from_op_name(
        "jit(f)/scan/jcm:bridge_to_dynamics/reshape"
    ) == "bridge_to_dynamics"
    assert profiling.label_from_op_name("jit(f)/scan/add") is None


def _kernels(*specs):
    """Build trace-shaped kernel events from (op, ts, dur) triples."""
    return [{"module": "jit__run_from_state", "op": op, "ts": ts, "dur": dur}
            for op, ts, dur in specs]


def _attribute(kernels, parsed):
    module, instructions, members = parsed
    return pt.attribute(kernels, {module: instructions}, {module: members})


def test_attribution_charges_each_kernel_to_its_scope(parsed):
    att = _attribute(_kernels(
        ("conv_fusion", 0.0, 10.0),
        ("dyn_fusion", 10.0, 30.0),
        ("bridge", 40.0, 5.0),
    ), parsed)
    assert att.per_label == {
        "tiedtke_convection": 10.0, "dynamics": 30.0, "bridge_to_physics": 5.0,
    }
    assert att.total_us == 45.0


def test_mixed_fusion_is_flagged_but_still_charged(parsed):
    """A fusion spanning two scopes goes wholly to its root's scope, and is counted."""
    att = _attribute(_kernels(("mixed_fusion", 0.0, 8.0)), parsed)
    assert att.per_label == {"tiedtke_convection": 8.0}
    assert att.mixed_us == 8.0


def test_pure_fusion_is_not_flagged_as_mixed(parsed):
    att = _attribute(_kernels(("conv_fusion", 0.0, 8.0)), parsed)
    assert att.mixed_us == 0.0


def test_unscoped_kernel_is_reported_not_redistributed(parsed):
    """Plumbing kernels form their own bucket rather than inflating components."""
    att = _attribute(_kernels(
        ("copy", 0.0, 4.0), ("dyn_fusion", 4.0, 6.0),
    ), parsed)
    assert att.per_label == {"unattributed": 4.0, "dynamics": 6.0}


def test_unjoinable_kernel_is_its_own_bucket(parsed):
    """A kernel whose instruction is missing is a join failure, not a component."""
    att = _attribute(_kernels(("no_such_instruction", 0.0, 7.0)), parsed)
    assert att.unjoinable_us == 7.0
    assert att.per_label == {"unjoinable": 7.0}
    # It must NOT be silently folded into 'unattributed', which would hide a
    # broken join behind a plausible-looking number.
    assert "unattributed" not in att.per_label


def test_kernel_from_unknown_module_is_unjoinable(parsed):
    module, instructions, members = parsed
    att = pt.attribute(
        [{"module": "jit_other", "op": "dyn_fusion", "ts": 0.0, "dur": 3.0}],
        {module: instructions}, {module: members},
    )
    assert att.unjoinable_us == 3.0


def test_device_span_covers_first_start_to_last_end():
    span = pt.device_span_us(_kernels(
        ("a", 100.0, 10.0), ("b", 50.0, 5.0), ("c", 200.0, 25.0),
    ))
    assert span == pytest.approx(175.0)  # 225 - 50


def test_device_span_of_empty_trace_is_zero():
    assert pt.device_span_us([]) == 0.0


def test_report_lists_zero_cost_terms(parsed):
    """A term that costs nothing must still appear, as 0.00 rather than absent."""
    att = _attribute(_kernels(("dyn_fusion", 0.0, 100.0)), parsed)
    result = {
        "preset": "fixture", "config": "physics=x dt=12min",
        "gpu_name": "GPU 0", "gpu_index": 0, "steps": 10,
        "device_span_us": 200.0, "terms": ["tiedtke_convection", "hines_gwd"],
        "attribution": att.as_dict(),
    }
    report = pt.render_report(result)
    assert "| hines_gwd | 0.00 |" in report
    assert "| dynamics | 0.01 |" in report
    assert "50% occupancy" in report


def test_index_warns_when_a_module_name_is_compiled_twice(tmp_path):
    """Two dumps of one jit function cannot be told apart from the trace."""
    (tmp_path / "module_0001.jit_f.sm_80_gpu_after_optimizations.txt").write_text(
        "HloModule jit_f\n\nENTRY %main () -> () {\n  %a = f32[] constant(1)\n}\n"
    )
    (tmp_path / "module_0002.jit_f.sm_80_gpu_after_optimizations.txt").write_text(
        "HloModule jit_f\n\nENTRY %main () -> () {\n  %b = f32[] constant(2)\n}\n"
    )
    index, members, notes = pt.index_hlo_dump(tmp_path)
    assert "jit_f" in index
    assert "main" in members["jit_f"]
    assert len(notes) == 1
    assert "compiled more than once" in notes[0]


def test_index_is_quiet_for_distinct_modules(tmp_path):
    (tmp_path / "module_0001.jit_f.sm_80_gpu_after_optimizations.txt").write_text(
        "HloModule jit_f\n\nENTRY %main () -> () {\n  %a = f32[] constant(1)\n}\n"
    )
    (tmp_path / "module_0002.jit_g.sm_80_gpu_after_optimizations.txt").write_text(
        "HloModule jit_g\n\nENTRY %main.1 () -> () {\n  %b = f32[] constant(2)\n}\n"
    )
    index, members, notes = pt.index_hlo_dump(tmp_path)
    assert set(index) == {"jit_f", "jit_g"}
    assert set(members) == {"jit_f", "jit_g"}
    assert notes == []


def test_empty_device_plane_raises_rather_than_dividing_by_zero(tmp_path):
    """A trace with host events but no kernels is a collection failure."""
    import gzip
    import json

    d = tmp_path / "plugins" / "profile" / "run"
    d.mkdir(parents=True)
    payload = {"traceEvents": [
        {"ph": "X", "pid": 701, "name": "host_op", "ts": 0, "dur": 1,
         "args": {}},
    ]}
    with gzip.open(d / "x.trace.json.gz", "wt") as fh:
        json.dump(payload, fh)

    with pytest.raises(RuntimeError, match="no GPU kernels"):
        pt.load_trace_kernels(tmp_path)


def test_missing_trace_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        pt.load_trace_kernels(tmp_path)


def test_prune_dump_keeps_only_optimized_hlo(tmp_path):
    """The PTX/LLVM IR that XLA also dumps is hundreds of MiB nothing reads."""
    (tmp_path / "m.sm_80_gpu_after_optimizations.txt").write_text("HloModule m")
    (tmp_path / "m.ptx").write_text("x" * 100)
    (tmp_path / "m.ll").write_text("y" * 50)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "more.ptx").write_text("z" * 10)

    freed = pt.prune_dump(tmp_path)

    assert freed == 160
    assert [p.name for p in tmp_path.iterdir()] == [
        "m.sm_80_gpu_after_optimizations.txt"
    ]


def test_every_preset_is_profilable():
    """The tool's --preset choices stay in step with benchmark.py's registry."""
    assert "ma-t63-l47" in pt.PRESETS
    assert all(isinstance(v, list) for v in pt.PRESETS.values())


def test_steps_seen_counts_the_steps_captured(parsed):
    """The completeness probe counts how many steps the trace actually holds."""
    att = _attribute(_kernels(
        ("dyn_fusion", 0.0, 1.0), ("dyn_fusion", 1.0, 1.0),
        ("dyn_fusion", 2.0, 1.0), ("dyn_fusion", 3.0, 1.0),
        ("conv_fusion", 4.0, 50.0), ("conv_fusion", 5.0, 50.0),
    ), parsed)
    assert att.steps_seen["dynamics"] == 4
    assert att.steps_seen["tiedtke_convection"] == 2


def test_steps_seen_is_robust_to_inner_loop_instructions(parsed):
    """One instruction running many times per step must not skew the probe.

    This is the failure that made the previous estimator wrong: RRTMGP's
    per-instruction counts ranged from 2 to 2816 over a window in which it ran
    exactly twice, so max and "dominant instruction" were both far off.
    """
    kernels = _kernels(*[("dyn_fusion", float(i), 1.0) for i in range(5)])
    kernels += _kernels(*[("copy", float(i), 0.1) for i in range(5)])
    # 'd' lives inside %fused_dyn, so it carries the dynamics label too, and
    # here it fires 40 times -- eight per step.
    kernels += _kernels(*[("d", float(i), 0.1) for i in range(40)])
    att = _attribute(kernels, parsed)
    assert att.steps_seen["dynamics"] == 5


def test_report_uses_the_configured_cadence_for_ms_per_call(parsed):
    """ms/call for the sub-cycled term is ms/step times its configured rate."""
    att = _attribute(_kernels(
        ("conv_fusion", 0.0, 500.0), ("conv_fusion", 1.0, 500.0),
    ), parsed)
    result = {
        "preset": "fixture", "config": "physics=x dt=12min",
        "gpu_name": "GPU 0", "gpu_index": 0, "steps": 20,
        "device_span_us": 2000.0, "terms": [],
        "radiation_subcycle_steps": 10,
        "subcycled_terms": ["tiedtke_convection"],
        "attribution": att.as_dict(),
    }
    report = pt.render_report(result)
    # 1000 us / 20 steps = 0.05 ms/step; x10 for the 1-in-10 cadence.
    assert "| tiedtke_convection | 0.05 | 100.0 | 0.50 |" in report
    assert "radiation sub-cycle: every 10 steps" in report


def test_report_leaves_every_step_components_alone(parsed):
    """A component that is not sub-cycled has ms/call equal to ms/step."""
    att = _attribute(_kernels(("dyn_fusion", 0.0, 1000.0)), parsed)
    result = {
        "preset": "fixture", "config": "physics=x dt=12min",
        "gpu_name": "GPU 0", "gpu_index": 0, "steps": 20,
        "device_span_us": 2000.0, "terms": [],
        "radiation_subcycle_steps": 10,
        "subcycled_terms": ["rrtmgp_radiation"],
        "attribution": att.as_dict(),
    }
    report = pt.render_report(result)
    assert "| dynamics | 0.05 | 100.0 | 0.05 |" in report


class _FakeParam:
    def __init__(self, value):
        self._value = value

    def get_value(self):
        return self._value


class _FakeRadiationParams:
    radiation_interval = 7200.0


class _FakeTerm:
    def __init__(self, params=None, name="term"):
        self.name = name
        if params is not None:
            self.params = _FakeParam(params)


class _FakePhysics:
    def __init__(self, terms):
        self.terms = terms


class _FakeModel:
    def __init__(self, terms):
        self.physics = _FakePhysics(terms)


def test_subcycle_steps_reads_the_radiation_interval():
    """7200 s of radiation interval is 10 steps at a 12 minute timestep."""
    model = _FakeModel([_FakeTerm(), _FakeTerm(_FakeRadiationParams(), "rad")])
    assert pt.subcycle_steps(model, 12.0) == (10, frozenset({"rad"}))
    assert pt.subcycle_steps(model, 15.0) == (8, frozenset({"rad"}))


def test_subcycle_steps_defaults_to_every_step():
    """No radiation term (or no interval) means nothing is sub-cycled."""
    assert pt.subcycle_steps(_FakeModel([_FakeTerm()]), 12.0) == (
        1, frozenset())
    assert pt.subcycle_steps(_FakeModel([]), 12.0) == (1, frozenset())


class _FakeGatedTerm:
    """A term riding the radiation gate, like JamOpticsTerm."""

    def __init__(self, interval_s, name="optics"):
        self.name = name
        self._radiation_interval_s = interval_s

    def configure_radiation_gate(self, interval_s):
        self._radiation_interval_s = interval_s if interval_s > 0 else None


def test_subcycle_steps_finds_every_gated_term():
    """Optics rides the radiation gate too, so its ms/call must be scaled.

    Missing this understated the second-largest component by 10x: its
    per-band Mie optics are consumed only by radiation, so the term is gated
    to the same cadence rather than running every step.
    """
    model = _FakeModel([
        _FakeTerm(_FakeRadiationParams(), "rad"),
        _FakeGatedTerm(7200.0, "optics"),
        _FakeTerm(name="convection"),
    ])
    per_cycle, gated = pt.subcycle_steps(model, 12.0)
    assert per_cycle == 10
    assert gated == frozenset({"rad", "optics"})


def test_subcycle_steps_ignores_a_disabled_gate():
    """A gate configured off (interval <= 0) means the term runs every step."""
    model = _FakeModel([
        _FakeTerm(_FakeRadiationParams(), "rad"),
        _FakeGatedTerm(None, "optics"),
    ])
    _, gated = pt.subcycle_steps(model, 12.0)
    assert gated == frozenset({"rad"})


def test_report_scales_every_gated_term(parsed):
    """Both gated terms get the cadence, not just the radiation one."""
    att = _attribute(_kernels(
        ("conv_fusion", 0.0, 400.0), ("dyn_fusion", 1.0, 100.0),
    ), parsed)
    result = {
        "preset": "fixture", "config": "physics=x dt=12min",
        "gpu_name": "GPU 0", "gpu_index": 0, "steps": 20,
        "device_span_us": 1000.0, "terms": [],
        "radiation_subcycle_steps": 10,
        "subcycled_terms": ["tiedtke_convection", "dynamics"],
        "attribution": att.as_dict(),
    }
    report = pt.render_report(result)
    assert "| tiedtke_convection | 0.02 | 80.0 | 0.20 |" in report
    assert "| dynamics | 0.01 | 20.0 | 0.05 |" in report
