# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""scripts/f2/core_plan.py, M9's measurements on F2, on the golden platform
with the tiny fixture: the timed decoding is Runtime.generate's, a correct
run passes, and a wrong token or logit names the first bad index. The
plan's numbers come only from F2."""

import copy
import json
from contextlib import nullcontext

import core_plan
import fixtures
import paths
import pytest
from commands import Layout
from golden import decoder, tiny
from golden_platform import GoldenPlatform
from plan_log import Log
from runtime import Runtime

CAP, PORTS, MHZ = 128, 4, 250.0
PCIS_BYTES = (4096, 1024)


@pytest.fixture(scope="module")
def tiny_model() -> decoder.Model:
    return decoder.from_state_dict(tiny.tiny_config(**fixtures.TINY_CONFIG), fixtures.tiny_state())


@pytest.fixture(scope="module")
def runs() -> list[dict]:
    return json.loads((paths.FIXTURES / "tiny_greedy.json").read_text())["runs"]


def golden(model: decoder.Model):
    return lambda: nullcontext(GoldenPlatform(Layout.of(model.config, CAP, ports=PORTS)))


def plan_lines(tmp_path, model, fixture_runs: dict) -> list[dict]:
    out = tmp_path / "plan.jsonl"
    core_plan.run_plan(Log(str(out)), golden(model), model, fixture_runs, MHZ, PCIS_BYTES, CAP)
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_timed_greedy_is_generate(tiny_model) -> None:
    rt = Runtime.open(golden(tiny_model)().enter_result, tiny_model, CAP)
    prompt = [3, 141, 59, 26]
    tokens, logits = rt.generate(prompt, 6)
    run = core_plan.timed_greedy(rt, prompt, 6)
    assert run["tokens"] == tokens
    assert run["logits_sha256"] == core_plan.arith.bits_sha256(logits)
    assert run["rows_sha256"] == [core_plan.arith.bits_sha256(row) for row in logits]
    assert len(run["wall_s"]) == len(run["cycles"]) == len(prompt) + 5


def test_the_tiny_fixture_passes(tmp_path, tiny_model, runs) -> None:
    lines = plan_lines(tmp_path, tiny_model, {"tiny_greedy.json": runs})
    steps = [line["step"] for line in lines]
    assert steps == ["pcis_write"] * 5 + ["pcis_read"] * 5 + ["open"] + ["fixture"] * 3 + ["done"]
    assert not any("error" in line["result"] for line in lines)
    fixture = [line["result"] for line in lines if line["step"] == "fixture"]
    assert all(r["tokens_ok"] and r["logits_ok"] for r in fixture)
    assert all(
        r["decode_steps"] == len(want["tokens"]) - 1 for r, want in zip(fixture, runs, strict=True)
    )


def test_a_wrong_token_or_row_is_named(tmp_path, tiny_model, runs) -> None:
    """A bad token at 7 and, in the per-position form (the long fixture's), a
    bad row hash at 9."""
    rt = Runtime.open(golden(tiny_model)().enter_result, tiny_model, CAP)
    want = runs[0]
    rows = core_plan.timed_greedy(rt, want["prompt"], len(want["tokens"]))["rows_sha256"]
    bad_token = copy.deepcopy(want)
    bad_token["tokens"][7] ^= 1
    bad_row = copy.deepcopy(want) | {"logits_sha256": rows[:9] + ["0" * 64] + rows[10:]}
    lines = plan_lines(tmp_path, tiny_model, {"x.json": [bad_token], "x_long.json": [bad_row]})
    first, long = (line for line in lines if line["step"] in ("fixture", "long"))
    assert first["result"]["first_bad_token"] == 7 and "error" in first["result"]
    assert long["step"] == "long" and long["result"]["first_bad_position"] == 9
    assert long["result"]["tokens_ok"] and "error" in long["result"]
