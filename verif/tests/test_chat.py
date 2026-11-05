# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""scripts/f2/chat.py, M9's chat recording, on the golden platform with the
tiny model and a lossless fake tokenizer: a turn feeds only what the chat
template adds, the answer stops at the end token, and the recording is
asciicast v2."""

import json

import chat
import fixtures
import pytest
from commands import Layout
from golden import decoder, tiny
from golden_platform import GoldenPlatform
from plan_log import Log
from runtime import Runtime

CAP, PORTS = 128, 4
START, USER, ASSISTANT, USER_END = 1, 2, 3, 4  # the fake template's markers
ROLES = {"user": USER, "assistant": ASSISTANT}


class FakeTokenizer:
    """One token a character: decode writes U+0100 + id, so the template
    reads an answer back as its tokens; the user's ASCII is its own code."""

    def __init__(self, end_id: int) -> None:
        self.end_id = end_id

    def convert_tokens_to_ids(self, _: str) -> int:
        return self.end_id

    def ids(self, text: str) -> list[int]:
        return [ord(c) - 0x100 if ord(c) >= 0x100 else ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(0x100 + t) for t in ids if t != self.end_id)

    def apply_chat_template(self, messages, add_generation_prompt, return_dict) -> dict:
        ids = []
        for m in messages:
            end = self.end_id if m["role"] == "assistant" else USER_END  # the prompt: end-free
            ids += [START, ROLES[m["role"]], *self.ids(m["content"]), end]
        return {"input_ids": ids + [START, ASSISTANT]}


@pytest.fixture(scope="module")
def tiny_model() -> decoder.Model:
    return decoder.from_state_dict(tiny.tiny_config(**fixtures.TINY_CONFIG), fixtures.tiny_state())


def runtime(model: decoder.Model) -> Runtime:
    return Runtime.open(GoldenPlatform(Layout.of(model.config, CAP, ports=PORTS)), model, CAP)


def run_chat(tmp_path, model, tok, max_new: int) -> list[dict]:
    chat.chat(
        Log(str(tmp_path / "chat.jsonl")), runtime(model), tok, ["hi", "yo"], max_new, tmp_path
    )
    return [json.loads(line) for line in (tmp_path / "chat.jsonl").read_text().splitlines()]


def test_a_turn_feeds_only_what_the_template_adds(tmp_path, tiny_model) -> None:
    # The end token is the model's first answer token, so turn 0 ends at once.
    probe = FakeTokenizer(end_id=0)
    first = probe.apply_chat_template([{"role": "user", "content": "hi"}], True, True)["input_ids"]
    end_id = runtime(tiny_model).generate(first, 1)[0][0]
    tok = FakeTokenizer(end_id)
    turns = run_chat(tmp_path, tiny_model, tok, max_new=8)
    assert turns[0]["result"]["ended"] and turns[0]["result"]["tokens"] == [end_id]
    assert turns[0]["params"]["feed"] == first
    assert turns[1]["params"]["feed"] == [START, USER, *tok.ids("yo"), USER_END, START, ASSISTANT]


def test_an_answer_stops_at_max_new_and_the_cast_is_v2(tmp_path, tiny_model) -> None:
    tok = FakeTokenizer(end_id=255)  # not among the tiny model's 5 tokens (the asserts check)
    turns = run_chat(tmp_path, tiny_model, tok, max_new=5)
    for t in turns:
        assert t["result"]["n_new"] == 5 and not t["result"]["ended"]
        assert len(t["result"]["wall_s"]) == len(t["result"]["cycles"]) == 4
    # A cut answer: the next turn feeds the end token the template closes it with.
    want = [tok.end_id, START, USER, *tok.ids("yo"), USER_END, START, ASSISTANT]
    assert turns[1]["params"]["feed"] == want
    lines = (tmp_path / "chat.cast").read_text().splitlines()
    assert json.loads(lines[0])["version"] == 2
    events = [json.loads(line) for line in lines[1:]]
    assert all(e[1] == "o" and "\n" not in e[2].replace("\r\n", "") for e in events)
    assert [e[0] for e in events] == sorted(e[0] for e in events)
    assert "> hi" in events[0][2]
    assert (tmp_path / "chat.txt").read_text().startswith("user: hi")


def test_a_partial_character_is_held_back() -> None:
    assert chat.new_text(lambda _: "ab�", [0], 1) == ("", 1)
    assert chat.new_text(lambda _: "abc", [0], 1) == ("bc", 3)
