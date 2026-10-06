"""
Tests for Olmo 3 support in ConversationEncoder and model configs.

Uses the real Olmo 3 tokenizer, downloaded from HuggingFace on first run.
The span logic under test is meaningless against a mocked tokenizer.
"""

from typing import Dict, List

import pytest
from transformers import AutoTokenizer

from assistant_axis.internals.conversation import ConversationEncoder
from assistant_axis.models import MODEL_CONFIGS, get_config

OLMO_SFT = "allenai/Olmo-3-7B-Instruct-SFT"

OLMO_CHECKPOINTS = [
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Instruct-SFT",
    "allenai/Olmo-3-7B-Instruct-DPO",
    "allenai/Olmo-3-7B-Instruct",
]

SPECIAL_MARKERS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


@pytest.fixture(scope="module")
def encoder() -> ConversationEncoder:
    tokenizer = AutoTokenizer.from_pretrained(OLMO_SFT)
    return ConversationEncoder(tokenizer, model_name=OLMO_SFT)


class TestGetResponseIndicesOlmo:
    """Tests for response_indices on Olmo 3 conversations."""

    @property
    def single_turn(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Ahoy matey!"},
        ]

    @property
    def multi_turn(self) -> List[Dict[str, str]]:
        return self.single_turn + [
            {"role": "user", "content": "Go on"},
            {"role": "assistant", "content": "Arr, the seven seas await."},
        ]

    def test_dispatches_to_olmo(self, encoder: ConversationEncoder) -> None:
        assert encoder._is_olmo()
        assert not encoder._is_qwen()

    def test_single_turn_decodes_to_exact_content(self, encoder: ConversationEncoder) -> None:
        indices = encoder.response_indices(self.single_turn)
        full_formatted = encoder.tokenizer.apply_chat_template(
            self.single_turn, tokenize=False, add_generation_prompt=False
        )
        all_ids = encoder.tokenizer(full_formatted, add_special_tokens=False)["input_ids"]
        decoded = encoder.tokenizer.decode([all_ids[i] for i in indices])
        assert decoded == "Ahoy matey!"

    def test_excludes_markers_and_role_newline(self, encoder: ConversationEncoder) -> None:
        indices = encoder.response_indices(self.single_turn)
        full_formatted = encoder.tokenizer.apply_chat_template(
            self.single_turn, tokenize=False, add_generation_prompt=False
        )
        all_ids = encoder.tokenizer(full_formatted, add_special_tokens=False)["input_ids"]
        decoded = encoder.tokenizer.decode([all_ids[i] for i in indices])
        for marker in SPECIAL_MARKERS:
            assert marker not in decoded
        assert not decoded.startswith("\n")

    def test_multi_turn_per_turn(self, encoder: ConversationEncoder) -> None:
        turns = encoder.response_indices(self.multi_turn, per_turn=True)
        assert len(turns) == 2
        full_formatted = encoder.tokenizer.apply_chat_template(
            self.multi_turn, tokenize=False, add_generation_prompt=False
        )
        all_ids = encoder.tokenizer(full_formatted, add_special_tokens=False)["input_ids"]
        expected = ["Ahoy matey!", "Arr, the seven seas await."]
        for turn_indices, expected_text in zip(turns, expected):
            decoded = encoder.tokenizer.decode([all_ids[i] for i in turn_indices])
            assert decoded == expected_text

    def test_multi_turn_flat_is_concatenation(self, encoder: ConversationEncoder) -> None:
        flat = encoder.response_indices(self.multi_turn)
        turns = encoder.response_indices(self.multi_turn, per_turn=True)
        assert flat == [i for turn in turns for i in turn]

    def test_no_assistant_turn_returns_empty(self, encoder: ConversationEncoder) -> None:
        conversation = self.single_turn[:2]
        assert encoder.response_indices(conversation) == []
        assert encoder.response_indices(conversation, per_turn=True) == []

    def test_empty_assistant_turn_kept_as_empty_list(self, encoder: ConversationEncoder) -> None:
        conversation = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Say again"},
            {"role": "assistant", "content": "ok"},
        ]
        turns = encoder.response_indices(conversation, per_turn=True)
        assert len(turns) == 2
        assert turns[0] == []
        assert len(turns[1]) > 0


class TestBuildTurnSpansOlmo:
    """Tests for build_turn_spans on Olmo 3 conversations."""

    @property
    def conversation(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are a medieval blacksmith."},
            {"role": "user", "content": "What are you working on?"},
            {"role": "assistant", "content": "A sword for the castle guard."},
            {"role": "user", "content": "How long will it take?"},
            {"role": "assistant", "content": "Three days at the forge."},
        ]

    def test_spans_cover_non_system_turns_in_order(self, encoder: ConversationEncoder) -> None:
        _, spans = encoder.build_turn_spans(self.conversation)
        assert [s["role"] for s in spans] == ["user", "assistant", "user", "assistant"]
        assert [s["turn"] for s in spans] == [0, 1, 2, 3]

    def test_span_tokens_decode_to_content(self, encoder: ConversationEncoder) -> None:
        full_ids, spans = encoder.build_turn_spans(self.conversation)
        expected_texts = [m["content"] for m in self.conversation if m["role"] != "system"]
        assert len(spans) == len(expected_texts)
        for span, expected_text in zip(spans, expected_texts):
            decoded = encoder.tokenizer.decode(full_ids[span["start"]:span["end"]])
            assert decoded == expected_text

    def test_n_tokens_matches_span_bounds(self, encoder: ConversationEncoder) -> None:
        _, spans = encoder.build_turn_spans(self.conversation)
        assert len(spans) == 4
        for span in spans:
            assert span["n_tokens"] == span["end"] - span["start"]
            assert span["n_tokens"] > 0

    def test_tool_turns_do_not_desync_spans(self, encoder: ConversationEncoder) -> None:
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        full_ids, spans = encoder.build_turn_spans(conversation)
        assert [s["role"] for s in spans] == ["user", "assistant", "assistant"]
        decoded_last = encoder.tokenizer.decode(full_ids[spans[-1]["start"]:spans[-1]["end"]])
        assert decoded_last == "The answer is 4."

    def test_empty_assistant_turn_keeps_span_position(self, encoder: ConversationEncoder) -> None:
        conversation = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Say again"},
            {"role": "assistant", "content": "ok"},
        ]
        _, spans = encoder.build_turn_spans(conversation)
        assert [s["turn"] for s in spans] == [0, 1, 2, 3]
        assert spans[1]["role"] == "assistant"
        assert spans[1]["n_tokens"] == 0


class TestOlmoSpecialIdsGuard:
    """Tests for _olmo_special_ids token resolution and its absence guard."""

    def test_real_tokenizer_resolves_ids(self, encoder: ConversationEncoder) -> None:
        special = encoder._olmo_special_ids()
        assert special is not None
        assert special["assistant_token_id"] == 78191
        assert special["newline_id"] == 198
        assert encoder.tokenizer.eos_token_id in special["terminator_ids"]

    def test_tokenizer_without_im_start_returns_none(self) -> None:
        # gpt2 has no <|im_start|>, so an olmo-looking name must not pass the guard.
        # convert_tokens_to_ids would return the unk id instead of failing.
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        fake = ConversationEncoder(tokenizer, model_name="my-olmo-variant")
        assert fake._is_olmo()
        assert fake._olmo_special_ids() is None


class TestOlmoModelConfigs:
    """Tests for the Olmo 3 entries in MODEL_CONFIGS."""

    @pytest.mark.parametrize("model_name", OLMO_CHECKPOINTS)
    def test_checkpoint_config(self, model_name: str) -> None:
        assert model_name in MODEL_CONFIGS
        config = get_config(model_name)
        assert config["target_layer"] == 16
        assert config["total_layers"] == 32
        assert config["short_name"] == "Olmo"
