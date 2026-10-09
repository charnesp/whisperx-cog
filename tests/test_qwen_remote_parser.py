"""GPU-free RED tests for the qwen3-asr remote reply PARSER (G4).

Change: openspec/changes/qwen3-asr-remote-vllm tasks.md group 4 (task 4.1):
- `language X<asr_text>text` -> text only (prefix never in segments);
- content without the prefix (structured output) -> used as-is;
- `language None<asr_text>` / empty transcription -> EMPTY segment text
  (local-path parity rule), NOT an error;
- empty/missing content -> typed parse error (not an empty transcript);
- multi-choice envelope -> FIRST choice only;
- finish_reason == "length" -> typed error (never silently-truncated text);
- the reply text NEVER determines segment timestamps (parser returns text
  only; the caller fuses VAD start/end — asserted structurally here).
Tests call qwen_remote_client.parse_response directly (pure function) AND
the vLLM-mirror double from test_qwen_remote_client for envelope shape
parity.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_remote_client import (  # noqa: E402
    QwenRemoteError,
    RemoteResponseParserError,
    parse_asr_content,
    parse_asr_language,
    parse_response,
)


def _envelope(content, finish_reason="stop", n_choices=1):
    choices = [
        {
            "index": i,
            "message": {"role": "assistant", "content": content if i == 0 else None},
            "finish_reason": finish_reason,
        }
        for i in range(n_choices)
    ]
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": choices,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class TestParseAsrContent(unittest.TestCase):
    def test_prefixed_reply_yields_text_only(self):
        self.assertEqual(
            parse_asr_content("language French<asr_text>Bonjour tu va bien"),
            "Bonjour tu va bien",
        )

    def test_prefix_never_leaks_into_segment_text(self):
        text = parse_asr_content("language French<asr_text>Reçu cinq sur cinq")
        self.assertNotIn("language", text)
        self.assertNotIn("<asr_text>", text)

    def test_structured_output_without_prefix_used_as_is(self):
        self.assertEqual(parse_asr_content("Bonjour tu va bien"), "Bonjour tu va bien")

    def test_language_prefix_without_asr_marker_is_typed_parse_error(self):
        """A `language X` prefix with NO <asr_text> marker must fail explicitly.

        Regression (review finding P1-A): the parser used to return the whole
        string (prefix included) in that case, leaking `language French` into
        the segment text.
        """
        with self.assertRaises(RemoteResponseParserError) as caught:
            parse_asr_content("language French hello world")
        self.assertTrue(str(caught.exception).startswith("QwenRemoteError: parse"))

    def test_capitalised_sentence_is_not_mistaken_for_the_prefix(self):
        """Structured output starting with the capitalised word "Language" is
        not the model's lowercase `language X` prefix grammar."""
        self.assertEqual(
            parse_asr_content("Language models are useful"),
            "Language models are useful",
        )

    def test_lowercase_language_word_in_structured_output_is_kept(self):
        """Structured output that merely STARTS with the word 'language' plus a
        non-language token is not the server prefix grammar (confirmation
        review reserve: 'language models are useful' was rejected as parse)."""
        self.assertEqual(
            parse_asr_content("language models are useful in production"),
            "language models are useful in production",
        )

    def test_single_language_word_still_rejected_without_marker(self):
        """The real prefix shape (language + known language name, no marker)
        must still fail explicitly: the narrow fix must not lose P1-A."""
        with self.assertRaises(RemoteResponseParserError):
            parse_asr_content("language German hallo welt")

    def test_language_none_marker_yields_empty_segment(self):
        self.assertEqual(parse_asr_content("language None<asr_text>"), "")

    def test_empty_transcription_yields_empty_segment(self):
        self.assertEqual(parse_asr_content("language English<asr_text>   "), "")

    def test_missing_content_is_typed_parse_error(self):
        with self.assertRaises(RemoteResponseParserError):
            parse_asr_content(None)

    def test_whitespace_only_content_is_typed_parse_error(self):
        with self.assertRaises(RemoteResponseParserError):
            parse_asr_content("   \n\t  ")

    def test_empty_content_is_typed_parse_error_not_empty_transcript(self):
        with self.assertRaises(RemoteResponseParserError):
            parse_asr_content("")

    def test_finish_reason_length_is_typed_error(self):
        with self.assertRaises(RemoteResponseParserError) as caught:
            parse_asr_content("language French<asr_text>coupe au", finish_reason="length")
        self.assertIn("length", str(caught.exception))

    def test_typed_parse_errors_carry_stable_prefix(self):
        for bad, kw in ((None, {}), ("", {}), ("x", {"finish_reason": "length"})):
            with self.assertRaises(QwenRemoteError) as caught:
                parse_asr_content(bad, **kw)
            self.assertTrue(str(caught.exception).startswith("QwenRemoteError: parse"))


class TestParseAsrLanguage(unittest.TestCase):
    """Server-detected language from the reply prefix (review finding P1-C)."""

    def test_name_token_normalized_to_iso_code(self):
        self.assertEqual(parse_asr_language("language French<asr_text>Bonjour"), "fr")
        self.assertEqual(parse_asr_language("language English<asr_text>hi"), "en")
        self.assertEqual(parse_asr_language("language German<asr_text>hallo"), "de")

    def test_iso_code_token_passes_through(self):
        self.assertEqual(parse_asr_language("language fr<asr_text>x"), "fr")

    def test_language_none_marker_is_no_language(self):
        self.assertIsNone(parse_asr_language("language None<asr_text>"))

    def test_structured_output_has_no_language(self):
        self.assertIsNone(parse_asr_language("Bonjour tu va bien"))

    def test_missing_content_has_no_language(self):
        self.assertIsNone(parse_asr_language(None))


class TestParseResponseEnvelope(unittest.TestCase):
    def test_multi_choice_envelope_uses_first_choice(self):
        body = _envelope("language French<asr_text>premier", n_choices=2)
        # second choice carries a DIFFERENT text to catch a zip over choices
        body["choices"][1]["message"]["content"] = "language French<asr_text>second"
        text = parse_response(200, json.dumps(body))
        self.assertEqual(text, "premier")

    def test_finish_reason_length_through_envelope_is_typed_error(self):
        body = _envelope("language French<asr_text>trunc", finish_reason="length")
        with self.assertRaises(RemoteResponseParserError):
            parse_response(200, json.dumps(body))

    def test_empty_choices_envelope_is_typed_error(self):
        body = _envelope("x")
        body["choices"] = []
        with self.assertRaises(RemoteResponseParserError):
            parse_response(200, json.dumps(body))

    def test_missing_message_content_is_typed_error(self):
        body = _envelope(None)
        with self.assertRaises(RemoteResponseParserError):
            parse_response(200, json.dumps(body))

    def test_parser_returns_text_only_never_timestamps(self):
        body = _envelope("language French<asr_text> texte > jamais d'horodatage")
        text = parse_response(200, json.dumps(body))
        self.assertEqual(text, "texte > jamais d'horodatage")
        self.assertFalse(any(ch.isdigit() for ch in text[:6]))


if __name__ == "__main__":
    unittest.main()
