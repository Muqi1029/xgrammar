"""Bounded raw XML strings must preserve delimiter ambiguity and speculative rollback."""

import pytest

import xgrammar as xgr

VOCAB = [
    b"a",
    b"aa",
    b"aaaa",
    "中".encode(),
    "中文".encode(),
    "😀".encode(),
    b"<",
    b"</",
    b"</p>",
    b"a</p>",
    b"</p>tail",
    b"<x",
    b"a<x",
    b"\n",
    b"\x00",
    b"\xe4",
    b"\xb8",
    b"\xad",
    b"\xf0\x9f",
    b"\x98\x80",
    b"p>",
    b"/p>",
    b"tail",
    b"z",
    "\U0010ffff".encode(),
]


def _mask(matcher, info):
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)
    matcher.fill_next_token_bitmask(mask)
    return [bool((int(mask[0, i // 32]) >> (i % 32)) & 1) for i in range(info.vocab_size)]


def _unrolled_grammar(lower, upper):
    # An independent expression of the same language, without counted repeat edges.
    rules = ['root ::= r0 "</p>"']
    for i in range(upper):
        optional = '"" | ' if i >= lower else ""
        rules.append(f"r{i} ::= {optional}[\\u0000-\\U0010ffff] r{i + 1}")
    rules.append(f'r{upper} ::= ""')
    return "\n".join(rules)


@pytest.mark.parametrize(
    "lower,upper", [(0, 0), (0, 1), (0, 3), (2, 4), (0, 129), (128, 130), (0, 400)]
)
@pytest.mark.parametrize("roundtrip", [False, True])
def test_xml_delimiter_overlap_matches_unrolled_grammar(lower, upper, roundtrip):
    info = xgr.TokenizerInfo(VOCAB)
    compiler = xgr.GrammarCompiler(info, cache_enabled=False)
    counted = compiler.compile_grammar(f'root ::= [\\u0000-\\U0010ffff]{{{lower},{upper}}} "</p>"')
    if roundtrip:
        counted = xgr.CompiledGrammar.deserialize_json(counted.serialize_json(), info)
    unrolled = compiler.compile_grammar(_unrolled_grammar(lower, upper))
    assert xgr.GrammarMatcher(counted).accept_string("中" * upper)
    prefixes = {"", "<", "</", "</p>", "<x", "a</p>", "中<", "😀"}
    for length in {max(0, lower - 1), lower, max(0, upper - 1), upper}:
        prefixes.update({"a" * length, "中" * length, "a" * length + "</"})
    for prefix in prefixes:
        actual = xgr.GrammarMatcher(counted, terminate_without_stop_token=True)
        expected = xgr.GrammarMatcher(unrolled, terminate_without_stop_token=True)
        accepted = actual.accept_string(prefix)
        assert accepted == expected.accept_string(prefix), (lower, upper, prefix)
        if not accepted or actual.is_terminated():
            continue
        actual_mask = _mask(actual, info)
        assert actual_mask == _mask(expected, info), (lower, upper, prefix)
        # Speculative traversal must restore the same mask after every accepted draft token,
        # including partial UTF-8 codepoints and tokens crossing the closing tag.
        for token_id, allowed in enumerate(actual_mask):
            fresh = xgr.GrammarMatcher(counted, terminate_without_stop_token=True)
            assert fresh.accept_string(prefix)
            assert fresh.accept_token(token_id) == allowed, (prefix, token_id)
            if allowed:
                assert actual.accept_token(token_id), (prefix, token_id)
                actual.rollback(1)
                assert _mask(actual, info) == actual_mask, (prefix, token_id)


@pytest.mark.parametrize("bound", [0, 1, 3, 129, 400])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_deepseek_xml_parameter_length_boundary(bound, roundtrip):
    info = xgr.TokenizerInfo(VOCAB)
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "maxLength": bound}},
        "required": ["text"],
        "additionalProperties": False,
    }
    tag = xgr.get_model_structural_tag(
        model="deepseek_v4",
        tools=[
            {"type": "function", "function": {"name": "f", "parameters": schema, "strict": True}}
        ],
        tool_choice={"type": "function", "function": {"name": "f"}},
        reasoning=False,
        parallel_tool_calls=False,
    )
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(tag)
    if roundtrip:
        compiled = xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)
    prefix = (
        '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="f">\n'
        '<｜DSML｜parameter name="text" string="true">'
    )
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_string(prefix + "中" * bound)
    allowed = _mask(matcher, info)
    assert not allowed[VOCAB.index("中".encode())]
    assert not allowed[VOCAB.index(b"a")]
    assert allowed[VOCAB.index(b"<")], "the closing tag must remain possible at maxLength"
    assert matcher.accept_string("</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>")
