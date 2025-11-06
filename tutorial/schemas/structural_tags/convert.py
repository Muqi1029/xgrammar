from xgrammar import Grammar

triggered_tags_response_format = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<mu"],
        "tags": [{"begin": "<muqi>1", "content": {"type": "any_text"}, "end": "</muqi>"}],
    },
}


grammar = Grammar.from_structural_tag(triggered_tags_response_format)
print(grammar)
