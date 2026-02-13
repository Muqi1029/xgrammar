import requests

tag_response_format = {
    "type": "structural_tag",
    "format": {
        "type": "tag",
        "begin": "<think>",
        "content": {"type": "any_text"},
        "end": "</think>",
    },
}

triggered_tags_response_format = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<mu"],
        "tags": [{"begin": "<muqi>1", "content": {"type": "any_text"}, "end": "</muqi>"}],
    },
}


res = requests.post(
    url="http://127.0.0.1:8888/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Who are you"}],
        # "response_format": triggered_tags_response_format,
        "response_format": tag_response_format,
        "max_completion_tokens": 1024,
    },
)
print(res.json())
