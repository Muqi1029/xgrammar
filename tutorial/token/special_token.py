import os

from transformers import AutoTokenizer


def qwen_tool_token():
    model_path = os.environ["QWEN3_8B"]
    special_tokens = ["<tool_call>", "</tool_call>"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for token in special_tokens:
        token_id = tokenizer.encode(token)[-1]
        decoded_token = tokenizer.decode(token_id)
        print(f"{token=}, {token_id=}, {decoded_token=}")


def dpsk31_tool_token():
    model_path = "deepseek-ai/DeepSeek-V3.1"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tool_calls_begin = "<｜tool▁calls▁begin｜>"
    tool_call_begin = "<｜tool▁call▁begin｜>"
    special_tokens = [tool_calls_begin, tool_call_begin]
    for token in special_tokens:
        token_id = tokenizer.encode(token)[-1]
        decoded_token = tokenizer.decode(token_id)
        print(f"{token=}, {token_id=}, {decoded_token=}")


if __name__ == "__main__":
    qwen_tool_token()
    dpsk31_tool_token()
