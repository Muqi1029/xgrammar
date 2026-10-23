"""Benchmark DeepSeek XML masks with bounded strings, without loading model weights.

Use --payload and --xml-text together to replay an existing chat request and its raw
DSML output. --hash-masks records every vocabulary mask for comparison across builds.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer

import xgrammar as xgr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--payload", type=Path, help="Chat request JSON or first line of JSONL")
    parser.add_argument("--xml-text", type=Path, help="Raw DSML output matching the request")
    parser.add_argument("--max-length", type=int, default=400)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--draft-tokens", type=int, default=6)
    parser.add_argument("--hash-masks", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.payload) != bool(args.xml_text):
        parser.error("--payload and --xml-text must be specified together")
    if args.max_length < 0 or args.num_iters < 1 or args.draft_tokens < 1:
        parser.error("length must be nonnegative; iterations and draft tokens must be positive")

    if args.payload:
        with args.payload.open() as stream:
            payload = json.loads(stream.readline())
        text = args.xml_text.read_text()
    else:
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string", "maxLength": args.max_length}},
            "required": ["text"],
            "additionalProperties": False,
        }
        payload = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": schema, "strict": True},
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "f"}},
        }
        body = ("这是一段用于检查字符串长度约束性能的中文文本。" * 6)[: args.max_length]
        text = (
            '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="f">\n'
            '<｜DSML｜parameter name="text" string="true">'
            + body
            + "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    tag = xgr.get_model_structural_tag(
        model="deepseek_v4",
        tools=payload["tools"],
        tool_choice=payload.get("tool_choice", "required"),
        reasoning=False,
        parallel_tool_calls=payload.get("parallel_tool_calls", False),
    )
    compiler = xgr.GrammarCompiler(info, cache_enabled=False)
    start = time.perf_counter()
    compiled = compiler.compile_structural_tag(tag)
    compile_s = time.perf_counter() - start
    tokens = tokenizer.encode(text, add_special_tokens=False)
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)
    runs = []
    hashes = []
    for iteration in range(args.num_iters):
        matcher = xgr.GrammarMatcher(compiled)
        fill_times = []
        draft_s = 0.0
        for index, token in enumerate(tokens):
            start = time.perf_counter()
            matcher.fill_next_token_bitmask(mask)
            fill_times.append(time.perf_counter() - start)
            if args.hash_masks and iteration == 0:
                hashes.append(hashlib.sha256(mask.numpy().tobytes()).hexdigest())
            if not matcher.accept_token(token):
                raise ValueError(
                    f"Token {index} rejected: {tokenizer.decode(tokens[:index + 1])!r}"
                )
            # A linear draft path, followed by rollback to the committed prefix. Measure
            # separately so the ordinary mask-fill result stays comparable to non-spec decoding.
            if index % 16 == 0:
                draft = tokens[index + 1 : index + args.draft_tokens]
                start = time.perf_counter()
                for candidate in draft:
                    matcher.fill_next_token_bitmask(mask)
                    if not matcher.accept_token(candidate):
                        raise ValueError("Draft token rejected")
                if draft:
                    matcher.rollback(len(draft))
                draft_s += time.perf_counter() - start
        runs.append(
            {
                "mask_total_s": sum(fill_times),
                "mask_median_ms": statistics.median(fill_times) * 1000,
                "mask_max_ms": max(fill_times) * 1000,
                "draft_traverse_s": draft_s,
            }
        )
    result = {"tokens": len(tokens), "compile_s": compile_s, "runs": runs}
    if args.hash_masks:
        result["mask_hashes"] = hashes
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "mask_hashes"}, indent=2))


if __name__ == "__main__":
    main()
