import json
import os
from pprint import pprint

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from xgrammar import CompiledGrammar, GrammarCompiler, GrammarMatcher, TokenizerInfo, hf

model_path = os.environ["QWEN38B"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def prepare_xgr(tokenizer):
    # 1. get info from transformers
    model_config = AutoConfig.from_pretrained(model_path)

    # 2. get TokenizerInfo
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer, vocab_size=model_config.vocab_size)

    # 3. get GrammarCompiler
    grammar_compiler = GrammarCompiler(tokenizer_info)

    # 4. compile to get compiled grammar with grammar_compiler
    compiled_grammar: CompiledGrammar = grammar_compiler.compile_builtin_json_grammar()

    # 5. get logits_processor
    xgr_logits_processor = hf.LogitsProcessor(compiled_grammar)

    return xgr_logits_processor


def main():
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Introduce yourself in JSON briefly."},
    ]
    texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(texts, return_tensors="pt").to(model.device)

    xgr_logits_processor = prepare_xgr(tokenizer)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature=0.7,
        logits_processor=[xgr_logits_processor],
    )

    model_input_len = len(model_inputs["input_ids"][0])
    output_text = tokenizer.decode(generated_ids[0][model_input_len:-1])
    print(output_text)

    d = json.loads(output_text)
    print(d)


if __name__ == "__main__":
    main()
