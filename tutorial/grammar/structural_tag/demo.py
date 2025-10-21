import os

from transformers import AutoConfig, AutoTokenizer

import xgrammar as xgr
from xgrammar import CompiledGrammar, GrammarCompiler, GrammarMatcher, TokenizerInfo

model_path = os.environ["QWEN3_06B"]
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_config = AutoConfig.from_pretrained(model_path)

tokenizer_info = TokenizerInfo.from_huggingface(tokenizer, model_config.vocab_size)

grammar_compiler = GrammarCompiler(tokenizer_info)
