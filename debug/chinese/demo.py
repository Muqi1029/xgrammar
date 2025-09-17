import xgrammar as xgr
from transformers import AutoTokenizer, AutoConfig
import json
import os

def pretty_title(title: str):
    print("\n" + f"\033[47;30m {title} \033[0m".center(100, "=") + "\n")
# Setup
model_id = os.environ["QWEN3_8B"]  # or any model with Chinese support
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
full_vocab_size = config.vocab_size
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)

compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

# Schema with Chinese enum
schema = {
    'type': 'object',
    'properties': {
        'activity': {
            'type': 'string',
            'enum': ['在线', '北京']
        }
    }
}

compiled_grammar = compiler.compile_json_schema(json.dumps(schema))
pretty_title("Compiled Grammar's grammar")
print(compiled_grammar.grammar)
# In v0.1.23, this shows: \u00e5\u009c\u00a8\u00e7\u00ba\u00bf (incorrect)
# Expected: \u5728\u7ebf (correct)

matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)

# Test Chinese token acceptance
test_json = '{"activity": "在线"}'
token_ids = tokenizer.encode(test_json, add_special_tokens=False)
pretty_title("token_ids(在线)")
print(token_ids)

pretty_title("Traverse to check token_id")
for i, token_id in enumerate(token_ids):
    if not matcher.accept_token(token_id):
        print(f"Failed at token {i}: {tokenizer.decode(token_id)}")
        break
# This fails at the Chinese token in v0.1.23
