import argparse
import json

from transformers import AutoConfig, AutoTokenizer

from xgrammar import Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo


def read_json(datapath):
    with open(datapath) as f:
        return json.load(f)


def accept(grammar, args):
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
    grammar_compiler = GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_grammar(grammar)
    matcher = GrammarMatcher(compiled_grammar)
    print(matcher.accept_string("```"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3.1-Terminus")

    args = parser.parse_args()
    data = read_json(args.path)
    print(" ENBF ".center(80, "="))

    ebnf = Grammar.from_json_schema(json.dumps(data), print_converted_ebnf=True)

    accept(ebnf, args)
