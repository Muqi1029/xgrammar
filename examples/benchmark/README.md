
## Run Benchmark

### Benchmark Grammar Compile and Mask Generation

#### Dependencies
```
outlines                          0.1.3
outlines_core                     0.1.14
lm-format-enforcer                0.10.6
```

#### Run
```bash
python3 bench_grammar_compile_mask_gen.py [-h] [--backend {xgrammar,outlines,lmformatenforcer}]
                                          [--num_iters NUM_ITERS] [--num_warmup NUM_WARMUP]
```


### Benchmark Apply Token Bitmask Inplace Kernels

#### Run
```bash
python3 examples/benchmark/bench_apply_token_bitmask_inplace.py
```

#### Results
H100
|   Batch |   Vocab |   Masked cnt |   Torch Compile |         Triton  |           CUDA  |
|    size |    size |              |     Baseline us |    us (speedup) |    us (speedup) |
|--------:|--------:|-------------:|----------------:|----------------:|----------------:|
|       1 |  128000 |            1 |            5.85 |    5.41 (1.08x) |    5.46 (1.07x) |
|       1 |  128000 |        64000 |            5.84 |    6.01 (0.97x) |    6.24 (0.94x) |
|       1 |  128000 |       127000 |            5.84 |    6.09 (0.96x) |    5.95 (0.98x) |
|       8 |  128000 |            1 |           10.75 |    5.86 (1.83x) |    5.90 (1.82x) |
|       8 |  128000 |        64000 |           10.75 |    7.59 (1.42x) |    9.85 (1.09x) |
|       8 |  128000 |       127000 |           10.77 |    7.85 (1.37x) |    8.06 (1.34x) |
|      64 |  128000 |            1 |           48.59 |   13.10 (3.71x) |    9.68 (5.02x) |
|      64 |  128000 |        64000 |           48.59 |   45.43 (1.07x) |   38.76 (1.25x) |
|      64 |  128000 |       127000 |           48.58 |   32.84 (1.48x) |   26.29 (1.85x) |
|     512 |  128000 |            1 |          349.84 |   67.34 (5.20x) |   37.06 (9.44x) |
|     512 |  128000 |        64000 |          346.94 |  330.36 (1.05x) |  256.53 (1.35x) |
|     512 |  128000 |       127000 |          345.54 |  249.66 (1.38x) |  157.51 (2.19x) |
|    4096 |  128000 |            1 |         2895.83 |  494.47 (5.86x) | 249.96 (11.59x) |
|    4096 |  128000 |        64000 |         2863.31 | 2517.85 (1.14x) | 1993.29 (1.44x) |
|    4096 |  128000 |       127000 |         2720.67 | 1935.24 (1.41x) | 1207.38 (2.25x) |

### DeepSeek XML Length Constraints

Replay XML tool-call tokens with `maxLength` enforced, without loading model weights:

```bash
python examples/benchmark/bench_deepseek_xml_mask.py \
  --tokenizer /path/to/DeepSeek-V4-Flash --max-length 400 --num-iters 3
```

To replay a real request, provide its JSON payload (or a JSONL file whose first line is the
request) and its raw DSML output using `--payload request.jsonl --xml-text output.txt`.
`--hash-masks --output result.json` also records SHA-256 hashes of every full vocabulary
mask, allowing exact comparison between builds on the same token sequence. Timing covers
CPU grammar compilation, per-token mask fills, and a separate linear speculative traversal
with rollback; it does not measure model execution or end-to-end TTFT.

For the DeepSeek-V4 tokenizer and a 285-token replay of a five-field evaluation tool,
with the original `maxLength: 400` constraints retained, three runs gave:

| Measurement | Before XML optimization (`614ab01`) | With XML optimization |
| --- | ---: | ---: |
| CPU mask fills for the full replay | 4.230–4.252 s | 2.008–2.130 ms |
| Linear draft traversal samples with rollback | 1.359–1.371 s | 0.610–0.646 ms |
| Grammar compilation | 737 ms | 20 ms |

All 285 full-vocabulary mask hashes matched between builds. These are CPU replay
measurements on one token sequence, not a model-serving latency guarantee. The optimized
compiled masks use serialization version v19; older serialized caches must be regenerated.
