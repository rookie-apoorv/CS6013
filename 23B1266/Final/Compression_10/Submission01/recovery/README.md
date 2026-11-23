# Post-compression recovery (size-neutral distillation)

`recovery/recover.py` turns the compressed checkpoint written by `compress.py` into a
trainable student **without changing a single integer code**:

* every coded weight is rebuilt as `(symbol + q_min) * step0 * exp(s)` per group of
  `group_size` columns (`s` trainable, starts at 0), escape positions take a trainable
  outlier value, pruned rows/columns (format v2) stay exact zeros;
* every tensor stored raw (`bf16` norms, linear-attention gates, conv) is trainable.

Objective: KL(teacher || student) per token on teacher-forced training traces, teacher =
the uncompressed (fine-tuned) bf16 model the checkpoint was compressed from.

After training, `export()` overwrites only the `<name>::steps` (fp16), `<name>::outliers`
(fp16) and raw tensors **in place** in a copy of the compressed shards: same keys,
shapes, dtypes, file sizes and `compression_config.json`, so the size fraction is
unchanged and the unchanged `decompress.py` restores the recovered model.

Driver notebook: `notebooks/nb5_recover.py` (molab). Settings used for this checkpoint:
teacher `grey-cat/cs6013-qwen35-4b-sft-v1`; training data `grey-cat/cs6013-sft-artifacts:runs/sft-v1/data/train_set.parquet`, sequences cut to 8,192 tokens
(1,216 sequences); 1 epoch = 152 steps of 8 sequences; AdamW lr 0.002 (log-step
multipliers) / 0.0001 (outliers) / 5e-05 (raw tensors), warm-up 5 steps, cosine to 10%,
grad-clip 1.0.
