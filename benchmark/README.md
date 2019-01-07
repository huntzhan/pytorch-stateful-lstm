# Benchmark

Test environment:

- GPU: 1080Ti
- CPU: Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz
- Memory: 64GiB

## 2 Layers Unidirectional Stateful LSTM (CUDA)

**50%** speedup by comparing `StatefulUnidirectionalLstm` with `AugmentedLstm` + `PytorchSeq2SeqWrapper`:

- `uni-lstm-cuda`: 0.011974 secs per batch
- `allennlp-seq2seq-cuda`: 0.02426 secs per batch

![uni-lstm-cuda](https://user-images.githubusercontent.com/5213906/50764188-e65c6300-12ac-11e9-801c-f9671b87e268.png)

Commands for profiling:

```
inv uni-lstm 2 128 256 256 32 20 1000 cuda ./data/uni-lstm-cuda.json
inv allennlp-seq2seq 2 128 256 256 32 20 1000 cuda ./data/allennlp-seq2seq-cuda.json
inv plot './data/*.json' '[("allennlp-seq2seq", "cuda"), ("uni-lstm", "cuda")]' ./data/uni-lstm-cuda.png
```

## 2 Layers Unidirectional Stateful LSTM (CPU)

**14%** speedup by comparing `StatefulUnidirectionalLstm` with `AugmentedLstm` + `PytorchSeq2SeqWrapper`:

- `uni-lstm-cpu`: 0.039998 secs per batch
- `allennlp-seq2seq-cpu`: 0.046657 secs per batch

![uni-lstm-cuda](https://user-images.githubusercontent.com/5213906/50764188-e65c6300-12ac-11e9-801c-f9671b87e268.png)

Commands for profiling:

```
inv uni-lstm 2 128 256 256 32 20 1000 cpu ./data/uni-lstm-cpu.json
inv allennlp-seq2seq 2 128 256 256 32 20 1000 cpu ./data/allennlp-seq2seq-cpu.json
inv plot './data/*.json' '[("allennlp-seq2seq", "cpu"), ("uni-lstm", "cpu")]' ./data/uni-lstm-cpu.png
```

## Single Layer Unidirectional LSTM (CUDA)

**46%** speedup by comparing `UnidirectionalSingleLayerLstm` with `LstmCellWithProjection`:

- `uni-sl-lstm-cuda`: 0.00567 secs per batch
- `allennlp-lstm-cell-cuda`: 0.01058 secs per batch

![sl-uni-lstm-cuda](https://user-images.githubusercontent.com/5213906/50735487-ea27b100-11ea-11e9-81b9-a0600c8dbd03.png)

Commands for profiling:

```
inv uni-sl-lstm 128 256 256 32 20 1000 cuda ./data/uni-sl-lstm-cuda.json
inv allennlp-lstm-cell 128 256 256 32 20 1000 cuda ./data/allennlp-lstm-cell-cuda.json
inv plot './data/*.json' '[("allennlp-lstm-cell", "cuda"), ("uni-sl-lstm", "cuda")]' ./data/sl-uni-lstm-cuda.png
```

## Single Layer Unidirectional LSTM (CPU)

**13%** speedup by comparing `UnidirectionalSingleLayerLstm` with `LstmCellWithProjection`:

- `uni-sl-lstm-cuda`: 0.01943 secs per batch
- `allennlp-lstm-cell-cuda`: 0.02235 secs per batch

![sl-uni-lstm-cpu](https://user-images.githubusercontent.com/5213906/50735486-e98f1a80-11ea-11e9-926f-a1d782784f98.png)


Commands for profiling:

```
inv uni-sl-lstm 128 256 256 32 20 1000 cpu ./data/uni-sl-lstm-cpu.json
inv allennlp-lstm-cell 128 256 256 32 20 1000 cpu ./data/allennlp-lstm-cell-cpu.json
inv plot './data/*.json' '[("allennlp-lstm-cell", "cpu"), ("uni-sl-lstm", "cpu")]' ./data/sl-uni-lstm-cpu.png
```
