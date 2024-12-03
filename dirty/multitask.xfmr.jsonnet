{
  "data": {
    "train_file": "data1/train-shard-*.tar",
    "dev_file": "data1/dev*.tar",
    "test_file": "data1/test.tar",
    "vocab_file": "data1/vocab.bpe10000",
    "typelib_file": "data1/typelib.json",
    "max_src_tokens_len": 510,
    "max_num_var": 32,
    "retype": true,
    "rename": true,
    "interleave": true,
  },
  "encoder":{
    "type": "XfmrSequentialEncoder",
    "source_embedding_size": 512,
    "hidden_size": 512,
    "vocab_file": $['data'].vocab_file,
    "dropout": 0.1,
    "num_layers": 6,
    "num_heads": 8,
  },
  "decoder": {
    "type": if $['data'].interleave then 'XfmrInterleaveDecoder' else 'XfmrDecoder',
    "vocab_file": $['data'].vocab_file,
    "typelib_file": "data1/typelib.json",
    "target_embedding_size": $['encoder'].source_embedding_size,
    "hidden_size": $['encoder'].hidden_size,
    "dropout": 0.1,
    "num_layers": $['encoder'].num_layers,
    "num_heads": $['encoder'].num_heads,
    "mem_mask": "soft",
  },
  "mem_encoder":{
    "type": "XfmrMemEncoder",
    "source_embedding_size": 256,
    "hidden_size": 256,
    "vocab_file": $['data'].vocab_file,
    "dropout": 0.1,
    "num_layers": 3,
    "num_heads": 1,
  },
  "mem_decoder": {
    "type": 'SimpleDecoder',
    "vocab_file": $['data'].vocab_file,
    "hidden_size": $['mem_encoder'].hidden_size,
  },
  "train": {
    "torch_float32_matmul": "medium", # high, highest
    # 16-mixed/AMP can lead to NaN errors
    "precision": "32", #bit
    "batch_size": 16,
    "grad_accum_step": 4,
    "max_epoch": 25,
    "lr": 1e-4,
    "patience": 2,
    "check_val_every_n_epoch": 1,
    "safety_margin": 0.01, # Increase this if you run out of GPU memory
  },
  "test": {
    "pred_file": "pred_mt.json",
    "batch_size": 64,
    "beam_size": 5,
  }
}
