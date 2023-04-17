# causal-transformer
A causal transformer based LM.

Example usage:
```
model = transformer_lm(
    dim = 512,
    vocab_size = 29,
    depth = 10,
    heads = 8,
    dim_head = 64,
    dropout=0.0,
    causal = True,
    shared_kv = True,
)
# intermeditate_logits are losses from intermediate layers if intermediate losses is enabled (False by default)
logits, interimediate_logits, cached_kvs = model(labels[t], length = length[t])

# cached_kvs can then be passed back into the model for easy recurrent training or inference
logits, interimediate_logits, cached_kvs = model(labels[t+1], length = length[t+1], cache = cached_kvs)

# see test function caching_test() for more details
```

Currently has the following features:
- caching previous keys and values for - see test function caching_test() - allows for incremental inference or training like with transformer-xl 
- coscine similarity based attention (i've found that this works better) https://arxiv.org/abs/2010.04245
- gradient checkpointing (save that precious memory)
- intermediate losses (improves performance esp. in lower data scenarious) https://arxiv.org/pdf/2102.03216.pdf
- dynamic position bias (generalizes well to v long distances - works very similar to AliBi but is learnt so more flexible)
- multi-query attention (keys and values only have 1 head, increases utilization when paired with kv caching, used in PaLM model) https://arxiv.org/pdf/1911.02150.pdf
- token-Shift (feature dim is shifted forwards by 1) https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py and idea from https://github.com/BlinkDL/RWKV-LM faster convergence on character level LMing
- talking-heads (used in x-transformers, I added setting 'pre' | 'post' | 'both' for this


Some of the code is taken or adapted from lucidrains/Phil Wangs x-transformers library
