![image-20260613165554775](/Users/ziangs/Library/Application Support/typora-user-images/image-20260613165554775.png)

Vocab size: V

Seq length: T

Batch size: B

Embedding dim: d_model, dv = d_model/h, #Parameters in FF = 2#Parameters in MHA

So for SwiGLU, d_ff = 8/3 d_model

Num of Layers: L
$$
Attention(Q,K,V) = softmax\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
Tensor shape: $X \in \mathbb{R}^{B\times T\times d_{model}}, Q, K \in \mathbb{R}^{B\times h\times T \times d_h}$

Number of trainable parameters :Vd + L( 4d^2 + 8d^2  + 2d ) + d + Vd

FLOPs: approx 2 * L * ( 6BTd^2 +  2BT^2d + 2 BTd^2  + 3BTdd_ff)  +  2 BTdV

QKV + attention + output projections

**Resource accouting for training with AdamW**

1. Memory (float32) 4 * N_params * 4 + 4 * N_activations. The activation depends on the batch size
2. One step of AdamW take FLOPs = FLOPs of forward pass * 3 + 14 * N_params



```python
# mmap
np.save("train.npy", tokens)
x = np.load("train.npy", mmap_mode="r")
# or for binary file
tokens.astype(np.uint16).tofile("train.bin")
x = np.memmap("train.bin", dtype=np.uint16, mode="r")
```

