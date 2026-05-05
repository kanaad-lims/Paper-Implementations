# Cross-Attention Flow: Encoder → Decoder

## End-to-End Sequence

```mermaid
flowchart TD
    subgraph ENCODER["🔵 Encoder (encoder.py)"]
        E1["Source tokens (x)"] --> E2["Word Embedding + Position Embedding"]
        E2 --> E3["Dropout"]
        E3 --> E4["TransformerBlock\n(self-attention: Q=x, K=x, V=x)"]
        E4 --> E5["enc_out"]
    end

    subgraph DECODER["🟠 Decoder (decoder.py)"]
        D1["Target tokens (x)"] --> D2["Word Embedding + Position Embedding"]
        D2 --> D3["Dropout"]

        subgraph DB["DecoderBlock"]
            direction TB
            D4["① Masked Self-Attention\nQ=x, K=x, V=x + trg_mask"]
            D4 --> D5["② LayerNorm + Residual + Dropout\nquery = norm(attention + x)"]
            D5 --> D6["③ ⭐ CROSS-ATTENTION ⭐\ntransformer_block(value, key, query, src_mask)"]
            D6 --> D7["④ LayerNorm + FFN + Residual"]
        end

        D3 --> D4
    end

    E5 -- "enc_out becomes BOTH\nkey and value" --> D6

    style E5 fill:#4a9eff,stroke:#fff,color:#fff
    style D6 fill:#ff6b35,stroke:#fff,color:#fff
```

## Zoomed In: What Happens at the Cross-Attention Step

```mermaid
flowchart LR
    subgraph ENC_OUT["From Encoder"]
        V["enc_out → value"]
        K["enc_out → key"]
    end

    subgraph DEC_SELF_ATT["From Decoder Step ②"]
        Q["decoder self-attn output → query"]
    end

    subgraph CROSS["TransformerBlock → SelfAttention.forward()"]
        direction TB
        C1["Linear projections:\nV = Wv · value\nK = Wk · key\nQ = Wq · query"]
        C2["Split into multi-heads"]
        C3["energy = einsum(Q, K)\n= Q · Kᵀ"]
        C4["Apply src_mask"]
        C5["softmax(energy / √d)"]
        C6["out = attention · V"]
        C7["Concat heads + Linear"]

        C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7
    end

    V --> C1
    K --> C1
    Q --> C1

    style C3 fill:#ff6b35,stroke:#fff,color:#fff
    style C6 fill:#4a9eff,stroke:#fff,color:#fff
```

## Mapping to Your Code

```
Encoder.forward(x, mask)
│
│  for layer in self.layers:
│      out = layer(x, x, x, mask)     ← self-attention (Q=K=V=x)
│                 ▲  ▲  ▲
│                 V  K  Q
│
└──► returns "out" (enc_out)
          │
          │  passed to DecoderBlock as "value" AND "key"
          ▼
DecoderBlock.forward(x, value=enc_out, key=enc_out, src_mask, trg_mask)
│
│  ① self.attention(x, x, x, trg_mask)     ← masked self-attention
│                   ▲  ▲  ▲
│                   V  K  Q  (all from decoder input)
│
│  ② query = dropout(norm(attention + x))   ← residual + norm
│
│  ③ self.transformer_block(value, key, query, src_mask)
│                            ▲      ▲    ▲
│                            │      │    └── from decoder (step ②)
│                            │      └─────── from encoder (enc_out)
│                            └────────────── from encoder (enc_out)
│
│        This is CROSS-ATTENTION:
│        Decoder queries "ask questions"
│        Encoder keys/values "provide answers"
│
└──► returns final output
```

> [!IMPORTANT]
> The **same `SelfAttention` class** handles both self-attention and cross-attention. The difference is purely in **what you pass as arguments**:
> - **Self-attention:** Q, K, V all come from the same source
> - **Cross-attention:** Q comes from decoder, K and V come from encoder

