from transformer_block import TransformerBlock
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        x = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            out = layer(x, x, x, mask)

        return out ## key and value sent to decoder block(context-aware embeddings)
        