from encoder import EncoderBlock
from attention import SelfAttention
from transformer_block import TransformerBlock
import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask) # here the key and value is of decoder itself
        query = self.dropout(self.norm(attention + x)) ## x is decoder input
        out  = self.transformer_block(value, key, query, src_mask) # key and value from the encoder block before entering transformer block.

        return out
    

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.device = device
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        x = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            out = layer(x, enc_out, enc_out, src_mask, trg_mask)

        return out