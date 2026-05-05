import torch
import torch.nn as nn 

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size must be divisible by number of heads"

        # Extracting the K, Q and V values using the Linear transformation.
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        
        # Concating all the multi head values.
        self.concat = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0] # Batch size: Number of examples taken in for training.
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] # Number of tokens in each sentence/sequence.

        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        # Split the embedding for applying multihead attention.
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys]) # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)

        out = torch.einsum( "nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.concat(out)

        return out

# class TransformerBlock(nn.Module):
#     def __init__(self, embed_size, heads, dropout, forward_expansion):
#         super(TransformerBlock, self).__init__()
#         self.attention = SelfAttention(embed_size, heads)

#         self.norm1 = nn.LayerNorm(embed_size)
#         self.norm2 = nn.LayerNorm(embed_size)

#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_size, forward_expansion * embed_size),
#             nn.ReLU(),
#             nn.Linear(forward_expansion * embed_size, embed_size)
#         )
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, value, key, query, mask):
#         attention = self.attention(value, key, query, mask)
#         x = self.norm1(attention + query) ## attention + query is residual connection.
#         x = self.dropout(x)
#         forward = self.feed_forward(x)
#         out = self.norm2(forward + x) ## x(attention + query) + forward is residual connection.

#         return out