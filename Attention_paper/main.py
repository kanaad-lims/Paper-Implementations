## Running an example of the transformer code
import torch
from transformer import Transformer

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 1]]).to(device) # source sequence
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2]]).to(device) # target sequence

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
  
    model=  Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx
    ).to(device)

    out = model(src, trg[:, :-1])
    print(out.shape)

