
import math
import torch
from torch import nn

class Attention():
    pass

class MultiheadAttention():
    pass

class Decoder():
    pass

class Encoder():
    """
    Stack of N identical layers. Each layer has two sublayers, the first being multi-head attention
    """
    
    pass

class Transformer():
    pass


if __name__ == "__main__":
    
    # I leave the office, take the stairs
    vocabulary = {
        "i": torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float16),
        "leave": torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float16),
        "the": torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float16),
        "office": torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float16),
        "take": torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float16),
        "stairs": torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float16)
    }
    
    # variables
    vocab_size = vocabulary.__len__()
    d_model = 512
    d_k = 64
    d_v = 64
    
    
    # stack input tokens into matrix
    sentence = torch.stack([vocabulary["i"], vocabulary["leave"], vocabulary["office"]])
    print(sentence.shape) # (tokens x vocab size)
    
    # linearly project input tokens
    embedding_linear_projection = nn.Linear(vocab_size, d_model, dtype=torch.float16)  # (vocab size x d_model)
    embedding = embedding_linear_projection(sentence) # (tokens x d_model)
    print(embedding.shape)
    
    # normally, positional encoding would go here, but I will skip it for a moment
    
    
    
    
    ### --- a single layer ---
    
    ## -- multi-head attention --
    
    # do prelayer normalization
    layer_norm = nn.LayerNorm(d_model) 
    embedding_normalization = layer_norm(embedding)
        
    # attention
    W_q = nn.Linear(d_model, d_k, dtype=torch.float16) # (d_model x d_k)
    W_k = nn.Linear(d_model, d_k, dtype=torch.float16) # (d_model x d_k)
    W_v = nn.Linear(d_model, d_v, dtype=torch.float16) # (d_model x d_k)
    
    Q = W_q(embedding_normalization) # (tokens x d_k)
    K = W_k(embedding_normalization) # (tokens x d_k)
    V = W_v(embedding_normalization) # (tokens x d_k)
    
    scale_factor = torch.sqrt(torch.tensor(d_k))
    
    K_transpose = torch.transpose(K, 0, 1) # # (d_k x tokens)
    T = Q@K_transpose / scale_factor# (tokens x tokens)
    
    softmax = nn.Softmax(dim=1) # normalize across rows ! needs to be reviewed
    
    attention = softmax(T) @ V # (tokens x d_k)
    print(attention.shape)
    
    multi_head_attention = torch.cat([attention, attention, attention, attention, attention, attention, attention, attention], dim=1)
    
    ## -- residual connections --
    X = embedding + multi_head_attention # (tokens x d_model)
    
    # normalize
    X_normalized = layer_norm(X)
    
    # feed forward network
    feed_forward = nn.Sequential(
        nn.Linear(d_model, d_model*2, dtype=torch.float16),
        nn.Linear(d_model*2, d_model, dtype=torch.float16)
    )
    
    feed = feed_forward(X_normalized) # (tokens x d_model)
    
    # residual connections 
    X_out = X + feed # (tokens x d_model)
    
    print(X_out.shape)
    