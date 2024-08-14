import torch.nn as nn
import torch
import math

#TODO1

class Attention(nn.Module):
    def __init__(self, dropout_rate, d_k):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.d_k = d_k
        
    def forward(self, Q, K, V):
        # Q,K,V: (batch, num_Head, n_token, d_k) or (batch, num_Head, n_token, d_v)
        dot_product = Q @ K.transpose(-2,-1) # (batch, n_token, n_token)
        gate = torch.nn.functional.softmax(dot_product / (self.d_k ** -0.5), dim=-1) # 第i個query對所有的key的權重加總為1
        gate = self.dropout(gate)
        attention = gate @ V # 權重*value，每一個row代表一個query的結果。 output: (batch, num_Head, n_token, d_v) 
        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = dim 
        self.d_k = dim // num_heads
        self.d_v = dim // num_heads
        self.num_heads = num_heads
        self.dropout_rate = attn_drop
        
        self.mul_W_Q = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False) # 把W^i全部橫向塞在一起，才能平行運算，所以是num_head * d_k
        self.mul_W_K = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False) 
        self.mul_W_V = nn.Linear(self.d_model, self.num_heads * self.d_v, bias=False)
        self.mul_W_O = nn.Linear(self.d_v*num_heads, self.d_model, bias=False)
        self.attention = Attention(attn_drop, self.d_k)
    
    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, n_token, dim = x.shape
        Q = self.mul_W_Q(x).view(batch_size, self.num_heads, n_token, self.d_k) # 乘完之後，把他重構成heads個
        K = self.mul_W_K(x).view(batch_size, self.num_heads, n_token, self.d_k)
        V = self.mul_W_V(x).view(batch_size, self.num_heads, n_token, self.d_v)
        attention = self.attention(Q,K,V) # shape: (batch, num_Head, n_token, d_v)   
        concat_attention = attention.view(batch_size,  n_token, self.d_v * self.num_heads) # 把他橫向接回來, d_model剛好是d_v*num_Head
        output = self.mul_W_O(concat_attention)
        return output
        
class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential): 
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module): # encoder block
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    