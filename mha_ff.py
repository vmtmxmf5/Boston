import torch
import torch.nn as nn
import torch.nn.functional as F
import math


######## 보조함수 ########
def aeq(*args):
    '''
    Assert all argu. have the same value
    '''
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        'Not all arguments have the same value : ' + str(args)


def sequence_mask(lengths, max_len=None):
    '''
    Create a boolean padding mask from sequence lengths
    '''
    batch_size = lengths.numel()       # int type & all dim
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)     # (1, max_len) -> (batch_size, max_len)
            .lt(lengths.unsqueeze(1))) # len 보다 크면 False

# print(torch.LongTensor(((4,5,6,8))).numel())
# print(torch.LongTensor([4,5,6,8]).unsqueeze(1))
# print(torch.arange(0,15).repeat(4,1).lt(torch.LongTensor([4,5,6,8]).unsqueeze(1)))

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) 
        * (x + 0.044715 * torch.pow(x, 3))))

#########################


class PositionwiseFeedForward(nn.Module):
    '''
    d_model : the size of input for the first-layer of the FFN
    d_ff : the hidden layer size of the second-layer of the FFN
    dropout : dropout probability in math [0, 1)
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    '''
    head_count : number of parallel heads
    model_dim : the dim. of keys/values/queries,
                must be divisible by head_count
    dropout : dropout parameter
    '''
    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)
    
    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):
        '''
        key   : (B, k_len, d)
        value : (B, v_len, d)
        query : (B, q_len, d)
        mask  : (B, q_len, k_len)
        
        Returns : (B, q_len, d), (B, q_len, k_len)
        '''
        
        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            # projection : (B, s_len, d) -> (B, nhead, s_len, head_dim)
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            # compute context : (B, nhead, s_len, head_dim) -> (B, s_len, d)
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)
        
        # 1) Project QKV
        if layer_cache is not None:
            if type == 'self':
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
            
                device = key.device
    
                # s_len axis로 결합
                if layer_cache['self_keys'] is not None:
                    key = torch.cat(
                        (layer_cache['self_keys'].to(device), key),
                        dim=2)
                if layer_cache['self_values'] is not None:
                    value = torch.cat(
                        (layer_cache['self_values'].to(device), value),
                        dim=2)
                layer_cache['self_keys'] = key
                layer_cache['self_values'] = value
            elif type == 'context':
                query = self.linear_query(query)
                if layer_cache['memory_keys'] is None:
                    key, value = self.linear_keys(key), \
                                    self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache['memory_keys'], \
                                 layer_cache['memory_values']
                layer_cache['memory_keys'] = key
                layer_cache['memory_values'] = value
            else:
                key, value = self.linear_keys(key), \
                             self.linear_values(value)
                key = shape(key)
                value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)
        
        query = shape(query)
        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3)) # (B, H, q_len, k_len)

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores) # (B, 1 -> H, q_len, k_len)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)

        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context
            
        
