import torch.nn as nn
import torch.nn.functional as F
import math
d_model = 512   #please change also the value in build_transformer function


class InputEmbeddings(nn.Module):

    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):          
        #giving x as a [1,seq_len]
        return self.embedding(x).transpose(1,2) * math.sqrt(d_model)
    

class PositionalEncodings(nn.Module):
    def __init__(self,d_model:int,seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        pe = torch.zeros(d_model, seq_len).double()

        for pos in range(seq_len):
            for i in range(0,d_model,2):
                pe[i,pos] = math.sin(pos/((10000)**(i/d_model)))

            for i in range(1,d_model,2):
                pe[i,pos] = math.cos(pos/((10000)**(i/d_model)))
        pe = pe.reshape(1,d_model,seq_len)
        self.pe = pe

    def forward(self):       
        # x is (batch, d_model, seq_len) --> plus with pe
        return self.pe.requires_grad_(False)

        
#in attention-is-all-you-need paper, in tranformer architecture, every layer normalization
#layer has residual connection connected, we could try applying residual connect in layer
#norm class only, but i'll do them separately as for future modification purposes
    
class LayerNormalization(nn.Module):
    def __init__(self,features:int,eps = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.parameter(torch.ones(features))   #so alpha, bias are learnable parameters
        self.bias = nn.parameter(torch.zeros(features))

    def forward(self,x): #x in batch,d_model,seq_len
        mean = x.mean(dim= -2, keepdim= True)
        std = x.std(dim=-2,keepdim=True)

        return ((x-mean)/(std + self.eps))*self.alpha +self.bias


class FeedForwardNetwork(nn.Module):

    def __init__(self,d_ff:int,dropout:float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)


    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self,dropout:float, h:int,d_model:int) -> None:
        super().__init__()
        self.d_model =d_model
        self.h = h
        assert d_model%h==0 , "d_model is not divisible y h(heads)"

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model,bias = False)
        self.w_k = nn.Linear(d_model,d_model,bias = False)
        self.w_v = nn.Linear(d_model,d_model,bias = False)
        self.w_o = nn.Linear(d_model,d_model,bias = False)

        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(self,q,k,v,mask, dropout:nn.Dropout):    #assuming q,k,v given as batch, h, seq_len, d_k
        d_k = q.shape[-1]
        #this will do batch, h,seq_len,d_k --> batch,h,seq_len,seq_len
        attention_scores = (q@(k.transpose(-1,-2)))/(math.sqrt(d_k))
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, 10**(-9))
        
        attention_scores = attention_scores.softmax(dim=-1)

        

        return (attention_scores)@v , attention_scores



    def forward(self,q,k,v,mask): #q,v,k --> batch,d_model,seq_len

        query = self.w_q(q.transpose(-1,-2))
        key = self.w_k(k.transpose(-1,-2))
        value = self.w_v(v.transpose(-1,-2))
        
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        multiattentions, self.attention_scores = MultiHeadAttention.attention(query,key,value,mask,self.dropout)

        self.concatenated_o = multiattentions.reshape(multiattentions.shape[0],multiattentions.shape[2],multiattentions.shape[1]*multiattentions.shape[3])

        return self.w_o(self.concatenated_o)


class ResidualConnection(nn.Module):
    def __init__(self,features:int,dropout:float) -> None:
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        return x + self.dropout(self.norm(sublayer(x)))
    
    
class EncoderBlock(nn.Module):

    def __init__(self, residualconnection : ResidualConnection, self_attention_block : MultiHeadAttention, feedforward : FeedForwardNetwork ) -> None:
        super().__init__()
        self.residualconnection = residualconnection
        self.self_attention_block = self_attention_block
        self.feedforward = feedforward

    def forward(self,x,src_msk):
        x = self.residualconnection(x,lambda x : self.self_attention_block(x,x,x,src_msk))
        x = self.residualconnection(x,self.feedforward)
        return x


class Encoder(nn.Module):
    def __init__(self,nblock:int,features :int,encoderblock : EncoderBlock) -> None:
        super().__init__()
        self.nblock = nblock
        self.encoderblock = encoderblock
        self.norm = LayerNormalization(features)

    def forward(self,x,src_mask):
        for block in range(self.nblock):
            x = self.encoderblock(x,src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, residualconnection : ResidualConnection, maskedmultiheadattention: MultiHeadAttention, crossattentionblock : MultiHeadAttention, feedforward :FeedForwardNetwork) -> None:
        super().__init__()
        self.residualconnection = residualconnection
        self.maskedmultiheadattention = maskedmultiheadattention
        self.crossattentionblock = crossattentionblock
        self.feedforward = feedforward

    def forward(self,x,enc_out,masked_mask, cross_mask):
        x = self.residualconnection(x, lambda x: self.maskedmultiheadattention(x,x,x,masked_mask))
        x = self.residualconnection(x, lambda x: self.crossattentionblock(x,enc_out,enc_out,cross_mask))
        x = self.residualconnection(x,self.feedforward)
        return x

class Decoder(nn.Module):
    def __init__(self,ndblock : int,decoderblock : DecoderBlock, norm : LayerNormalization) -> None:
        super().__init__()
        self.ndblock = ndblock
        self.decoderblock = decoderblock
        self.norm = norm


    def forward(self,x,enc_out,masked_mask,cross_mask):
        for block in range(self.ndblock):
            x = self.decoderblock(x,enc_out,masked_mask,cross_mask)
        return self.norm(x)
    
class LinearLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        x = x.transpose(-1,-2)
        return self.linear1(x)


class Transformers(nn.Module):
    def __init__(self,enc_emb :InputEmbeddings,dec_emb:InputEmbeddings, pos_enc : PositionalEncodings,encoder : Encoder,decoder :Decoder,lin:LinearLayer) -> None:
        super().__init__()
        self.enc_emb = enc_emb
        self.dec_emb = dec_emb
        self.pos_enc = pos_enc
        self.encoder = encoder
        self.decoder = decoder
        self.proj = lin

    def encode(self,x, src_mask):
        x= self.enc_emb(x)
        x = self.pos_enc(x)
        return self.encoder(x,src_mask)

    def decode(self,x,enc_out,masked_mask,cross_mask):
        x = self.dec_emb(x)
        x = self.pos_enc(x)
        return self.decoder(x,enc_out,masked_mask,cross_mask)

    def lin(self,x):
        return self.proj(x)


def build_transformer(d_model:int = 512,vocab_size:int, seq_len : int,features, dropout: float = 0.1, heads: int = 8,d_ff:int = 2048, neblock:int=6 , ndblock:int=6):

    src_emb = InputEmbeddings(d_model,vocab_size)
    dec_emb = InputEmbeddings(d_model,vocab_size)
    pos_enc = PositionalEncodings(d_model,seq_len)

    #Encoder Block
    enc_residualconnection = ResidualConnection(features, dropout)
    enc_selfatt = MultiHeadAttention(dropout, heads, d_model)
    enc_feedforward = FeedForwardNetwork(d_ff,dropout)
    encoderblock = EncoderBlock(enc_residualconnection,enc_selfatt,enc_feedforward)
    #Encoder
    encoder = Encoder(neblock,features,encoderblock)

    #decoder block:
    dec_residualconnection = ResidualConnection()
    dec_feedforward = FeedForwardNetwork(d_ff,dropout)
    dec_maskedatt = MultiHeadAttention(dropout,heads,d_model)
    dec_crossatt = MultiHeadAttention(dropout,heads,d_model)
    decoderblock = DecoderBlock(dec_residualconnection,dec_maskedatt,dec_crossatt,dec_feedforward)
    dec_norm = LayerNormalization(features)
    decoder = Decoder(ndblock,decoderblock,dec_norm)

    lin = LinearLayer(d_model,vocab_size)


    #our transformer:
    transformer = Transformers(src_emb,dec_emb,pos_enc,encoder,decoder,lin)

    for p in transformer.parameters():
        if p.dim() >1:
            nn.init.xavier_uniform_(p)

    return transformer