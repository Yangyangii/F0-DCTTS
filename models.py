from config import ConfigArgs as args
import torch
import torch.nn as nn
from networks import *
from torch.nn.utils import weight_norm as norm

import data
import layers as ll

class DCTTS(nn.Module):
    """
    DCTTS
    Args:
        L: (N, Tx) text
        S: (N, Ty/r, n_mels) previous audio
    Returns:
        Y: (N, Ty/r, n_mels)
    """
    def __init__(self, args):
        super(DCTTS, self).__init__()
        self.args = args
        self.embed = nn.Embedding(len(data.get_vocab(args.lang)), args.Ce, padding_idx=0)
        self.TextEnc = TextEncoder(d_in=args.Ce, d_out=args.Cx*2, d_hidden=args.Cx*2)
        self.AudioEnc = AudioEncoder(d_in=args.n_mels, d_out=args.Cx, d_hidden=args.Cx)
        self.Attention = DotProductAttention(d_hidden=args.Cx)
        self.AudioDec = AudioDecoder(d_in=args.Cx*2, d_out=args.n_mels, d_hidden=args.Cy)
        self.PostNet = PostNet(d_in=args.n_mels, d_out=args.n_mels, d_hidden=args.Cx)
        self.F0Enc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, args.Cx*2),
            nn.Tanh(),
        )
    
    def forward(self, L, S, f0):
        L = self.embed(L).transpose(1,2) # -> (N, Cx, Tx) for conv1d
        S = S.transpose(1,2) # (N, n_mels, Ty/r) for conv1d
        K, V = self.TextEnc(L) # (N, Cx, Tx) respectively
        
        Q = self.AudioEnc(S) # -> (N, Cx, Ty/r)
        R, A = self.Attention(Q, K, V) # -> (N, Cx, Ty/r)
        R_ = torch.cat((R, Q), 1) # -> (N, Cx*2+Df, Ty/r)

        if self.args.f0_mode:
            F = self.F0Enc(f0.unsqueeze(-1)).transpose(1, 2) # (N, Df, Ty)
            R_ = R_ + F
        
        Y = self.AudioDec(R_) # -> (N, n_mels, Ty/r)
        P = self.PostNet(Y)
        return Y.transpose(1, 2), P.transpose(1, 2), A # (N, Ty/r, n_mels)

    def synthesize(self, L, GO_mel, f0):
        L = self.embed(L).transpose(1,2) # -> (N, Cx, Tx) for conv1d
        K, V = self.TextEnc(L) # (N, Cx, Tx) respectively
        F = self.F0Enc(f0.unsqueeze(-1)).transpose(1, 2) # (N, Df, Ty)
        S = GO_mel.transpose(1,2) # (N, n_mels, Ty/r) for conv1d
        
        for t in range(S.shape[-1]-1):
            Q = self.AudioEnc(S) # -> (N, Cx, Ty/r)
            R, A = self.Attention(Q, K, V) # -> (N, Cx, Ty/r)
            R_ = torch.cat((R, Q), 1) # -> (N, Cx*2, Ty/r)
            R_ = R_ + F
            
            Y = self.AudioDec(R_) # -> (N, n_mels, Ty/r)
            S[:, :, t+1] = Y[:, :, t]
        P = self.PostNet(S)
        return Y.transpose(1, 2), P.transpose(1,2), A # (N, Ty/r, n_mels)
    
    def custom_load_state_dict(self, pretrained_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
