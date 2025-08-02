import torch

import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    """
    Neural network that uses syllables and words to label said syllables.
    """
    def __init__(self, syll_vocab_size, word_vocab_size, tagset_size, PAD_SYL, PAD_WORD, 
                 syll_emb_dim=64, word_emb_dim=32, hidden_dim=128):
        super().__init__()
        self.syll_emb = nn.Embedding(syll_vocab_size, syll_emb_dim, padding_idx=PAD_SYL)
        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=PAD_WORD)
        
        self.lstm = nn.LSTM(syll_emb_dim + word_emb_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, syll_input, word_input, tags=None, mask=None):
        syll_embed = self.syll_emb(syll_input)
        word_embed = self.word_emb(word_input)
        
        combined = torch.cat([syll_embed, word_embed], dim=2)
        lstm_out, _ = self.lstm(combined)
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)
