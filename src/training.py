# -*- coding: utf-8 -*-
#
import torch
import glob
import random
import json
from tqdm import tqdm

from sudachipy import tokenizer
from sudachipy import dictionary
from pymagnitude import Magnitude
from pymagnitude import MagnitudeUtils
from torchtext import data as ttd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import summarize

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

EMBEDDING_DIM = HIDDEN_DIM = 256

def sudachiTokenizer(text: str, tokenizer_obj, mode):
    return [m.surface() for m in tokenizer_obj.tokenize(text, mode)]

def training():
    vectors = Magnitude("E:\\Development\\elasticsearch\\data\\message\\chive-1.2-mc5.magnitude", use_numpy=True)
    print(vectors.length)

    summarized_id_dict = {}
    with open(summarize.SUMMARY_ID_DICT_FILE_PATH, encoding='utf-8', mode='r') as rf:
        summarized_id_dict = json.load(rf)

    all_data = []
    for summary_id_file_path in tqdm(glob.glob('E:\\Development\\messages\\summary\\summary_id_*.csv')):
        with open(summary_id_file_path, encoding='utf-8', mode='r') as rf:
            for txt in rf.readlines():
                all_data.append(np.array([float(n) for n in txt.split(',')]))
      
    n_vocab = len(summarized_id_dict.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epoch = 75000
    model = RNNLM(EMBEDDING_DIM, HIDDEN_DIM, n_vocab).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    dataset = MessageDataSet(np.array(all_data))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(1, n_epoch+1):
        
        if epoch % 10 == 0:
            print("Epoch %i: %.2f" % (epoch, loss.item()))
        
        for batch_idx, batch_data in enumerate(dataloader):
            model.zero_grad()
            model.init_hidden(device)
            
            batch_tensor = torch.tensor(batch_data, device=device, dtype=torch.long)
            input_tensor = batch_tensor[:, :-1]
            target_tensor = batch_tensor[:, 1:].contiguous()
            outputs = model(input_tensor)
            outputs = outputs.view(-1, n_vocab)
            targets = target_tensor.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print("-- Batch %i-%i: %.2f" % (epoch, batch_idx, loss.item()))
                    
        if epoch % 10000 == 0:
            model_name = f"message{EMBEDDING_DIM}_v{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, model_name)

    torch.save(model.state_dict(), "message.pth")


class MessageDataSet:
    def __init__(self, all_data: list):
        self.all_data = all_data

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        return self.all_data[index]

class RNNLM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size=10, num_layers=1):
        super(RNNLM, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=0.5)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=self.num_layers)

        self.output = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, device=None):
        self.hidden_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device=device)

    def forward(self, indices):
        embed = self.word_embeddings(indices) # batch_len x sequence_length x embedding_dim
        drop_out = self.dropout(embed)
        if drop_out.dim() == 2:
            drop_out = torch.unsqueeze(drop_out, 1)
        gru_out, self.hidden_state = self.gru(drop_out, self.hidden_state)# batch_len x sequence_length x hidden_dim
        gru_out = gru_out.contiguous()
        return self.output(gru_out)
    

if __name__ == '__main__':
    training()
