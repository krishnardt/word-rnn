# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 05:17:33 2020

@author: krish
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


from collections import Counter
import numpy as np



train_file = 'E://anna.txt'
seq_size = 32 #--> length of each sequence or each line in a batch
batch_size = 16 #--> not of texts per batch with the given sequence size
embedding_size = 64  #--> size of the embedding
lstm_size = 64 #--> hidden units
n_layers = 2 #--> layers can be increased to more
gradients_norm=5
#no of cycles including all batches is 50, 
#model is subjected to the no of epochs,hidden_layers, hidden units and embedding sizes
epochs = 50
#starting words of the predicting text.
initial_words=['I']
predict_top_k=5
#based upon this length, the no of tokens to be predicted is generated.
predicting_text_tokens_length = 100
checkpoint_path='checkpoint'



with open(train_file, 'r', encoding='utf-8') as f:
    text = f.read()


text = text.split()
print(text[:100])
print(len(text))


word_counts = Counter(text)
print(len(word_counts))
sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
vocab_to_int = {w: k for k, w in int_to_vocab.items()}
n_vocab = len(int_to_vocab)

print('Vocabulary size', n_vocab)

int_text = [vocab_to_int[w] for w in text]
num_batches = int(len(int_text) / (seq_size * batch_size))
in_text = int_text[:num_batches * batch_size * seq_size]
out_text = np.zeros_like(in_text)
out_text[:-1] = in_text[1:]
out_text[-1] = in_text[0]
in_text = np.reshape(in_text, (batch_size, -1))
out_text = np.reshape(out_text, (batch_size, -1))


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        #print(in_text[:, i:i+seq_size])
        #print(out_text[:, i:i+seq_size])
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]





class RNN(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNN, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, n_layers, dropout = 0.5, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(lstm_size, n_vocab)
    
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        
        logits = self.dense(output)

        return logits, state
    
    def zero_state(self, batch_size):
        weight = next(self.parameters()).data
        
        if (device):
            hidden = (weight.new(n_layers, batch_size, self.lstm_size).zero_().cuda(),
                  weight.new(n_layers, batch_size, self.lstm_size).zero_().cuda())
        else:
            hidden = (weight.new(n_layers, batch_size, self.lstm_size).zero_(),
                      weight.new(n_layers, batch_size, self.lstm_size).zero_())
        
        return hidden
        
#        return (torch.zeros(1.0, batch_size, self.lstm_size),
#                torch.zeros(1.0, batch_size, self.lstm_size))
#    


def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#in_text, out_text = batch_size, seq_size


net = RNN(n_vocab, seq_size,
                    embedding_size, lstm_size)
net = net.to(device)
criterion, optimizer = get_loss_and_train_op(net, 0.01)


def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])
    
    for _ in range(predicting_text_tokens_length):
        ix = torch.tensor([[choice]]).type(torch.LongTensor).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    print(' '.join(words))



iteration = 0

for e in range(epochs):
    batches = get_batches(in_text, out_text, batch_size, seq_size)
    state_h, state_c = net.zero_state(batch_size)
    
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    print(e, iteration)
    for x, y in batches:
        
        iteration += 1
        
        net.train()
        
        optimizer.zero_grad()
        #print(x.shape)
        #print(y.shape)
        #break
        
        x = torch.tensor(x).type(torch.LongTensor).to(device)
        y = torch.tensor(y).type(torch.LongTensor).to(device)
        
        logits, (state_h, state_c) = net(x, (state_h, state_c))
        loss = criterion(logits.transpose(1, 2), y)
        
        state_h = state_h.detach()
        state_c = state_c.detach()
        
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
        
        
        if iteration % 100 == 0:
            print('Epoch: {}/{}'.format(e, epochs),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

        if iteration % 1000 == 0:
            predict(device, net, initial_words, n_vocab,
                        vocab_to_int, int_to_vocab, top_k=predict_top_k)
            torch.save(net.state_dict(),
                           'checkpoint_pt/model-{}.pth'.format(iteration))


        