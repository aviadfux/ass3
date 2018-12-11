import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import gen_examples as gen
from random import shuffle


VOCAB_SIZE= 13
EPOCHS = 2
HIDDEN_DIM = 100
EMBEDDING_DIM=50
HIDDEN_DIM2 = 50
LABEL_SIZE =2
LR= 0.01
use_gpu=torch.cuda.is_available()



def get_letter_index_structure(letter_to_index,index_to_letter,lines):
    for line in lines:
        for l in line.rstrip():
            if str(l) not in letter_to_index:
                letter_to_index[str(l)]= len(letter_to_index)
                index_to_letter.append(l)
    return letter_to_index,index_to_letter

# def letter_to_tensor(letter):
#     tensor = torch.zeros(1,VOCAB_SIZE)
#     tensor[0][letter_to_index(letter)]=1
#     return tensor

def category_to_tensor(category):
    tensor= torch.LongTensor([category])
    return tensor

# def line_to_tensor(line):
#     tensor= torch.zeros(len(line),1,VOCAB_SIZE)
#     for li,letter in enumerate(line.rstrip()):
#         tensor[li][0][letter_to_index(letter)]=1
#     return tensor

def read_examples(file_path):
    with open(file_path,'r') as f:
        data=f.readlines()
    return data

def letter_index_structures(pos_examples,neg_examples):
    pos_data = read_examples(pos_examples)
    neg_data = read_examples(neg_examples)
    letter_to_index = {}
    index_to_letter = []
    letter_to_index,index_to_letter = get_letter_index_structure(letter_to_index,index_to_letter,pos_data)
    letter_to_index, index_to_letter = get_letter_index_structure(letter_to_index, index_to_letter, neg_data)
    return letter_to_index,index_to_letter

def create_data_set(pos_filepath,neg_filepath):
    pos_list=[]
    neg_list=[]
    pos_data=read_examples(pos_filepath)
    neg_data = read_examples(neg_filepath)
    for p in pos_data:
        t= (p.rstrip(),0)
        pos_list.append(t)
    for n in neg_data:
        t=(n.rstrip(),1)
        neg_list.append(t)
    shuffle(neg_list)
    shuffle(pos_list)
    size=round(len(neg_list)*0.85)
    train_set = neg_list[:size] + pos_list[:size]
    test_set = neg_list[size:] + pos_list[size:]
    return train_set,test_set

class LSTMLanguageClassifier(nn.Module):

    def __init__(self, vocab_size,embedding_dim, hidden_dim, label_size,batch_size,use_gpu):
        super(LSTMLanguageClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu=use_gpu
        self.batch_size=batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_output=nn.Linear(hidden_dim,label_size)


    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, input,hidden):
        embeds = self.word_embeddings(input)
        x = embeds.view(len(input), self.batch_size, -1)
        lstm_out, hidden = self.lstm(x, hidden)
        h = lstm_out[-1]
        y=torch.tanh(self.hidden_to_output(h))
        probs = F.softmax(y,dim=1)
        return probs


def run_epoch(data,model,letter_to_index,loss_function,optimizer,train):
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    for sentence, o in data:
        index_list = [letter_to_index[l] for l in sentence]
        input_tensor = torch.LongTensor(index_list)
        output_tensor = category_to_tensor(o)
        if use_gpu:
            input_tensor, output_tensor = Variable(input_tensor.cuda()), output_tensor.cuda()
        else:
            input_tensor = Variable(input_tensor)

        hidden = model.init_hidden()
        output = model(input_tensor,hidden)
        loss = loss_function(output, output_tensor)
        if train:
            loss.backward()
            optimizer.step()
            model.zero_grad()
        _, predicted = torch.max(output.data, 1)
        if predicted[0]==output_tensor[0]:
            total_acc += 1
        total += 1
        total_loss += loss.item()
    return total_acc/total, total_loss/total

def run_model(model,letter_to_index,optimizer,loss_function,train_set,test_set):
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    for epoch in range(EPOCHS):
        shuffle(train_set)
        model.train()
        model.zero_grad()
        accuracy,loss=run_epoch(train_set,model,letter_to_index,loss_function,optimizer,True)
        train_loss_.append(loss)
        train_acc_.append(accuracy)

        shuffle(test_set)
        model.eval()
        accuracy,loss = run_epoch(test_set,model,letter_to_index,loss_function,optimizer,False)
        test_loss_.append(loss)
        test_acc_.append(accuracy)
        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, EPOCHS, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))


def main():

    letter_to_index,index_to_letter=letter_index_structures("pos_examples","neg_examples")
    train_set,test_set= create_data_set("pos_examples","neg_examples")
    model=LSTMLanguageClassifier(VOCAB_SIZE,EMBEDDING_DIM,HIDDEN_DIM,LABEL_SIZE,1,use_gpu)
    if use_gpu:
        model= model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()
    run_model(model,letter_to_index,optimizer,loss_function,train_set,test_set)

if __name__ == '__main__':
     main()