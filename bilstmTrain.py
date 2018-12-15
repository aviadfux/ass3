import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import gen_examples as gen
from random import shuffle

UNKNOWN= "__UNKNOWWN__"
EMBEDDING_DIM=5
HIDDEN_DIM=10
EPOCHS=5
SENTENCE_BATCH=500
LR=0.01
use_gpu=torch.cuda.is_available()

class data_holder:

    def __init__(self,train_filepath,test_filepath,dev_filepath=None):
        self.file_path = train_filepath
        self.word_to_index={UNKNOWN:0}
        self.index_to_word=[UNKNOWN]
        self.label_to_index={}
        self.index_to_label = []
        self.char_to_index = {}
        self.index_to_char = []
        self.prefix_to_index={}
        self.index_to_prefix = []
        self.suffix_to_index={}
        self.index_to_suffix=[]
        self.sentences= []
        self.dev_sentences=[]
        self.update_data_structues()
        if dev_filepath != None:
            self.load_dev_data(dev_filepath)
        self.load_test_data(test_filepath)

    def update_data_structues(self):
        f=open(self.file_path,"r")
        for i,line in enumerate(f):
            if line == '\n':
                self.sentences.append(sentence)
                sentence= []
            else :
                word,label = line.split()
                sentence.append((word,label))
                if label not in self.label_to_index:
                    self.label_to_index[label]=len(self.label_to_index)
                    self.index_to_label.append(label)
                if word not in self.word_to_index:
                    self.word_to_index[word]=len(self.word_to_index)
                    self.index_to_word.append(word)
                for char in word:
                    if char not in self.char_to_index:
                        self.char_to_index[char]=len(self.char_to_index)
                        self.index_to_char.append(char)
                pref=word[:3]
                if pref not in self.prefix_to_index:
                    self.prefix_to_index[pref]=len(self.prefix_to_index)
                    self.index_to_prefix.append(pref)
                suff=word[-3:]
                if suff not in self.suffix_to_index:
                    self.suffix_to_index[suff]=len(self.suffix_to_index)
                    self.index_to_suffix.append(suff)
        f.close()
    def word_to_index_f(self,word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        else:
            return 1

    def load_dev_data(self,file_path):
        f = open(file_path, "r")
        for i, line in enumerate(f):
            if line == '\n':
                self.dev_sentences.append(sentence)
                sentence = []
            else:
                word, label = line.split()
                sentence.append((word, label))
        f.close()

    def load_test_data(self, file_path):
        f = open(file_path, "r")
        for i, line in enumerate(f):
            if line == '\n':
                self.train_sentences.append(sentence)
                sentence = []
            else:
                word, label = line.split()
                sentence.append((word, label))
        f.close()

class biLSTM(nn.Module):
    def __init__(self,data_holder,embedding_dim,batch_size,hidden_dim,prefix_vocab_size,suffix_vocab_size,char_vocab_size,word_vocab_size,output_size):
        super(biLSTM, self).__init__()
        self.batch_size=batch_size
        self.data_holder=data_holder
        self.word_embedding=nn.Embedding(word_vocab_size,embedding_dim)
        self.char_embedding = nn.Embedding(char_vocab_size,embedding_dim)
        self.preffix_embedding= nn.Embedding(prefix_vocab_size,embedding_dim)
        self.suffix_embedding= nn.Embedding(suffix_vocab_size,embedding_dim)

        self.char_lstm = nn.LSTM(embedding_dim,embedding_dim)
        self.lstm_first= nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        self.lstm_second =nn.LSTM(hidden_dim*2,hidden_dim,bidirectional=True)
        self.linear= nn.Linear(hidden_dim,output_size)
        self.linear_d=nn.Linear(embedding_dim*2,embedding_dim)

    def init_hidden(self):
        if self.use_gpu:
            h0_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            h0_2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim*2).cuda())
            c0_2= Variable(torch.zeros(1, self.batch_size, self.hidden_dim*2).cuda())
        else:
            h0_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            h0_2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0_2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0_1, c0_1),(h0_2,c0_2)

    def set_word_input(self,word_input,repr):
        if repr=='a':
            embeder=self.word_embedding(self.data_holder.word_to_index_f(word_input))
            return embeder
        elif repr=='b':
            char_embedders=[]
            for c in word_input:
                char_embedders.append(self.char_embedding(self.data_holder.char_to_index[c]))
            char_lstm_out= self.char_lstm(char_embedders)
            return char_lstm_out[-1]
        elif repr== 'c':
            embeder = self.word_embedding(self.data_holder.word_to_index_f(word_input))
            pref_embeder= self.preffix_embedding(self.data_holder.prefix_to_index(word_input[:3]))
            suff_embeder = self.suffix_embedding(self.data_holder.suffix_to_index(word_input[-3:]))
            embeder=embeder.add(pref_embeder).add(suff_embeder)
            return embeder
        elif repr== 'd':
            embeder=self.word_embedding(self.data_holder.word_to_index_f(word_input))
            char_embedders = []
            for c in word_input:
                char_embedders.append(self.char_embedding(self.data_holder.char_to_index[c]))
            char_lstm_out = self.char_lstm(char_embedders)
            concat_embeder= torch.cat(embeder,char_lstm_out[-1])
            embeder=torch.tanh(self.linear_d(concat_embeder))
            return embeder


    def forward(self, input,first_hidden,second_hidden,repr):
        input=[self.set_word_input(word_input) for word_input in input]
        first_lstm_output,first_hidden=self.lstm_first(input,first_hidden)
        second_lstm_output,second_hidden=self.lstm_second(first_lstm_output[-1],second_hidden)
        output=self.linear[second_lstm_output[-1]]
        props= torch.F.softmax(output,dim=1)
        return props


def run_epoch(data_holder,model,use_gpu,loss_function,optimizer,train,repr):
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    accuracy_list=[]
    if train:
        sentences=data_holder.sentences
    else:
        sentences=data_holder.dev_sentences
    for i,sentence in enumerate(sentences):
        s= [w for w,l in sentence]
        l= [data_holder.label_to_index[l] for w,l in sentence]
        input_tensor = torch.tensor(s,dtype=torch.long)
        output_tensor = torch.tensor(l, dtype=torch.long)
        if use_gpu:
            input_tensor, output_tensor = Variable(input_tensor.cuda()), output_tensor.cuda()
        else:
            input_tensor = Variable(input_tensor)

        hidden1,hidden2 = model.init_hidden()
        output = model(input_tensor,hidden1,hidden2,repr)
        loss = loss_function(output, output_tensor)
        if train:
            loss.backward()
            optimizer.step()
            model.zero_grad()
        _, predicted = torch.max(output.data, 1)
        for pred,out in zip(predicted,output_tensor):
            total+=1
            if pred==out:
                if data_holder.index_to_label[out]== 'O':
                    total -= 1
                else:
                    total_acc +=1
        total_loss += loss.item()
        if not train and i % SENTENCE_BATCH == 0:
                accuracy_list.append(total_acc)
    return total_acc/total, total_loss/total,accuracy_list


def train_model(model,loss_function,train_data_holder,dev_data,lr,epochs,repr,save_model_path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        shuffle(train_data_holder)
        model.train()
        model.zero_grad()
        train_loss, train_acc,_ = run_epoch(train_data_holder,model,use_gpu,loss_function,optimizer,True,repr)
        model.eval()
        dev_loss, dev_acc,acc_list = run_epoch(train_data_holder,model,use_gpu,loss_function,optimizer,True,repr)
        print("{} - train loss {} train-accuracy {} dev loss {}  dev-accuracy {}".format(epoch, train_loss, train_acc,
                                                                                         dev_loss, dev_acc))
        torch.save(model.state_dict(), save_model_path+str(epoch))
    return acc_list





def turn_model_flag_type(repr):
    word_flag = False
    char_flag = False
    pref_suffix_flag = False
    if repr == 'a':
        word_flag = True
    elif repr == 'b':
        char_flag = True
    elif repr == 'c':
        word_flag = True
        pref_suffix_flag = True
    elif repr == 'd':
        word_flag = True
        char_flag = True
    return word_flag,char_flag,pref_suffix_flag

def main():
    repr=sys.argv[1]
    train_file= sys.argv[2]
    model_file=sys.argv[3]
    if len(sys.argv)>=5:
        dev_file = sys.argv[4]

    word_flag, char_flag, pref_suffix_flag=turn_model_flag_type(repr)