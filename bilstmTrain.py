import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import gen_examples as gen
from random import shuffle
import pylab
import copy
UNKNOWN= "__UNKNOWWN__"
EMBEDDING_DIM=50
HIDDEN_DIM=70
EPOCHS=5
SENTENCE_BATCH=500
LR=0.005
# use_gpu=torch.cuda.is_available()
use_gpu=False
class Data_holder:

    def __init__(self,train_filepath,dev_filepath=None):
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
        self.test_sentences=[]
        self.update_data_structues()
        if dev_filepath != None:
            self.load_dev_data(dev_filepath)

    def update_data_structues(self):
        f=open(self.file_path,"r")
        sentence = []
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
            return 0

    def suffix_to_index_f(self,word):
        if word in self.suffix_to_index:
            return self.suffix_to_index[word]
        else:
            return 0

    def prefix_to_index_f(self,word):
        if word in self.prefix_to_index:
            return self.prefix_to_index[word]
        else:
            return 0

    def load_dev_data(self,file_path):
        f = open(file_path, "r")
        sentence = []
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
        sentence = []
        for i, line in enumerate(f):
            if line == '\n':
                self.train_sentences.append(sentence)
                sentence = []
            else:
                sentence.append(line)
        f.close()

class biLSTM(nn.Module):
    def __init__(self,data_holder,embedding_dim,batch_size,hidden_dim,prefix_vocab_size,suffix_vocab_size,char_vocab_size,word_vocab_size,output_size):
        super(biLSTM, self).__init__()
        self.batch_size=batch_size
        self.data_holder=data_holder
        self.hidden_dim=hidden_dim
        self.embedding_dim=embedding_dim
        self.word_embedding=nn.Embedding(word_vocab_size,embedding_dim)
        self.char_embedding = nn.Embedding(char_vocab_size,embedding_dim)
        self.preffix_embedding= nn.Embedding(prefix_vocab_size,embedding_dim)
        self.suffix_embedding= nn.Embedding(suffix_vocab_size,embedding_dim)

        self.char_lstm = nn.LSTM(embedding_dim,embedding_dim)
        self.lstm_forward1= nn.LSTM(embedding_dim,hidden_dim)
        self.lstm_backward1 = nn.LSTM(embedding_dim,hidden_dim)
        self.lstm_forward2 = nn.LSTM(hidden_dim*2, hidden_dim)
        self.lstm_backward2 = nn.LSTM(hidden_dim*2, hidden_dim)

        self.linear= nn.Linear(hidden_dim*2,output_size)
        self.linear_d=nn.Linear(embedding_dim*2,embedding_dim)

    def init_hidden(self):
        if use_gpu:
            h0_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0_1, c0_1)

    def init_hidden_embedding(self):
        if use_gpu:
            h0_1 = Variable(torch.zeros(1, self.batch_size, self.embedding_dim).cuda())
            c0_1 = Variable(torch.zeros(1, self.batch_size, self.embedding_dim).cuda())
        else:
            h0_1 = Variable(torch.zeros(1, self.batch_size, self.embedding_dim))
            c0_1 = Variable(torch.zeros(1, self.batch_size, self.embedding_dim))
        return (h0_1, c0_1)

    def forward(self, input,hidden,repr,output_size):
        input = input.view(len(input), 1, -1)
        first_hidden_f=hidden
        first_hidden_b = copy.copy(hidden)
        second_hidden_f = copy.copy(hidden)
        second_hidden_b = copy.copy(hidden)
        first_lstm_output_f, first_hidden_f = self.lstm_forward1(input, first_hidden_f)
        first_lstm_output_b, first_hidden_b = self.lstm_backward1(reversed(input), first_hidden_b)
        first_concat=[torch.cat([f,b],dim=1) for f,b in zip(first_lstm_output_f,reversed(first_lstm_output_b))]
        first_concat= torch.stack(first_concat)
        second_lstm_output_f, second_hidden_f = self.lstm_forward2(first_concat, second_hidden_f)
        second_lstm_output_b, second_hidden_b = self.lstm_backward2(reversed(first_concat), second_hidden_b)
        second_concat=[torch.cat([f,b],dim=1) for f,b in zip(second_lstm_output_f,reversed(second_lstm_output_b))]
        second_concat=torch.stack(second_concat)
        predictions=torch.zeros(len(input),1,output_size)
        for i,s in enumerate(second_concat):
            output=self.linear(s)
            probs= torch.softmax(output,dim=1)
            predictions[i]=probs
        return predictions

    def set_word_input(self,word_input,embedding_dim,repr):
        if repr=='a':
            input_tensor= torch.tensor(self.data_holder.word_to_index_f(word_input),dtype=torch.long)
            embeder=self.word_embedding(input_tensor)
            return embeder
        elif repr=='b':
            char_embedders=torch.zeros(len(word_input),embedding_dim)
            for i,c in enumerate(word_input):
                char_tensor=[self.data_holder.char_to_index[c]]
                char_tensor=torch.tensor(char_tensor)
                char_embedders[i]=self.char_embedding(char_tensor)
            if use_gpu:
                char_embedders=Variable(char_embedders.cuda())
            else:
                char_embedders = Variable(char_embedders)
            hidden= self.init_hidden_embedding()
            char_embedders=char_embedders.view(len(char_embedders),1,-1)
            char_lstm_out,hidden= self.char_lstm(char_embedders,hidden)
            return char_lstm_out[-1]
        elif repr== 'c':
            embeder=[self.data_holder.word_to_index_f(word_input)]
            embeder = self.word_embedding(torch.tensor(embeder,dtype=torch.long))
            pref=[self.data_holder.prefix_to_index_f(word_input[:3])]
            pref_embeder= self.preffix_embedding(torch.tensor(pref,dtype=torch.long))
            suff=[self.data_holder.suffix_to_index_f(word_input[-3:0])]
            suff_embeder = self.suffix_embedding(torch.tensor(suff,dtype=torch.long))
            embeder=embeder.add(pref_embeder).add(suff_embeder)
            return embeder
        elif repr== 'd':
            input_tensor = torch.tensor(self.data_holder.word_to_index_f(word_input), dtype=torch.long)
            embeder = self.word_embedding(input_tensor)
            char_embedders = torch.zeros(len(word_input), embedding_dim)
            for i, c in enumerate(word_input):
                char_tensor = [self.data_holder.char_to_index[c]]
                char_tensor = torch.tensor(char_tensor)
                char_embedders[i] = self.char_embedding(char_tensor)
            if use_gpu:
                char_embedders = Variable(char_embedders.cuda())
            else:
                char_embedders = Variable(char_embedders)
            hidden = self.init_hidden_embedding()
            char_embedders = char_embedders.view(len(char_embedders), 1, -1)
            char_lstm_out, hidden = self.char_lstm(char_embedders, hidden)
            out=char_lstm_out[-1].view(embedding_dim)
            concat_embeder= torch.cat([embeder,out],dim=0)
            embeder=torch.tanh(self.linear_d(concat_embeder))
            return embeder
def dev_accuracy(data_holder,model,embedding_dim,output_size,repr):
    total_acc = 0.0
    total = 0.0
    model.eval()
    for i,sentence in enumerate(data_holder.dev_sentences):
        s= [w for w,l in sentence]
        input_tensor = torch.zeros((len(s),embedding_dim)).float()
        for i,w in enumerate(s):
            input_tensor[i]=model.set_word_input(w,embedding_dim,repr)
        l= [data_holder.label_to_index[l] for w,l in sentence]
        output_tensor = torch.tensor(l, dtype=torch.long)
        if use_gpu:
            input_tensor, output_tensor = Variable(input_tensor.cuda()), output_tensor.cuda()
        else:
            input_tensor = Variable(input_tensor)
        hidden=model.init_hidden()
        output = model(input_tensor,hidden,repr,output_size)
        output_tensor=output_tensor.view(len(output_tensor))
        for pred,out in zip(output,output_tensor):
            pred = torch.argmax(pred, 1)
            total+=1
            if pred[0]==out.item():
                if data_holder.index_to_label[out]== 'O':
                    total -= 1
                else:
                    total_acc +=1
    print (str(total_acc/total) + "\n")
    return total_acc/total

def run_epoch(data_holder,model,use_gpu,loss_function,optimizer,embedding_dim,output_size,repr,accuracy_list):
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    sentences=data_holder.sentences
    for i,sentence in enumerate(sentences):
        s= [w for w,l in sentence]
        input_tensor = torch.zeros((len(s),embedding_dim)).float()
        for i,w in enumerate(s):
            input_tensor[i]=model.set_word_input(w,embedding_dim,repr)
        l= [data_holder.label_to_index[l] for w,l in sentence]
        output_tensor = torch.tensor(l, dtype=torch.long)
        if use_gpu:
            input_tensor, output_tensor = Variable(input_tensor.cuda()), output_tensor.cuda()
        else:
            input_tensor = Variable(input_tensor)
        hidden=model.init_hidden()
        output = model(input_tensor,hidden,repr,output_size)
        output_loss=output.view(len(output),-1)
        output_tensor=output_tensor.view(len(output_tensor))
        loss = loss_function(output_loss, output_tensor)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        for pred,out in zip(output,output_tensor):
            pred = torch.argmax(pred, 1)
            total+=1
            if pred[0]==out.item():
                if data_holder.index_to_label[out]== 'O':
                    total -= 1
                else:
                    total_acc +=1
        total_loss += loss.item()
        if  i % SENTENCE_BATCH == 0:
            acc=dev_accuracy(data_holder, model,embedding_dim,output_size,repr)
            accuracy_list.append(acc)
            model.train()
    return total_acc/total, total_loss/total


def train_model(model,loss_function,data_holder,lr,epochs,embedding_dim,output_size,repr,save_model_path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    accuracy_list = []
    for epoch in range(epochs):
        shuffle(data_holder.sentences)
        model.train()
        model.zero_grad()
        train_loss, train_acc = run_epoch(data_holder,model,use_gpu,loss_function,optimizer,embedding_dim,output_size,repr,accuracy_list)
        model.eval()
        print("{} - train loss {} train-accuracy {} ".format(epoch, train_loss, train_acc))

        torch.save(model.state_dict(), save_model_path+str(epoch))
    write_acc_to_file("pos_acc.txt",repr,accuracy_list)



def plotdata(a_data,b_data,c_data,d_data,x_axis_name,y_axis_name,title):
    size=len(a_data)
    x=[]
    for i in range(size):
        x.append((i+1)*5)
    pylab.plot(x,a_data,"-r")
    pylab.plot(x, b_data, "-b")
    pylab.plot(x, c_data, "-g")
    pylab.plot(x, d_data, "-y")
    pylab.title(title)
    pylab.xlabel(x_axis_name)
    pylab.ylabel(y_axis_name)
    pylab.show()


def write_acc_to_file(filename,repr,acc_list):
    f=open(filename,"a")
    f.write("repr   " + repr + "\n")
    for acc in acc_list:
        f.write(str(acc)+ "\n")
    f.write("\n\n")
    f.close()
def main():
    repr=sys.argv[1]
    train_file= sys.argv[2]
    model_file=sys.argv[3]
    if len(sys.argv)>=5:
        dev_file = sys.argv[4]

    data_holder=Data_holder(train_file,dev_file)
    prefix_vocab_size=len(data_holder.prefix_to_index)
    suffix_vocab_size = len(data_holder.suffix_to_index)
    char_vocab_size = len(data_holder.char_to_index)
    word_vocab_size=len(data_holder.word_to_index)
    output_size= len(data_holder.label_to_index)
    model= biLSTM(data_holder,EMBEDDING_DIM,1,HIDDEN_DIM,prefix_vocab_size,suffix_vocab_size,char_vocab_size,word_vocab_size,output_size)
    loss_function=nn.CrossEntropyLoss()
    train_model(model,loss_function,data_holder,LR,EPOCHS,EMBEDDING_DIM,output_size,repr,model_file)


if __name__ == '__main__':
    main()