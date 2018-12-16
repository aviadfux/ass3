import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import gen_examples as gen
from random import shuffle
import bilstmTrain as bt
EMBEDDING_DIM= bt.EMBEDDING_DIM
HIDDEN_DIM=bt.HIDDEN_DIM
use_gpu=bt.use_gpu


def predict_test(model,model_file,data_holder,output_file,repr,output_size,use_gpu):
    model.load_state_dict(torch.load(model_file))
    model.eval()
    f=open(output_file,"w")
    for sentence in data_holder.test_sentences:
        input= torch.tensor(sentence,dtype=torch.long)
        if use_gpu:
            input=Variable(input.cuda())
        else:
            input=Variable(input)
        hidden=model.init_hidden()
        predict=model(input,hidden,repr,output_size)
        for s,p in zip(sentence,predict):
            p=data_holder.index_to_label[p]
            f.write(s + " "+ p + "\n")
    f.close()
def main():

    repr= sys.argv[1]
    model_file= sys.argv[2]
    input_file= sys.argv[3]
    output_file= sys.argv[4]
    train_file=sys.argv[5]

    data_holder = bt.Data_holder(train_file)
    data_holder.load_test_data(input_file)
    prefix_vocab_size = len(data_holder.prefix_to_index)
    suffix_vocab_size = len(data_holder.suffix_to_index)
    char_vocab_size = len(data_holder.char_to_index)
    word_vocab_size = len(data_holder.word_to_index)
    output_size = len(data_holder.label_to_index)
    model = bt.biLSTM(data_holder, EMBEDDING_DIM, 1, HIDDEN_DIM, prefix_vocab_size, suffix_vocab_size, char_vocab_size,
                   word_vocab_size, output_size)
    predict_test(model,model_file,data_holder,output_file,output_size,repr)

if __name__ == '__main__':
    main()