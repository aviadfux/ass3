import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import gen_examples as gen
from random import shuffle
from bilstmTrain import Data_holder




def predict_test(model,model_file,data_holder,output_file,repr,use_gpu):
    model.load_state_dict(torch.load(model_file))
    model.eval()
    f=open(output_file,"w")
    for sentence in data_holder.test_sentences:
        input= torch.tensor(sentence,dtype=torch.long)
        if use_gpu:
            input=Variable(input.cuda())
        hidden1,hidden2=model.init_hidden()
        predict=model(input,hidden1,hidden2,repr)
        for s,p in zip(sentence,predict):
            p=data_holder.index_to_label[p]
            f.write(s + " "+ p + "\n")
        f.close()
def main():

    repr= sys.argv[1]
    model_file= sys.argv[2]
    input_file= sys.argv[3]
    output_file= sys.argv[4]


