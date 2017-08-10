import utils
import pandas
import numpy
import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import torch.nn as nn
import torch.nn.functional as F
import sys
import code
import os

import progressbar

# PARAMS:
CUDA = True
EPOCHS = 3000
BATCH_STEP_SIZE = 64

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.FEATURES = 1
        self.HIDDEN_SIZE = 60
        self.DEPTH = 3
        self.DROPOUT = 0.4

        self.gru_e = nn.GRU(self.FEATURES, self.HIDDEN_SIZE, self.DEPTH, batch_first=True, dropout=self.DROPOUT)

        self.gru_cell1 = nn.GRUCell(self.FEATURES, self.HIDDEN_SIZE)
        self.gru_cell2 = nn.GRUCell(self.HIDDEN_SIZE, self.HIDDEN_SIZE)
        self.gru_cell3 = nn.GRUCell(self.HIDDEN_SIZE, self.HIDDEN_SIZE)

        self.linear = nn.Linear(self.HIDDEN_SIZE, self.FEATURES)
        self.dropout = nn.Dropout(p=self.DROPOUT)


    def forward(self, inputs, outputs):
        seq_length = inputs.size()[1]
        outputs_transposed = outputs.transpose(0,1)

        encoded_seq = self.gru_e(inputs)
        encoded_vec = encoded_seq[1]     # [3 x 200 x 60] (hlt)

        ctl_out = []
        ht1 = encoded_vec[2]
        ht2 = encoded_vec[1]
        ht3 = encoded_vec[0]

        c0l = outputs_transposed[0]
        ht1 = self.gru_cell1(c0l, ht1)      # [200 x 32]
        ht2 = self.gru_cell2( self.dropout(ht1) , ht2)      # [200 x 32]
        ht3 = self.gru_cell3( self.dropout(ht2) , ht3)      # [200 x 32]
        ctl_out += [ self.linear( self.dropout(ht3) ) ]    # [200 x 1]

        for t in range(seq_length - 1):
            ht1 = self.gru_cell1( outputs_transposed[t + 1], ht1 )      # [200 x 32]
            ht2 = self.gru_cell2( self.dropout(ht1) , ht2 )      # [200 x 32]
            ht3 = self.gru_cell3( self.dropout(ht2) , ht3 )      # [200 x 32]
            ctl_out += [ self.linear( self.dropout(ht3) ) ]    # [200 x 1]

        ctl_out = torch.stack(ctl_out)        # [2000 x 200 x 1]
        ct_out = ctl_out.transpose(0,1)        # [200 x 2000 x 1]

        return ct_out, encoded_vec.transpose(0,1)

print('preparing data')
datasetTrain = []
datasetTest = []
availableDatasets = [
    # "synthetic_control",
    # "PhalangesOutlinesCorrect",
    "DistalPhalanxOutlineAgeGroup",
    # "DistalPhalanxOutlineCorrect",
    # "DistalPhalanxTW",
    # "MiddlePhalanxOutlineAgeGroup",
    # "MiddlePhalanxOutlineCorrect",
    # "MiddlePhalanxTW",
    # "ProximalPhalanxOutlineAgeGroup",
    # "ProximalPhalanxOutlineCorrect",
    # "ProximalPhalanxTW",
    "ElectricDevices",
    # "MedicalImages",
    # "Swedish_Leaf",
    # "Two Patterns",
    # "ECG5000",
    "ECGFiveDays",
#     "Wafer",
#     "ChlorineConcentration",
    # "Adiac",
#     "Strawberry",
#     "Cricket_X",
    # "Cricket_Y",
#     "Cricket_Z",
#     "uWaveGestureLibrary_X",
    # "uWaveGestureLibrary_Y",
#     "uWaveGestureLibrary_Z",
#     "yoga",
    # "FordA",
#     "FordB",
]



for directory, subdirectories, files in os.walk('datasets'):
    for file in files:
        filename = os.path.join(directory, file)
        if any(datasetName in file for datasetName in availableDatasets):
            if "TRAIN" in file:
                filedata = pandas.read_csv(filename, header=None).values[:,1:]
                if filedata.shape[1] <= 512 and filedata.shape[0] >= 64:
                    normalize(filedata, norm='l2')
                    local_dataset = []
                    for i in range(filedata.shape[0]):
                        local_dataset.append( numpy.expand_dims(filedata[i], axis=2) )
                    datasetTrain.append(local_dataset)

            elif "TEST" in file:
                filedata = pandas.read_csv(filename, header=None).values[:,1:]
                if filedata.shape[1] <= 512 and filedata.shape[0] >= 64:
                    normalize(filedata, norm='l2')
                    local_dataset = []
                    for i in range(filedata.shape[0]):
                        local_dataset.append( numpy.expand_dims(filedata[i], axis=2) )
                    datasetTest.append(local_dataset)
sequences_amount = 0
for file in range(len(datasetTrain)):
    seqs = datasetTrain[file]
    for seq in range(len(seqs)):
        sequences_amount += 1

print("%s sequences have been prepared and shuffled" % sequences_amount)

datasetY = numpy.array([])

net = model()
net.double()
net.load_state_dict(torch.load('models/model-sae3-lowestlossperepoch-0.0111093344694-epoch-119.pt'))
net.eval()
if CUDA:
    net.cuda()


output_encoded = numpy.array([[[0]*60] * 3])
classes = numpy.array([])
for i in range(1):
    classI=-1
    for file in range(len(datasetTrain)):
        current_dataset = datasetTest[file]
        classI+=1

        setps_per_epoch = len(current_dataset) / BATCH_STEP_SIZE
        for j in range(setps_per_epoch):
            j = j * BATCH_STEP_SIZE
            starts_from = j
            ends_at = min(j + BATCH_STEP_SIZE, len(current_dataset))

            local_batch_x = Variable( torch.DoubleTensor( current_dataset[starts_from : ends_at] ), requires_grad=False )
            if CUDA: local_batch_x = local_batch_x.cuda()

            local_batch_x_reversed = local_batch_x.data.cpu().numpy() if CUDA else local_batch_x.data.numpy()
            local_batch_x_reversed = numpy.flip(local_batch_x_reversed, axis=1).copy()
            local_batch_x_reversed = Variable( torch.from_numpy(local_batch_x_reversed).double(), requires_grad=False )
            if CUDA: local_batch_x_reversed = local_batch_x_reversed.cuda()

            predicted, encoded = net(local_batch_x, local_batch_x_reversed)


            output_encoded = numpy.concatenate((output_encoded, encoded.cpu().data.numpy()))
            classes = numpy.concatenate((classes, [classI] * encoded.size()[0]))


hidden_state = numpy.zeros((output_encoded.shape[0], 180))
for batch in range(output_encoded.shape[0]):
    hidden_state[batch] = numpy.concatenate(output_encoded[batch])

print(hidden_state.shape)
model = TSNE(n_components=2, random_state=2)
clusters = model.fit_transform(hidden_state)

numpy.savetxt('datasets/clusters3.csv', clusters, delimiter=',')
numpy.savetxt('datasets/class3.csv', classes, delimiter=',')

code.interact(local=locals())
