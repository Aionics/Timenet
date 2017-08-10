import utils
import pandas
import numpy
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import code
import os
import random
from sklearn.preprocessing import normalize

import progressbar

# PARAMS:
CUDA = True
EPOCHS = 3000
DEPTH = 3
HIDDEN_SIZE = 60
FEATURES = 1
BATCH_LENGTH = 0
BATCH_STEP_SIZE = 64
WINDOW_SIZE = 136
DROPOUT = 0.4

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
        encoded_vec = encoded_seq[1]     # [3 x 200 x 32] (hlt)

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
            ht2 = self.gru_cell2( self.droput(ht1) , ht2 )      # [200 x 32]
            ht3 = self.gru_cell3( self.droput(ht2) , ht3 )      # [200 x 32]
            ctl_out += [ self.linear( self.dropout(ht3) ) ]    # [200 x 1]

        ctl_out = torch.stack(ctl_out)        # [2000 x 200 x 1]
        ct_out = ctl_out.transpose(0,1)        # [200 x 2000 x 1]

        return ct_out

print('preparing data')
datasetTrain = []
datasetTest = []
availableDatasets = [
    "synthetic_control",
    "PhalangesOutlinesCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "ElectricDevices",
    "MedicalImages",
    "Swedish_Leaf",
    "Two Patterns",
    "ECG5000",
    "ECGFiveDays",
    "Wafer",
    "ChlorineConcentration",
    "Adiac",
    "Strawberry",
    "Cricket_X",
    "Cricket_Y",
    "Cricket_Z",
    "uWaveGestureLibrary_X",
    "uWaveGestureLibrary_Y",
    "uWaveGestureLibrary_Z",
    "yoga",
    "FordA",
    "FordB",
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


net = model()
net.double()
net.load_state_dict(torch.load('models/model-sae3-checkpoint.pt'))
if CUDA: net.cuda()

criterion = nn.MSELoss()
learning_rate = 0.006
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

lowest_loss_per_epoch = 1000

for i in range(EPOCHS):
    loss_to_show = 0
    losses_per_epoch = numpy.array([])
    random.shuffle(datasetTrain)


    for file in range(len(datasetTrain)):
        current_dataset = datasetTrain[file]
        setps_per_epoch = len(current_dataset) / BATCH_STEP_SIZE


        for j in range(setps_per_epoch):
            j = j * BATCH_STEP_SIZE
            starts_from = j
            ends_at = min(j + BATCH_STEP_SIZE, len(current_dataset))

            local_batch_x = Variable( torch.DoubleTensor( current_dataset[starts_from : ends_at] ), requires_grad=False )
            if CUDA: local_batch_x = local_batch_x.cuda()

            optimizer.zero_grad()
            local_batch_x_reversed = local_batch_x.data.cpu().numpy() if CUDA else local_batch_x.data.numpy()
            local_batch_x_reversed = numpy.flip(local_batch_x_reversed, axis=1).copy()
            local_batch_x_reversed = Variable( torch.from_numpy(local_batch_x_reversed).double(), requires_grad=False )
            if CUDA: local_batch_x_reversed = local_batch_x_reversed.cuda()

            predicted = net(local_batch_x, local_batch_x_reversed)

            optimizer.zero_grad()
            loss = criterion(predicted, local_batch_x_reversed)

            loss_to_show = loss.data.cpu().numpy()[0] if CUDA else loss.data.numpy()[0]
            lossArr = numpy.append(lossArr, [loss_to_show], axis=0)
            losses_per_epoch = numpy.append(losses_per_epoch, [loss_to_show], axis=0)

            print("epoch: %s, total_i: %s , current_length: %s, file: %s/%s, step: %s/%s, loss: %s" % (i, total_iterations ,current_dataset[0].shape[0],file + 1, len(datasetTrain), ends_at, len(current_dataset), loss_to_show))
            loss.backward()
            optimizer.step()
            total_iterations += 1

    loss_per_epoch = numpy.average(losses_per_epoch)
    print('current_loss: %s, lowest_loss: %s' % (loss_per_epoch, lowest_loss_per_epoch))
    if (loss_per_epoch < lowest_loss_per_epoch):
        lowest_loss_per_epoch = loss_per_epoch
        if CUDA:
            torch.save(net.cpu().state_dict(), 'models/model-sae3-lowestlossperepoch-%s-epoch-%s.pt' % (lowest_loss_per_epoch, i))
            net.cuda()
        else:
            torch.save(net.state_dict(), 'models/model-sae3-lowestlossperepoch-%s-epoch-%s.pt' % (lowest_loss_per_epoch, i))
    torch.save(net.state_dict(), 'models/model-sae3-checkpoint.pt')
    numpy.savetxt('loss3.csv', lossArr, delimiter=',')

if CUDA:
    net.cpu()
torch.save(net.state_dict(), 'models/model-sae3.pt')

code.interact(local=locals())
