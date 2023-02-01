import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


class VGGTrainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_dl=None,  # Validation (or test) data set
                 cuda=True  # Whether to use the GPU
                 ):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_dl
        self._cuda = cuda

        if self._cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, iteration, valid_accuracy):
        torch.save({'state_dict': self._model.state_dict()},
                   'checkpoints/checkpoint_{:04d}_{:.3f}.ckp'.format(iteration, valid_accuracy))

    def restore_checkpoint(self, checkpointfile):
        ckp = torch.load(checkpointfile, 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._model.zero_grad()

        # -propagate through the network
        predictions = self._model(x)

        # -calculate the loss
        # print('predictions:', predictions)
        y = torch.argmax(y, -1)
        # print('y:', y)
        loss = self._crit(predictions, y)

        # -compute gradient by backward propagation
        loss.backward()

        # -update weights
        if self._optim is not None:
            self._optim.step()

        # -return the loss
        return loss

    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        predictions = self._model(x)
        y = torch.argmax(y, -1)

        loss = self._crit(predictions, y)

        # return the loss and the predictions
        return loss, predictions

    def train_epoch(self):

        # set training mode
        self._model.train()
        # iterate through the training set
        train_dl = DataLoader(self._train_dl, batch_size=32, shuffle=True)

        loss = 0
        for x, y in tqdm(train_dl):
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            # perform a training step
            loss += self.train_step(x, y)
        # calculate the average loss for the epoch and return it

        return loss / len(train_dl)

    def val_test(self):

        # set eval mode
        self._model.eval()

        # disable gradient computation
        with torch.no_grad():

            # iterate through the validation set
            valid_dl = DataLoader(self._val_test_dl, batch_size=32, shuffle=True)

            # perform a validation step
            loss = 0
            pred_list = []
            y_list = []
            for x, y in tqdm(valid_dl):
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()

                loss_batch, predictions = self.val_test_step(x, y)
                loss += loss_batch

                # save the predictions and the labels for each batch
                y_list = np.append(y_list, y.cpu())

                pred_list = np.append(pred_list, np.around(predictions.cpu()))

        # calculate the average loss and average metrics of your choice. You might want to calculate these
        # metrics in designated functions
        loss = loss / len(valid_dl)
        accuracy = accuracy_score(y_list, pred_list)
        print("accuracy = ", accuracy)

        # return the loss and print the calculated metrics
        return loss, accuracy

    def fit(self, epochs=-1):

        # create a list for the train and validation losses, and create a counter for the epoch
        train_losses = []
        valid_losses = []
        accuracies = []
        epoch_id = 0

        while True:
            print("\nepoch: ", epoch_id)

            '''
            if epoch_id % 25 == 0:
                for param_group in self._optim.param_groups:
                    if param_group['lr']>0.0001:
                        param_group['lr'] = param_group['lr'] * 0.5
            '''

            # stop by epoch number
            if epoch_id >= epochs:
                break

            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            print("train loss = ", train_loss)
            valid_loss, accuracy = self.val_test()
            print("valid_loss = ", valid_loss)

            # append the losses to the respective lists
            train_losses = np.append(train_losses, train_loss.cpu().detach())
            valid_losses = np.append(valid_losses, valid_loss.cpu().detach())
            accuracies = np.append(accuracies, accuracy)

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if epoch_id % 10 == 0:
                self.save_checkpoint(epoch_id, accuracy)

            # check whether early stopping should be performed using the early stopping criterion and stop if so

            epoch_id += 1


if __name__ == '__main__':
    from Model_VGG16 import VGGnet
    import pandas as pd
    from data import CAPTCHADataset

    train_dataset = pd.read_csv('data_train.csv', sep=';')
    train_dl = CAPTCHADataset(train_dataset, 'train')

    valid_dataset = pd.read_csv('data_valid.csv', sep=';')
    valid_dl = CAPTCHADataset(train_dataset, 'valid')

    net = VGGnet(fine_tuning=True, num_classes=12)

    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    criterion = torch.nn.CrossEntropyLoss()

    # set up the optimizer (see t.optim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)  # lr=0.001

    # create an object of type Trainer and set its early stopping criterion
    trainer = VGGTrainer(net, criterion, optimizer, train_dl, valid_dl, cuda=True)

    trainer.fit(20)
