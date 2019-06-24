import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from traitlets.config.tests import test_loader
from sann import SentimentLSTM


class senti_analysis():
    """
        Dataloaders are required to store the trained weights and to retrieve them whenever you predict values
        In this one we are using Dataloaders of tensorflow to make this process easier
    """

    def training(self, features, encoded_labels, vocab_to_int):
        self.net=None
        self.crigterion=None
        self.test_loader=None
        self.valid_loader=None
        self.train_loader=None

        # Split fraction for training and accuracy check
        split_frac = 0.8

        # Total lenght of the fetures to break them for training and testing purpouses
        len_feat = len(features)

        train_x = features[0:int(split_frac * len_feat)]
        train_y = encoded_labels[0:int(split_frac * len_feat)]

        remaining_x = features[int(split_frac * len_feat):]
        remaining_y = encoded_labels[int(split_frac * len_feat):]

        valid_x = remaining_x[0:int(len(remaining_x) * 0.5)]
        valid_y = remaining_y[0:int(len(remaining_y) * 0.5)]

        test_x = remaining_x[int(len(remaining_x) * 0.5):]
        test_y = remaining_y[int(len(remaining_y) * 0.5):]

        # create Tensor datasets
        train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
        test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        # dataloaders
        self.batch_size = 50

        # make sure to SHUFFLE your data
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        self.valid_loader = DataLoader(valid_data, shuffle=True, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_data, shuffle=True, batch_size=self.batch_size)

        # obtain one batch of training data
        dataiter = iter(self.train_loader)
        sample_x, sample_y = dataiter.next()

        print('Telugu Sample input size: ', sample_x.size())  # batch_size, seq_length
        print('Telugu Sample input: \n', sample_x)
        print()
        print('Telugu Sample label size: ', sample_y.size())  # batch_size
        print('Telugu Sample label: \n', sample_y)

        # Instantizing a neural network
        vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding
        output_size = 1
        embedding_dim = 400
        hidden_dim = 256
        n_layers = 2
        self.net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
        # print(self.net)

        # Checking if gpu support is available in this system
        train_on_gpu = torch.cuda.is_available()

        # loss and optimization functions
        lr = 0.001

        self.criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # training params

        epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

        counter = 0
        print_every = 100
        clip = 5  # gradient clipping

        # move model to GPU, if available
        if (train_on_gpu):
            self.net.cuda()

        self.net.train()
        # train for some number of epochs
        for e in range(epochs):
            # initialize hidden state
            h = self.net.init_hidden(self.batch_size)

            # batch loop
            for inputs, labels in self.train_loader:
                counter += 1

                if (train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                self.net.zero_grad()

                # get the output from the model
                inputs = inputs.type(torch.LongTensor)
                output, h = self.net(inputs, h)

                # calculate the loss and perform backprop
                loss = self.criterion(output.squeeze(), labels.float())
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                optimizer.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = self.net.init_hidden(self.batch_size)
                    val_losses = []
                    self.net.eval()
                    for inputs, labels in self.valid_loader:

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        if (train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        inputs = inputs.type(torch.LongTensor)
                        output, val_h = self.net(inputs, val_h)
                        val_loss = self.criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                    self.net.train()
                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))

    def testing(self):

        # Checking if gpu support is available in this system
        train_on_gpu = torch.cuda.is_available()

        # Get test data loss and accuracy

        test_losses = []  # track loss
        num_correct = 0

        # init hidden state
        h = self.net.init_hidden(self.batch_size)

        self.net.eval()
        # iterate over test data
        for inputs, labels in self.test_loader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # get predicted outputs
            inputs = inputs.type(torch.LongTensor)
            output, h = self.net(inputs, h)

            # calculate loss
            test_loss = self.criterion(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())

            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(output.squeeze())  # rounds to the nearest integer

            # compare predictions to true label
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
                correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)

        # -- stats! -- ##
        # avg test loss
        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        # accuracy over all test data
        test_acc = num_correct / len(test_loader.dataset)
        print("Test accuracy: {:.3f}".format(test_acc))

    def prediction(self):
        pass