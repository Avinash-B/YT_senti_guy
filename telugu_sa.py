import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


class senti_analysis():
    """
        Dataloaders are required to store the trained weights and to retrieve them whenever you predict values
        In this one we are using Dataloaders of tensorflow to make this process easier
    """

    def training(self, features, encoded_labels):
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
        batch_size = 50

        # make sure to SHUFFLE your data
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

        # obtain one batch of training data
        dataiter = iter(train_loader)
        sample_x, sample_y = dataiter.next()

        print('Sample input size: ', sample_x.size())  # batch_size, seq_length
        print('Sample input: \n', sample_x)
        print()
        print('Sample label size: ', sample_y.size())  # batch_size
        print('Sample label: \n', sample_y)

    def testing(self):

        pass


class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        #CHecking if gpu support is available
        train_on_gpu = torch.cuda.is_available()

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
