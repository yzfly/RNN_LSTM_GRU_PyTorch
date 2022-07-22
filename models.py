import torch
import torch.nn as nn


class GRUNet(nn.Module):
    def __init__(self, args, input_size=1, hidden_dim=51):
        super(GRUNet, self).__init__()
        if args.gru == 'custom':
            from cells import GRUCell
        else:
            GRUCell = nn.GRUCell
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        self.gru1 = GRUCell(self.input_size, self.hidden_dim)
        self.gru2 = GRUCell(self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_dim, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.hidden_dim, dtype=torch.double)


        for input_t in input.split(1, dim=1):
            h_t = self.gru1(input_t, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):# if we should predict the future
            h_t = self.gru1(output, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class SequenceLSTM(nn.Module):
    def __init__(self):
        super(SequenceLSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class SequenceGRU(nn.Module):
    def __init__(self, input_size=1, hidden_dim=51):
        super(SequenceGRU, self).__init__()
        self.gru1 = nn.GRUCell(1, 51)
        self.gru2 = nn.GRUCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)


        for input_t in input.split(1, dim=1):
            h_t = self.gru1(input_t, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):# if we should predict the future
            h_t = self.gru1(output, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs