import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.t_layer = nn.Linear(input_size, 3*hidden_dim)
        self.pre_t_layer = nn.Linear(hidden_dim, 3*hidden_dim)
    
    def forward(self, t_in, pre_t=None):
        # Inputs:
        #       t_in: of shape (batch_size, input_size)
        #       pre_t: of shape (batch_size, hidden_size)
        # Output:
        #       out_state: of shape (batch_size, hidden_size)

        if pre_t is None:
            pre_t = torch.zeros((t_in.shape(0), self.hidden_dim)).to(t_in.device)
        
        t_hidden = self.t_layer(t_in)
        pre_t_hidden = self.pre_t_layer(pre_t)

        reset_t, update_t, new_t = t_hidden.chunk(3, dim=1)
        reset_pre_t, update_pre_t, new_pre_t = pre_t_hidden.chunk(3, dim=1)

        reset_gate = torch.sigmoid(reset_pre_t + reset_t)
        update_gate = torch.sigmoid(update_pre_t + update_t)
        new_gate = torch.tanh(new_pre_t * reset_gate + new_t)
        
        out_state = update_gate*pre_t + (1-update_gate)*new_gate

        return out_state

        
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.t_layer = nn.Linear(input_size, 4*hidden_dim)
        self.pre_t_layer = nn.Linear(hidden_dim, 4*hidden_dim)
    
    def forward(self, t_in, pre_t=None):
        # Inputs:
        #       t_in: of shape (batch_size, input_size)
        #       pre_t: of shape (batch_size, hidden_size)
        # Output:
        #       out_state: of shape (batch_size, hidden_size)
        #       out_hidden: of shape (batch_size, hidden_size)

        if pre_t is None:
            pre_t = torch.zeros((t_in.shape(0), self.hidden_dim)).to(t_in.device)
            pre_t = (pre_t, pre_t)
        cell_state, cell_hidden = pre_t

        t_in = self.t_layer(t_in)
        cell_hidden = self.pre_t_layer(cell_hidden)

        forget_t, input_t, update_t, out_t = t_in.chunk(4, dim=1)
        forget_ch, input_ch, update_ch, out_ch = cell_hidden.chunk(4, dim=1)

        forget_gate = torch.sigmoid(forget_t+forget_ch)
        cell_state = cell_state * forget_gate

        input_gate = torch.sigmoid(input_t + input_ch)
        out_state = cell_state + input_gate * torch.tanh(update_t+update_ch)

        out_gate = torch.sigmoid(out_t+out_ch)
        
        out_hidden = out_gate * torch.tanh(out_state)
        
        return (out_state, out_hidden)

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_dim, activation='tanh'):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.t_layer = nn.Linear(input_size, hidden_dim)
        self.pre_t_layer = nn.Linear(hidden_dim, hidden_dim)
        
    
    def forward(self, t_in, pre_t=None):
        # Inputs:
        #       t_in: of shape (batch_size, input_size)
        #       pre_t: of shape (batch_size, hidden_size)
        # Output:
        #       out: of shape (batch_size, hidden_size)

        if pre_t is None:
            pre_t = torch.zeros((t_in.shape(0), self.hidden_dim)).to(t_in.device)
            
        t_in = self.t_layer(t_in)
        pre_t = self.pre_t_layer(pre_t)
        out = t_in + pre_t
        if self.activation == 'tanh':
            out = torch.tanh(out)
        else:
            out = torch.relu(out)

        return out