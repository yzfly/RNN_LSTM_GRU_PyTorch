import torch
import torch.nn as nn

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_layer = nn.Linear(input_size, 3*hidden_dim)
        self.pre_t_layer = nn.Linear(hidden_dim, 3*hidden_dim)
    
    def forward(self, t_in, pre_t=None):
        # Inputs:
        #       t_in: of shape (batch_size, input_size)
        #       pre_t: of shape (batch_size, hidden_size)
        # Output:
        #       out_state: of shape (batch_size, hidden_size)

        if pre_t is None:
            pre_t = torch.zeros((t_in.shape(0), self.hidden_dim)).to(self.device)
        
        t_hidden = self.t_layer(t_in)
        pre_t_hidden = self.pre_t_layer(pre_t)

        reset_t, update_t, new_t = t_hidden.chunk(3, dim=1)
        reset_pre_t, update_pre_t, new_pre_t = pre_t_hidden.chunk(3, dim=1)

        reset_gate = torch.sigmoid(reset_pre_t + reset_t)
        update_gate = torch.sigmoid(update_pre_t + update_t)
        new_gate = torch.tanh(new_pre_t * reset_gate + new_t)
        
        out_state = update_gate*pre_t + (1-update_gate)*new_gate

        return out_state

        
