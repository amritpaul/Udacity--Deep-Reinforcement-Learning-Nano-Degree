import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)                 # Fully connected layer with state_size as input and fc1_units as output
        self.fc2 = nn.Linear(fc1_units, fc2_units)                  # Fully connected layer with fc1_units as input and fc2_units as output
        self.fc3 = nn.Linear(fc2_units, action_size)                # Fully connected layer with fc2_units as input and action_size as output

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))                                 # Pass the state to fully connected layer (fc1) and apply ReLU activation function
        x = F.relu(self.fc2(x))                                     # Pass the output to fully connected layer (fc2) and apply ReLU activation function
        return self.fc3(x)                                          # Pass the output to fully connected layer (fc3) and return 
    
    
