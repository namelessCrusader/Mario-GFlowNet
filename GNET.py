import torch.nn as nn
import torch
class FlowMatching(nn.Module):
    def __init__(self, forward, backward,num_tiles):
        super().__init__()
        self.forward_policy = forward
        self.backward_policy = backward
        self.num_tiles = num_tiles

    def normalize(self,state):
        return state / self.num_tiles

    def forward(self, x):
        x = self.normalize(x)
        F = self.forward_policy(x).exp()
        return F
    
    # Backward Policy
    # Update as needed
    def backward(self,state):
        return self.backward_policy(state)  
    
class TrajectoryBalance(nn.Module):
  def __init__(self, forward_policy,backward_policy,num_tiles):
    super().__init__()
    # The input dimension is 6 for the 6 patches.
    self.forward_policy = forward_policy
    self.backward_policy = backward_policy
    self.num_tiles = num_tiles
    # log Z is just a single number
    self.logZ = nn.Parameter(torch.ones(1))

  def normalize(self,state):
      return state / (self.num_tiles * 2)
  
  def forward(self, x):
    x = self.normalize(x)
    logits = self.forward_policy(x)

    P_F = logits[..., :self.num_tiles].exp()
    P_B = logits[..., self.num_tiles:].exp()
    return P_F, P_B

  def backward(self,state):
    return self.backward_policy(state)  
