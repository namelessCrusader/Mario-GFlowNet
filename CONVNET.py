import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self, length, width,num_actions):
        super().__init__()
        self.length = length
        self.width = width
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = torch.flatten

        self.linear_dims = (self.width - 2) * (self.length - 2) * 2
        self.fc3 = nn.Linear(self.linear_dims, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute((1,0,2,3))
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.act3(self.conv3(x))
        x = self.pool3(x)
        x = self.flat(x, start_dim=1)
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        x = self.fc4(x)
        return x
    
class ConvNetFeatureMaps(nn.Module):
    def __init__(self, length, width,num_actions):
        super().__init__()
        self.length = length
        self.width = width
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(self.num_actions+2, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = torch.flatten

        self.linear_dims = self.width * self.length * 8
        self.fc3 = nn.Linear(self.linear_dims, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, self.num_actions)
        # self.softmax = nn.Softmax

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.flat(x, start_dim=1)
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        x = self.fc4(x)
        return x
    
class ConvNetFeatureMapsTB(nn.Module):
    def __init__(self, length, width,num_actions):
        super().__init__()
        self.length = length
        self.width = width
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(self.num_actions, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = torch.flatten

        self.linear_dims = self.width * self.length * 8
        self.fc3 = nn.Linear(self.linear_dims, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, self.num_actions)
        
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.flat(x, start_dim=1)
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        x = self.fc4(x)
        return x
