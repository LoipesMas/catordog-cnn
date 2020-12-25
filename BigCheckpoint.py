import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

IMG_SIZE = 100

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 128, 3)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        
        x = torch.randn((IMG_SIZE, IMG_SIZE)).view(-1,1,IMG_SIZE, IMG_SIZE)
        fc1_size = self.convs(x).size()[1]
        
        self.fc1 = nn.Linear(fc1_size, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 2)
        
    def convs(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, (2,2))

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.max_pool2d(x, (2,2))

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, (2,2))
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, (2,2))
        
        
        x = torch.flatten(x, 1)
        
        return x

    def forward(self, x):
        x = self.convs(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        x = F.softmax(x, 1)
        return x