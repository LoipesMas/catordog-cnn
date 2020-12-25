import os
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

REBUILD_DATA = False

TEST_SIZE = 0.1

class CatOrDog():
    IMG_SIZE = 64
    CATS = "PetImages/0"
    DOGS = "PetImages/1"

    LABELS = {CATS: 0, DOGS: 1}

    training_data = []

    cat_count = 0
    dog_count = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                try:
                    img = cv2.imread(os.path.join(label,f), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    img = img / 255
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                    if label == self.CATS:
                        self.cat_count += 1
                    else:
                        self.dog_count += 1

                except Exception as e:
                    pass
        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data, allow_pickle=True)
        print("Cats:", self.cat_count)
        print("Dogs:", self.dog_count)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(346112, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, 1)
        return x

        
if REBUILD_DATA:
    cat_or_dog = CatOrDog()
    cat_or_dog.make_training_data()

training_data = np.load('training_data.npy', allow_pickle=True)
np.random.shuffle(training_data)
trainset = training_data[int(len(training_data)*(1-TEST_SIZE)):]
testset = training_data[:int(len(training_data)*(TEST_SIZE))]

BATCH_SIZE = 20

trainset_batches = []
while len(trainset) != 0:
    batch = [[],[]]
    for _ in range(BATCH_SIZE):
        if len(trainset) == 0:
            break
        data, trainset = trainset[-1], trainset[:-1]
        X, y = data
        batch[0].append(X)
        batch[1].append(y)
    trainset_batches.append(batch)




net = Net()
loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(0):
    batches_bar = tqdm(trainset_batches)
    for data in batches_bar:
        X, y = data
        X = torch.Tensor(X).view((-1, 1, 64, 64))
        y = torch.Tensor(y)
        net.zero_grad()
        output = net(X)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        batches_bar.set_description_str(f"Loss: {loss.item():7.5}")

correct = 0
total = 0
net.eval()
with torch.no_grad():
    for X, y in tqdm(testset):
        real_class = torch.argmax(torch.Tensor(y))
        X = torch.Tensor(X).view((-1, 1, 64, 64))
        net_out = net(X)[0]  # returns a list, 
        predicted_class = torch.argmax(net_out)

        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))

