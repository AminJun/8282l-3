import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader

LOUD = False
BATCH_SIZE = 128  # Must be in range (16, 100)
EPOCHS = 12
pre_learn_weights = []
post_learn_weights = []
DATA_SET = 'Adult'


def load_data():
    x = np.load(DATA_SET + '/data.npy')
    y = np.load(DATA_SET + '/labels.npy')
    return train_test_split(x, y, test_size=0.15)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dout(x)
        x = self.fc2(x)
        x = self.prelu(x)
        x = self.out(x)
        x = self.out_act(x)
        return x


def train(my_net, my_optimizer, my_criterion, my_loader, my_device='cpu'):
    my_net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(my_loader):
        inputs, targets = inputs.to(my_device), targets.to(my_device)
        my_optimizer.zero_grad()
        outputs = my_net(inputs)
        loss = my_criterion(outputs, targets)
        loss.backward()
        my_optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Loss: %.3f | ACC: %.3f' % (train_loss, 100. * correct / total))


def test(my_net, my_criterion, my_loader, my_device):
    my_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(my_loader):
            inputs, targets = inputs.to(my_device), targets.to(my_device)
            outputs = my_net(inputs)
            loss = my_criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Loss: %.3f | ACC: %.3f' % (test_loss, 100. * correct / total))


class ThreeLoader(Dataset):
    def __init__(self, x_arr, y_arr):
        self.x_arr = x_arr
        self.y_arr = y_arr

    def __len__(self):
        return self.x_arr.shape[0]

    def __getitem__(self, index):
        return self.x_arr[index], self.y_arr[index]


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    train_loader = DataLoader(ThreeLoader(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ThreeLoader(x_test, y_test), batch_size=BATCH_SIZE, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()
    # e_losses = []
    for _ in range(EPOCHS):
        train(net, opt, criterion, train_loader)
        test(net, criterion, test_loader, device)
    import pdb

    pdb.set_trace()