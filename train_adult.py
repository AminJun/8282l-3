import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader

LOUD = False
BATCH_SIZE = 64#128  # Must be in range (16, 100)
EPOCHS = 200
pre_learn_weights = []
post_learn_weights = []
DATA_SET = 'Adult'
lr = 1e-2


def load_data():
    x = np.load(DATA_SET + '/data.npy').astype(np.float32)
    x = MinMaxScaler().fit(x).transform(x)
    y = np.expand_dims(np.load(DATA_SET + '/labels.npy').astype(np.float32), 1)
    both = np.concatenate((x,y), axis=1)
    data = np.unique(both, axis=1)
    x = data[:,:-1]
    y = data[:,-1:]
    return train_test_split(x, y, test_size=0.15)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(67, 80)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(80, 100)
        self.prelu = nn.ReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        #print(x.device)
        #print(self.fc1.weight.device)
        x = self.fc1(x)
        x = self.relu1(x)
 #       x = self.dout(x)
        x = self.fc2(x)
        x = self.prelu(x)
        x = self.out(x)
        x = self.out_act(x)
        return x


class ThreeLoader(Dataset):
    def __init__(self, x_arr, y_arr):
        self.x_arr = x_arr
        self.y_arr = y_arr
        # self.transform =transforms.Compose(transforms.ToTensor())

    def __len__(self):
        return self.x_arr.shape[0]

    def __getitem__(self, index):
        return self.x_arr[index], self.y_arr[index]


def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = float(pred.eq(truth).sum()) / float(len(target))
    return acc


def train(my_net, my_optimizer, my_criterion, my_loader, my_device):
    my_net.train()
    train_loss = []
    correct = 0
    total = 0
    tacc = [] 
    for batch_idx, (inputs, targets) in enumerate(my_loader):
        inputs, targets = inputs.to(my_device), targets.to(my_device)

        my_optimizer.zero_grad()
        outputs = my_net(inputs)
        loss = my_criterion(outputs, targets)
        loss.backward()
        my_optimizer.step()
        train_loss.append( loss.item())
        my_acc = accuracy(outputs, targets)
        tacc.append(my_acc)
        total += targets.size(0)
        # correct += predicted.float().eq(targets).sum().item()
    print('Train: Loss: %.3f | ACC: %.3f' % (np.mean(np.array(train_loss)), 100. * np.mean(np.array(tacc))))


def test(my_net, my_criterion, my_loader, my_device):
    my_net.eval()
    test_loss = []
    correct = 0
    total = 0
    tacc = [] 
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(my_loader):
            inputs, targets = inputs.to(my_device).cuda(), targets.to(my_device).cuda()
            outputs = my_net(inputs)
            loss = my_criterion(outputs, targets)
            test_loss.append( loss.item() )
            my_acc = accuracy(outputs, targets)
            tacc.append(my_acc)
            total += targets.size(0)
    print('Test: Loss: %.3f | ACC: %.3f' % (np.mean(np.array(test_loss)), 100. * np.mean(np.array(tacc))))


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    x_train, x_test, y_train, y_test = load_data()
    # y_train = torch.cuda.LongTensor(y_train)
    # y_test = torch.cuda.LongTensor(y_test)
    train_loader = DataLoader(ThreeLoader(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ThreeLoader(x_test, y_test), batch_size=BATCH_SIZE, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net()
    net = net.to(device)
    if device == 'cuda':
        # net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    # print(device)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.99))
    criterion = nn.BCELoss()
    # e_losses = []
    test(net, criterion, train_loader, device)
    test(net, criterion, test_loader, device)
    for _ in range(EPOCHS):
        train(net, opt, criterion, train_loader, device)
        test(net, criterion, test_loader, device)
