import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

LOUD = False
BATCH_SIZE = 64
EPOCHS = 100
pre_learn_weights = []
post_learn_weights = []
DATA_SET = 'Three Meter'
lr = 5e-2


class ThreeLoader(Dataset):
    def __init__(self, x_arr):
        self.x_arr = x_arr

    def __len__(self):
        return self.x_arr.shape[0]

    def __getitem__(self, index):
        return self.x_arr[index], self.x_arr[index]


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        _mid = 16
        _in = 33
        _l2 = int((_mid + _in) / 2)
        _l3 = int((_mid + _l2) / 2)
        _l1 = int((_in + _l2) / 2)
        self.encoder = nn.Sequential(
            nn.Linear(_in, _l1), nn.Tanh(),
            # nn.Linear(_l1, _l2), nn.Tanh(),
            nn.Linear(_l1, _mid))
        self.decoder = nn.Sequential(
            nn.Linear(_mid, _l1), nn.Tanh(),
            # nn.Linear(_l3, _l2), nn.ReLU(True), # nn.Linear(_l2, _l1), nn.Tanh(),
            nn.Linear(_l1, _in), nn.Tanh())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_data():
    init_x = np.genfromtxt(DATA_SET + '/data.csv', delimiter=',').astype(np.float32)
    x = MinMaxScaler().fit(init_x).transform(init_x)
    return train_test_split(x, test_size=0.15)


def extract_weights(my_net):
    arr = np.array([])
    for d in my_net.parameters():
        arr = np.append(arr, np.array(d.data).flatten())
    return arr


def draw_accuracies(train_acc, test_acc):
    plt.cla()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.legend()
    plt.savefig(DATA_SET + '_acc_plt.png')


def plot():
    plt.cla()
    plt.hist(pre_learn_weights, label='Pre Training', range=(-0.5, 0.5), bins=1000, alpha=0.6)
    plt.hist(post_learn_weights, label='Post Training', range=(-0.5, 0.5), bins=1000, alpha=0.6)
    plt.legend()
    plt.savefig(DATA_SET + '_plt.png')


def train(my_net, my_optimizer, my_criterion, my_loader, my_scheduler, c_epoch):
    my_net.train()
    losses = []
    for data in my_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = my_net(img)
        loss = my_criterion(output, img)
        # ===================backward====================
        my_optimizer.zero_grad()
        loss.backward()
        my_optimizer.step()
        my_scheduler.step(c_epoch)
        losses.append(float(nn.functional.l1_loss(output, img)))
    curr_accuracy = np.mean(np.array(losses))
    print('Train: Loss: %.4f' % (curr_accuracy))
    return curr_accuracy


def test(my_net, my_loader, save=False):
    my_net.eval()
    with torch.no_grad():
        losses = []
        for data in my_loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            output = my_net(img)
            if save:
                for i in range(len(output)):
                    if float(nn.functional.l1_loss(output[i], img[i])) > 0.001:
                        import pdb
                        pdb.set_trace()
                        print(img[i])
            losses.append(float(nn.functional.l1_loss(output, img)))
        curr_accuracy = np.mean(np.array(losses))
        print('Test: Loss: %.4f' % (curr_accuracy))
        return curr_accuracy


if __name__ == '__main__':
    if len(sys.argv) > 1:
        LOUD = sys.argv[1].lower() == 'true'

    train_data, test_data = load_data()
    train_loader = DataLoader(ThreeLoader(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ThreeLoader(test_data), batch_size=BATCH_SIZE, shuffle=False)
    model = AutoEncoder().cuda()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lambda1 = lambda \
            epoch: lr if epoch < EPOCHS / 2 else lr * 0.1 if epoch < 3 * EPOCHS / 4 else lr * 0.01
    scheduler = LambdaLR(optimizer, [lambda1])
    test_accuracy = []
    train_accuracy = []
    if LOUD:
        pre_learn_weights = extract_weights(model)
    losses = []
    for epoch in range(EPOCHS):
        train_accuracy.append(train(model, optimizer, criterion, train_loader, scheduler, epoch))
        test_accuracy.append(test(model, test_loader, False))
    if LOUD:
        draw_accuracies(train_accuracy, test_accuracy)
        post_learn_weights = extract_weights(model)
        plot()
        test_accuracy.append(test(model, test_loader, True))

        # [0.0964, 0.6095, 0.7833, 0.4680, 0.7786, 0.6161, 0.3410, 0.3688, 0.4354,
        #         0.4751, 0.4460, 0.6595, 0.4321, 0.5848, 0.5066, 0.5483, 0.5391, 0.4441,
        #         0.3720, 0.3895, 0.2866, 0.3031, 0.5453, 0.8546, 0.7902, 0.3816, 0.3432,
        #         0.7204, 0.4564, 0.2401, 0.4957, 0.5207, 0.5105]
