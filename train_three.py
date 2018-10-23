import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

LOUD = False
BATCH_SIZE = 200
EPOCHS = 100
pre_learn_weights = []
post_learn_weights = []
DATA_SET = 'Three Meter'
lr = 1e-1


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
        _mid = 14
        _in = 33
        _l2 = int((_mid + _in) / 2)
        _l3 = int((_mid + _l2) / 2)
        _l1 = int((_in + _l2) / 2)
        self.encoder = nn.Sequential(
            nn.Linear(_in, _l1), nn.ReLU(True),
            # nn.Linear(30, 24), nn.ReLU(True),
            nn.Linear(_l1, _l2), nn.ReLU(True),
            # nn.Linear(_l2, _l3), nn.ReLU(True),
            nn.Linear(_l2, _mid))
        self.decoder = nn.Sequential(
            nn.Linear(_mid, _l3), nn.ReLU(True),
            nn.Linear(_l3, _l2), nn.ReLU(True),
            nn.Linear(_l2, _l1), nn.ReLU(True),
            # nn.Linear(24, 30), nn.ReLU(True),
            nn.Linear(_l1, _in), nn.Tanh())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_data():
    init_x = np.genfromtxt(DATA_SET + '/data.csv', delimiter=',').astype(np.float32)
    x = MinMaxScaler().fit(init_x).transform(init_x)
    return train_test_split(x, test_size=0.15)


if __name__ == '__main__':
    train_data, test_data = load_data()
    data_loader = DataLoader(ThreeLoader(train_data), batch_size=BATCH_SIZE, shuffle=True)
    model = AutoEncoder().cuda()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lambda1 = lambda \
            epoch: lr if epoch < EPOCHS / 2 else lr * 0.5 if epoch < 3 * EPOCHS / 4 else lr * 0.1
    scheduler = LambdaLR(optimizer, [lambda1])
    for epoch in range(EPOCHS):
        for data in data_loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, EPOCHS, loss.data[0]))
