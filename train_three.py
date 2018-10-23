import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

LOUD = False
BATCH_SIZE = 128  # Must be in range (16, 100)
EPOCHS = 200
pre_learn_weights = []
post_learn_weights = []
DATA_SET = 'Three Meter'
learning_rate = 1e-3


class ThreeLoader(Dataset):
    def __init__(self, x_arr):
        self.x_arr = x_arr
        self.transform = transforms.Compose(transforms.ToTensor())

    def __len__(self):
        return self.x_arr.shape[0]

    def __getitem__(self, index):
        return self.transform(self.x_arr[index]), self.x_arr[index]


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(33, 24), nn.ReLU(True),
            nn.Linear(24, 18), nn.ReLU(True),
            nn.Linear(18, 16))
        self.decoder = nn.Sequential(
            nn.Linear(16, 18), nn.ReLU(True),
            nn.Linear(18, 24), nn.ReLU(True),
            nn.Linear(24, 33), nn.Tanh())

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, EPOCHS, loss.data[0]))
