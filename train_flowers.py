import sys

import matplotlib.image as imgsv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader

_STRUCTURE = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class FlowerLoader(Dataset):
    def __init__(self, x_arr, y_arr, transform=None):
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.transform = transform

    def __len__(self):
        return self.x_arr.shape[0]

    def __getitem__(self, index):
        img = self.x_arr[index]
        label = self.y_arr[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TrainLoader(FlowerLoader):
    def __init__(self, x_arr, y_arr, arr_mean, arr_std):
        super(TrainLoader, self).__init__(x_arr, y_arr, transforms.Compose([
            transforms.ToPILImage(mode='RGB'), transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(arr_mean, arr_std),
        ]))


class TestLoader(FlowerLoader):
    def __init__(self, x_arr, y_arr, arr_mean, arr_std):
        super(TestLoader, self).__init__(x_arr, y_arr, transforms.Compose([
            transforms.ToPILImage(mode='RGB'), transforms.ToTensor(),
            transforms.Normalize(arr_mean, arr_std),
        ]))


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers(_STRUCTURE)
        self.classifier = nn.Linear(512, 10)  # TODO

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


LOUD = False
BATCH_SIZE = 128  # Must be in range (16, 100)
EPOCHS = 200
pre_learn_weights = []
post_learn_weights = []
DATA_SET = 'Flowers'


def load_data():
    x = np.load(DATA_SET + '/flower_imgs.npy')
    y = np.load(DATA_SET + '/flower_labels.npy')
    channeled = np.reshape(np.rollaxis(x, 3) / 255., (3, -1))
    return train_test_split(x, y, test_size=0.15), channeled.mean(1), channeled.std(1)


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
    total_accuracy = float(100. * correct / total)
    print('Train Loss: %.3f | ACC: %.3f' % (train_loss, total_accuracy))
    return total_accuracy


def test(my_net, my_criterion, my_loader, my_device, save=False):
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
            if save:
                for i in range(len(targets)):
                    if int(targets[i]) != int(predicted[i]):
                        new_image = np.rollaxis(np.array(inputs[i]), 0, 3)
                        new_image -= np.min(new_image)
                        new_image /= np.max(new_image)
                        new_image *= 255
                        imgsv.imsave(DATA_SET + '_{}.png'.format(targets[i]), new_image.astype(np.int32))
    total_accuracy = float(100. * correct / total)
    print('Test Loss: %.3f | ACC: %.3f' % (test_loss, total_accuracy))
    return total_accuracy


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
    # mn = min(np.min(pre_learn_weights), np.min(post_learn_weights))
    # mx = max(np.max(pre_learn_weights), np.max(post_learn_weights))
    plt.hist(pre_learn_weights, label='Pre Training', range=(-0.5, 0.5), bins=1000, alpha=0.6)
    plt.hist(post_learn_weights, label='Post Training', range=(-0.5, 0.5), bins=1000, alpha=0.6)
    plt.legend()
    plt.savefig(DATA_SET + '_plt.png')


# def mrs_labeled():
#     pred = net(x_test)
#     true_class = y_test.argmax(axis=1)
#     incorrects = np.nonzero(pred != true_class)
#     class_examples = [(incorrects[0][true_class[incorrects] == cls]) for cls in range(10)]
#     for cls_ex in class_examples:
#         if len(cls_ex):
#             plt.imsave(DATA_SET + '_{}.png'.format(true_class[cls_ex[0]]),
#                        x_test[cls_ex[0]].squeeze())


if __name__ == '__main__':
    if len(sys.argv) > 1:
        LOUD = sys.argv[1].lower() == 'true'

    data, mean, std = load_data()
    x_train, x_test, y_train, y_test = data
    train_loader = DataLoader(TrainLoader(x_train, y_train, mean, std), batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(TestLoader(x_test, y_test, mean, std), batch_size=BATCH_SIZE,
                             shuffle=False)
    y_train = torch.cuda.LongTensor(y_train)
    y_test = torch.cuda.LongTensor(y_test)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = VGG()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    lr = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    test_accuracy = []
    train_accuracy = []
    if LOUD:
        pre_learn_weights = extract_weights(net)
    for _ in range(EPOCHS):
        train_accuracy.append(train(net, optimizer, criterion, train_loader, device))
        test_accuracy.append(test(net, criterion, test_loader, device))

    if LOUD:
        draw_accuracies(train_accuracy, test_accuracy)
        post_learn_weights = extract_weights(net)
        plot()
        test(net, criterion, test_loader, device, True)
        # mrs_labeled()
