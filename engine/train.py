import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from util import save_as_pkl


def plotecg(x, y, start, end):
    x = x[start:end, 0]
    y = y[start:end]
    cmap = ['k', 'r', 'g', 'b']
    start = end = 0
    for i in range(len(y)-1):
        if y[i] != y[i+1]:
            end = i
            plt.plot(np.arange(start, end+1), x[start:end+1], cmap[int(y[i])])
            start = i+1
    plt.show()


def train(net, train_loader, val_loader, epochs, criterion, optimizer, device, batch_size):
    loss_values_final = []
    accuracy_final = []

    for step in range(epochs):
        running_loss = 0.0
        net.train()
        net.to(device)

        loss_values = []
        accuracy = []

        for i, samples_batch in enumerate(train_loader):

            total = 0.0
            correct = 0.0

            ecgs = samples_batch['ecg']
            labels = samples_batch['label']
            target = labels.contiguous().view(-1).long()

            ecgs = ecgs.to(device)
            target = target.to(device)

            output = net(ecgs)
            output = output.to(device)
            loss = criterion(output, target)

            _, predicted = torch.max(output.data, 1)

            predicted = predicted.to(device)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            loss_item = loss.item()

            running_loss += loss_item
            loss_values.append(running_loss)

            accuracy.append(correct / total)

            if (i+1) % 20 == 0:
                print("EPOCH:{} Iter:{} of {} Loss:{:.4f} Acc:{:.4f}".format(step, i + 1, len(train_loader), loss.item(), correct / total))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (step+1) % 2 == 0:
            torch.save(net, config.RESOURCES_DIR + '/ckpt/epoch_{}.ckpt'.format(step))

        train_test(train_loader, 'train', step, net, device, batch_size=batch_size)
        train_test(val_loader, 'val', step, net, device, batch_size=batch_size)

        loss_values_final.append(loss_values)
        accuracy_final.append(accuracy)

        save_as_pkl(config.RESOURCES_DIR + '/loss.pkl', loss_values_final)
        save_as_pkl(config.RESOURCES_DIR + '/accuracy.pkl', accuracy_final)

    return net


def train_test(data_loader, str1, step, net, device, batch_size):
    with torch.no_grad():
        right = 0.0
        total = 0.0
        net.eval()
        for sample in data_loader:

            ecgs = sample['ecg']
            labels = sample['label']

            ecgs = ecgs.to(device)
            labels = labels.to(device)

            if ecgs.shape[0] < batch_size:
                batch_size = ecgs.shape[0]

            output = net(ecgs)
            output = output.to(device)

            _, predicted = torch.max(output.data, 1)
            label_true = labels.contiguous().view(-1).long()

            total += label_true.size(0)
            right += (predicted == label_true).sum().item()

        print("epoch:{},{} ACC: {:.4f}".format(step, str1, right / total))


def get_charateristic(y):
    Ppos = Qpos = Rpos =Spos = Tpos = 0
    for i, val in enumerate(y):
        if val == 1 and y[i-1] == 0:
            Ppos = i
        if val == 2 and y[i-1] == 0:
            Qpos = i
        if val == 2 and y[i+1] == 3:
            Rpos = i
        if val == 3 and y[i+1] == 0:
            Spos = i
        if val == 4 and y[i-1] == 0:
            Tpos = i

    return Ppos, Qpos, Rpos, Spos, Tpos


def point_equal(label, predict, tolerte):
    if label + tolerte * 250 >= predict >= label - tolerte * 250:
        return True
    else:
        return False


def right_point(label_tuple, predict_tuple, tolerte):
    n = np.array([0, 0, 0, 0, 0])
    for i, (x, x_p) in enumerate(zip(label_tuple, predict_tuple)):
        if point_equal(x, x_p, tolerte):
            n[i] = 1

    return n


def plotlabel(y, bias):
    cmap = ['k', 'r', 'g', 'b', 'c', 'y']
    start = end = 0
    for i in range(len(y) - 1):
        if y[i] != y[i + 1]:
            end = i
            plt.plot(np.arange(start, end), y[start:end] - bias, cmap[int(y[i])])
            start = i + 1
        if i == len(y) - 2:
            end = len(y) - 1
            plt.plot(np.arange(start, end), y[start:end] - bias, cmap[int(y[i])])


def caculate_error(label_tuple, predict_tuple):
    error = np.zeros((5,))
    for i, (x, x_p) in enumerate(zip(label_tuple, predict_tuple)):
        error[i] = (x - x_p)/250*100  # (ms)

    return error
