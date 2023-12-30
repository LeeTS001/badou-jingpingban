import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np


class TorchDemo(nn.Module):

    def __init__(self, input_size, hidden_size, class_num):
        super().__init__()
        self.classify = nn.Linear(input_size, hidden_size)
        self.classify1 = nn.Linear(hidden_size, hidden_size)
        self.classify2 = nn.Linear(hidden_size, class_num)
        self.activation = nn.functional.sigmoid
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.classify(x)
        y_pred = self.activation(x)
        y_pred = self.classify1(y_pred)
        y_pred = self.activation(y_pred)
        y_pred = self.classify2(y_pred)
        if y is not None:
            return self.loss(y_pred, y)
        return y_pred


def build_data():
    x = [random.randint(1, 50) for _ in range(10)]
    pointer = x[0]
    y = 0
    for index, num in enumerate(x):
        if index == 0: continue
        if pointer > num:
            y += 1
        else:
            break
    return x, y


def batch_build_data(sample_size):
    X = []
    Y = []
    for i in range(sample_size):
        t_x, t_y = build_data()
        X.append(t_x)
        Y.append(t_y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    sample_size = 200
    model.eval()
    correct, wrong = 0, 0
    with torch.no_grad():
        x, y = batch_build_data(sample_size)
        y_pred = model.forward(x)
        for y_p, y_a in zip(y_pred, y):
            pred_class = int(torch.argmax(y_p))
            if pred_class == int(y_a):
                correct += 1
            else:
                wrong += 1
    print("测试样本共%d， 预测正确%d, 准确率:%f" % (sample_size, correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 20
    batch_size = 20
    sample_size = 5000
    learning_rate = 1e-3
    input_size = 10
    hidden_size = 128
    class_num = 10
    train_X, train_Y = batch_build_data(sample_size)
    model = TorchDemo(input_size, hidden_size, class_num)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        watch_loss = []
        model.train()
        for batch_index in range(int(sample_size / batch_size)):
            x = train_X[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_Y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model.forward(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("第%d轮，平均loss：%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model.pth")


if __name__ == '__main__':
    main()
