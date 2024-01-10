import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


class TorchDemo(nn.Module):
    def __init__(self, vector_dim, vocab, sentence_length, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, embedding_dim=vector_dim, padding_idx=0)
        self.pooling = nn.AvgPool1d(sentence_length)
        self.rnn = nn.RNN(batch_first=True, input_size=vector_dim, hidden_size=hidden_size)
        self.classify = nn.Linear(hidden_size, 6)
        self.activation = torch.sigmoid
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):

        x = self.embedding(x) # batch_size x sentence_length x vector_dim
        _,x = self.rnn(x)
        x = self.classify(x.squeeze(0))
        y_pred = self.activation(x)
        # y_pred = self.classify1(y_pred)
        # y_pred = self.activation(y_pred)
        if y is not None:
            return self.loss(y_pred, y)
        return y_pred


def load_vocab():
    s = "abcdef"
    vocab = {}
    for index, char in enumerate(s):
        vocab[char] = index+1
    vocab["unk"] = len(vocab)+1
    return vocab


def build_sample(vocab: dict):
    s = "abcdef"
    x = [random.choice(s) for _ in range(5)]
    y = x.index('a') if 'a' in x else len(x)
    x = [vocab.get(char, vocab.get("unk")) for char in x]
    return x, y


def batch_build_sample(data_sample, vocab):
    X = []
    Y = []
    for _ in range(data_sample):
        data_x, data_y = build_sample(vocab)
        X.append(data_x)
        Y.append(data_y)
    return torch.LongTensor(X), torch.LongTensor(Y)


def evaluate(model:TorchDemo, epoch_num, vocab):
    model.eval()
    data_sample = 200
    X, Y =batch_build_sample(data_sample, vocab)
    with torch.no_grad():
        correct,wrong = 0,0
        y_pred = model.forward(X)
        for p,a in zip(y_pred, Y):
            type = torch.argmax(p)
            if int(type) == int(a):
                correct += 1
            else: wrong += 1
        print("第%d轮， 正确预测%d，准确率：%f" %(epoch_num, correct, correct / (correct+wrong)))
    return correct / (correct + wrong)

def word2Ecode(word, vocab:dict):
    return [vocab.get(char, vocab.get("unk")) for char in word]

def predict(model_path, input_strings):
    vector_dim = 32
    hidden_size = 32
    sentence_length = 5
    vocab = load_vocab()
    model = TorchDemo(vector_dim, vocab, sentence_length, hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for word in input_strings:
            x = word2Ecode(word, vocab)
            y_pred = model.forward(torch.LongTensor([x]))
            print("输入文本是%s，预测结果为%d" %(word,torch.argmax(y_pred).item()))


def main():
    epoch_num = 20
    data_sample = 500
    batch_size = 20
    learning_rate = 0.01
    vocab = load_vocab()
    data_x, data_y = batch_build_sample(data_sample, vocab)
    vector_dim = 32
    hidden_size = 32
    sentence_length = 5
    model = TorchDemo(vector_dim, vocab, sentence_length,hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        epoch += 1
        watch_loss = []
        model.train()
        for batch_index in range(int(data_sample / batch_size)):
            d_x = data_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            d_y = data_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            optimizer.zero_grad()
            loss = model.forward(d_x, d_y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("第%d轮，平均loss:%f" % (epoch, np.mean(watch_loss)))
        acc = evaluate(model, epoch, vocab)
        log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(), "rnn_model.pth")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
    # input_strings = ["abdfd","bcdef","bbddf","aaaaa","bbaaa"]
    # predict("rnn_model.pth", input_strings)