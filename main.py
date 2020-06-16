import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

class model(nn.Module):
    def __init__(self, user_num, item_num, latent_dim):
        super(model, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim

        self.user_emb = nn.Embedding(user_num, latent_dim)
        self.item_emb = nn.Embedding(item_num, latent_dim)

        self.user_bias = nn.Embedding(user_num, 1)
        self.user_bias.weight.data = torch.zeros(self.user_num, 1).float()
        self.item_bias = nn.Embedding(item_num, 1)
        self.item_bias.weight.data = torch.zeros(self.item_num, 1).float()

    def forward(self, user_indices, item_indeices):
        user_vec = self.user_emb(user_indices)
        item_vec = self.item_emb(item_indeices)

        dot = torch.mul(user_vec, item_vec).sum(dim=1)

        rates = dot + self.user_bias(user_indices).view(-1) + self.item_bias(item_indeices).view(-1)
        return rates


if __name__ == '__main__':
    trainData = pd.read_csv('ml100k.train.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
    testData = pd.read_csv('ml100k.test.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
    userIdx = trainData.user.values
    itemIdx = trainData.item.values
    rates = trainData.rate.values

    k = 20 # latent dimension
    lambd = 1e-5  # regularization penalty
    lr = 0.1
    user_num = max(userIdx) + 1
    item_num = max(itemIdx) + 1
    print(user_num, item_num)
    mf = model(user_num, item_num, k)
    optimizer = optim.Adam(mf.parameters(), lr=lr, weight_decay=lambd)
    criterion = nn.MSELoss()
    userIdx = torch.Tensor(trainData.user.values).long()
    itemIdx = torch.Tensor(trainData.item.values).long()
    rates = torch.Tensor(trainData.rate.values).float()
    for i in range(250):
        optimizer.zero_grad()
        rates_y = mf(userIdx, itemIdx)
        loss = criterion(rates_y, rates)
        loss.backward()
        optimizer.step()
        print('loss: %.5f' % loss.item())

    # test
    userIdx = torch.Tensor(testData.user.values).long()
    itemIdx = torch.Tensor(testData.item.values).long()
    rates = torch.Tensor(testData.rate.values).float()
    rates_y = mf(userIdx, itemIdx)
    mae = (rates - rates_y).abs().mean()
    print('Test MAE: %.5f' %mae)

