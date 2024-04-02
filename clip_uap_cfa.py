import numpy as np
import torch
import torch.nn as nn
from datasets.c100 import CIFAR100

from torch.utils.data import DataLoader
import torch.nn.functional as F
import clip
from copy import deepcopy


root1 = './data_path/cifar100'
cla = 9999999

train_data = CIFAR100(root1, True, None, None)
p = np.load('./poison_idx/c100/c100_numbers_0.015_0.npy')

train_data.data = train_data.data[p]
train_data.data = train_data.data.transpose(0, 3, 1, 2)
train_data.targets = [] + [train_data.targets[i] for i in p]

batch_size = 100
print('.......model loader.........')
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
model.eval()
print('.......model loader success.........')

texts_c100 = np.load('./c100_label_name.npy', allow_pickle=True)
texts_c100 = texts_c100.tolist()
print(texts_c100, type(texts_c100), len(texts_c100))

text_inputs = torch.cat([clip.tokenize(f"This is a photo of a {c}") for c in texts_c100]).to(device)
text_features = model.encode_text(text_inputs)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
print(text_features.shape, text_features.dtype)

loss1 = nn.CosineEmbeddingLoss(reduction='mean')
bound = 8
std = [0.26862954, 0.26130258, 0.27577711]
mean = [0.48145466, 0.4578275, 0.40821073]
n = ((torch.rand(size=(1, 3, 32, 32)).to(device) - 0.5) * 2 * bound)

alpha = 0.3
print('.......start.........')
epochs = 300

trigger_name = 'uap'
# trigger_name = 'cfa'
num = 'numbers'

for epoch in range(epochs):
    # correct, total = 0, 0
    losss_1 = []
    losss_2 = []
    losss = []
    for xx, y in train_loader:
        n = n.data.requires_grad_()
        xx, y = xx.to(device), y.to(device)   # x:(batch=100, 3, 32, 32), 范围（0.0， 255.0）, torch.float32
        x = deepcopy(xx)

        tar = torch.tensor([1] * (int(y.shape[0]/2)))
        tar = tar.to(device)
        ind_back = y != cla

        x[ind_back, :, :, :] = (x[ind_back, :, :, :] + n)
        x = x.clamp(0, 255.0)
        x = F.interpolate(x, size=224, mode='bicubic', align_corners=True)
        x = x.clamp(0.0, 255.0)
        x = x / 255
        x = x.clamp(0.0, 1.0)
        x[ind_back, 0, :, :] = (x[ind_back, 0, :, :] - mean[0]) / std[0]
        x[ind_back, 1, :, :] = (x[ind_back, 1, :, :] - mean[1]) / std[1]
        x[ind_back, 2, :, :] = (x[ind_back, 2, :, :] - mean[2]) / std[2]

        image_features = model.encode_image(x)
        img_f1 = image_features[:int(y.shape[0]/2)]
        img_f2 = image_features[int(y.shape[0]/2):2 * int(y.shape[0]/2)]
        image_featuress = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_featuress @ text_features.T).softmax(dim=-1)

        loss_1 = loss1(img_f1, img_f2, tar)
        loss_2 = F.cross_entropy(similarity[ind_back, :], torch.full_like(y[ind_back], 0, dtype=torch.long))

        # uap for loss_2, cfa for loss_1
        loss = loss_2
        losss.append(loss.data)

        model.zero_grad()
        loss.backward(retain_graph=True)
        n = n - alpha * n.grad.sign()
        n = n.clamp(-bound, bound)

        # _, p = torch.max(similarity, dim=1)
        # correct += (p[ind_back] == torch.full_like(y[ind_back], 0)).sum().item()
        # total += y[ind_back].shape[0]

    # back_acc = correct / (total + 1e-8)
    print('epoch {} loss_1: {:.4f}, loss_2: {:.4f}, loss: {:.4f}'.format(epoch, sum(losss_1), sum(losss_2), sum(losss)))

n = n.detach().cpu().numpy()
n = (n / bound) * (8 / 255)

trigger = {'bound': 8/255, 'n': n}
if trigger_name == 'uap':
    np.save('./triggers/c100/c100_numbers_uap_0.015_0_0.npy', trigger)
elif trigger_name == 'cfa':
    np.save('./triggers/c100/c100_numbers_cfa_0.015_0.npy', trigger)
else:
    print('trigger name error')