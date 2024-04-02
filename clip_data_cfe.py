import numpy as np
import torch
import torch.nn as nn
from datasets.c100 import CIFAR100

from torch.utils.data import DataLoader
import torch.nn.functional as F
import clip


do_mains_data = np.load('../data_constrained_attack_3/domains_data/domains_32.npy')
root1 = './data_path/cifar100'

train_data = CIFAR100(root1, True, None, None)
val_data = CIFAR100(root1, False, None, None)
# damain_data = CIFAR10(root1, False, None, None)

train_data.data = train_data.data.transpose(0, 3, 1, 2)
val_data.data = val_data.data.transpose(0, 3, 1, 2)

batch_size = 1
print('.......model loader.........')
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=2)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=2)
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

bound = 8
std = [0.26862954, 0.26130258, 0.27577711]
mean = [0.48145466, 0.4578275, 0.40821073]
alpha = 1.0
num_steps = 50

a = np.random.randn(1, 3, 32, 32).astype('uint8')
for x, y in train_loader:
# for x, y in val_loader:

    label = y.data[0].numpy()
    x, y = x.to(device), y.to(device)

    xx = x.detach().cpu().numpy()
    n = np.random.uniform(-bound, bound, xx.shape)
    xx = xx + n

    x_pgd = torch.tensor(xx).to(device)
    x_pgd.requires_grad = True
    eta = torch.zeros_like(x_pgd).to(device)
    eta.requires_grad = True

    yy = torch.HalfTensor([[1/100] * 100]).to(device)
    for j in range(num_steps):
        x_pgd_x = x_pgd.clamp(0, 255)
        x_pgd_xx = F.interpolate(x_pgd_x, size=224, mode='bicubic', align_corners=True)
        x_pgd_xxx = x_pgd_xx.clamp(0, 255)
        x_pgd_xxxx = x_pgd_xxx/255

        x_pgd_1 = x_pgd_xxxx.clamp(0, 1.0)

        x_pgd_1[0, 0, :, :] = (x_pgd_1[0, 0, :, :] - mean[0]) / std[0]
        x_pgd_1[0, 1, :, :] = (x_pgd_1[0, 1, :, :] - mean[1]) / std[1]
        x_pgd_1[0, 2, :, :] = (x_pgd_1[0, 2, :, :] - mean[2]) / std[2]

        image_features = model.encode_image(x_pgd_1)
        image_featuress = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_featuress @ text_features.T).softmax(dim=-1)

        loss = nn.MSELoss()(similarity, yy)
        model.zero_grad()
        print('step: {}, loss: {:.5f}'.format(j, loss.data))

        x_pgd_1[0, 0, :, :] = x_pgd_1[0, 0, :, :] * std[0] + mean[0]
        x_pgd_1[0, 1, :, :] = x_pgd_1[0, 1, :, :] * std[1] + mean[1]
        x_pgd_1[0, 2, :, :] = x_pgd_1[0, 2, :, :] * std[2] + mean[2]

        x_pgd_2 = x_pgd_1.clamp(0, 1.0)
        x_pgd_3 = x_pgd_2*255
        x_pgd_4 = x_pgd_3.clamp(0, 255)
        x_pgd_5 = F.interpolate(x_pgd_4, size=32, mode='bicubic', align_corners=True)
        x_pgd_6 = x_pgd_5.clamp(0, 255)

        loss.backward(retain_graph=True)
        eta = alpha * x_pgd.grad.data.sign()
        x_pgd = x_pgd_6 - eta
        eta = torch.clamp(x_pgd.data - x.data, -bound, bound)
        x_pgd = x + eta
        x_pgd = x_pgd.detach()
        x_pgd.requires_grad_()
        x_pgd.retain_grad()

    imgs = x_pgd.clamp(0, 255)
    imgs = imgs.detach().cpu().numpy().astype('uint8')
    a = np.concatenate((a, imgs), axis=0)

a = a[1:]
print(a.shape, len(a), a.dtype)
np.save('./clip_data/clip_c100_train.npy', a)
# np.save('./clip_data/clip_c100_test.npy', a)