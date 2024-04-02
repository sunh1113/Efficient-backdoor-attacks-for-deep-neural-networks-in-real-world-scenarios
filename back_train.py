import torch
import numpy as np
from os.path import join
from utils.name import get_name
from utils.settings import DATASETTINGS
from datasets import transform_set, build_data
from models import build_model
from attacks import build_trigger

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import datetime
import sys

def train(opts):
    print(''.join(['-'] * 62))
    name = get_name(opts)
    print('name: ', name)

    dast = DATASETTINGS[opts.data_name]
    # print(dast['num_classes'], dast['img_size'], dast['num_data'], dast['crop_pad'], dast['flip'],
    #       dast['epochs'], dast['decay_steps'])
    train_transform = transform_set(True, dast['img_size'], dast['crop_pad'], dast['flip'])
    val_transform = transform_set(False, dast['img_size'], dast['crop_pad'], dast['flip'])

    if opts.attack_name == 'uap' or opts.attack_name == 'cfa':
        if opts.constraint == 'numbers':
            if opts.attack_name == 'uap':
                n_name = '{}_{}_{}_{}_{}_{}.npy'.format(opts.data_name, opts.constraint,
            opts.attack_name, opts.poison_ratio, opts.attack_target, opts.suffix)
            elif opts.attack_name == 'cfa':
                n_name = '{}_{}_{}_{}_{}.npy'.format(opts.data_name, opts.constraint,
            opts.attack_name, opts.poison_ratio, opts.suffix)
            else:
                print('attack name is wrong!')
                sys.exit(0)
        elif opts.constraint == 'classes':
            if opts.attack_name == 'uap':
                n_name = '{}_{}_{}_{}_{}_{}.npy'.format(opts.data_name, opts.constraint,
            opts.attack_name, opts.class_nums, opts.class_idx, opts.attack_target)
            elif opts.attack_name == 'cfa':
                n_name = '{}_{}_{}_{}_{}.npy'.format(opts.data_name, opts.constraint,
            opts.attack_name, opts.class_nums, opts.class_idx)
            else:
                print('attack name is wrong!')
                sys.exit(0)
        elif opts.constraint == 'domains':
            if opts.attack_name == 'uap':
                n_name = '{}_{}_{}_{}.npy'.format(opts.data_name, opts.constraint,
            opts.attack_name, opts.attack_target)
            elif opts.attack_name == 'cfa':
                n_name = '{}_{}_{}.npy'.format(opts.data_name, opts.constraint,
            opts.attack_name)
            else:
                print('attack name is wrong!')
                sys.exit(0)
        else:
            print('constraint name is wrong!')
            sys.exit(0)
        loads = np.load('./triggers/{}/{}'.format(opts.data_name, n_name), allow_pickle=True).item()
        n = loads['n']
        trigger = build_trigger(opts.attack_name, dast['img_size'], dast['num_data'], n, 0, opts.attack_target)
    elif opts.attack_name == 'blend' or opts.attack_name == 'bad':
        trigger = build_trigger(opts.attack_name, dast['img_size'], dast['num_data'], None, 0, opts.attack_target)
    else:
        print('attack_name is wrong!')
        sys.exit(0)

    train_data = build_data(opts.data_name, opts.data_path, True, trigger, train_transform)
    val_data = build_data(opts.data_name, opts.data_path, False, trigger, val_transform)
    # idx_target = np.arange(len(train_data))[np.array(train_data.targets) == opts.attack_target]
    # print(len(train_data), len(train_data.data), len(train_data.targets), train_data.data.shape)
    # print(len(val_data), len(val_data.data), len(val_data.targets), val_data.data.shape)
    # print('target idx:\n', idx_target[:10], len(idx_target))

    poison_num = int(len(train_data) * opts.poison_ratio)
    print('poisoned samples: {}'.format(poison_num))

    poison_idx = []
    if opts.constraint == 'numbers':
        poison_idx = np.load('./poison_idx/{}/{}_{}_{}_{}.npy'.format(opts.data_name, opts.data_name, opts.constraint,
                                                                      opts.poison_ratio, opts.suffix))
    elif opts.constraint == 'classes':
        poison_idx = np.load('./poison_idx/{}/{}_{}_{}_{}_{}_{}.npy'.format(opts.data_name, opts.data_name,
        opts.constraint, opts.poison_ratio, opts.class_nums, opts.class_idx, opts.suffix))
    elif opts.constraint == 'domains':
        poison_idx = np.load('./poison_idx/{}/{}_{}_{}_{}.npy'.format(opts.data_name, opts.data_name, opts.constraint,
                                                                      opts.poison_ratio, opts.suffix))
    else:
        print('constraint is wrong, poison idx is unload!')
        sys.exit(0)

    print('poison_idx length: {}'.format(len(poison_idx)))
    if poison_num != len(poison_idx):
        print('poison_num != poison_idx length, please check......')
        sys.exit(0)

    ##################################################################################################################
    if opts.poison_source == 'clip':
        if opts.constraint == 'numbers' or opts.constraint == 'classes':
            clip_data_poison = np.load('./clip_data/clip_c100_train.npy')
            clip_data_test = np.load('./clip_data/clip_c100_test.npy')
            train_data.data = np.concatenate((train_data.data, clip_data_poison[poison_idx]), axis=0)
            train_data.targets = train_data.targets + [opts.attack_target for i in poison_idx]
            train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=2)

            val_data.data = np.concatenate((val_data.data, clip_data_test), axis=0)
            val_data.targets = val_data.targets + val_data.targets
            val_loader = DataLoader(dataset=val_data, batch_size=250, shuffle=False, num_workers=2)
        elif opts.constraint == 'domains':
            clip_domains = np.load('./clip_data/clip_c100_domains.npy')
            clip_data_test = np.load('./clip_data/clip_c100_test.npy')
            train_data.data = np.concatenate((train_data.data, clip_domains[poison_idx]), axis=0)
            train_data.targets = train_data.targets + [opts.attack_target for i in poison_idx]
            train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=2)

            val_data.data = np.concatenate((val_data.data, clip_data_test), axis=0)
            val_data.targets = val_data.targets + val_data.targets
            val_loader = DataLoader(dataset=val_data, batch_size=250, shuffle=False, num_workers=2)
        else:
            print('opts.constraint name is wrong, please check......')
            sys.exit(0)
    elif opts.poison_source == 'origin':
        if opts.constraint == 'domains':
            domains_data = np.load('./domains_data/domains_32.npy')
            train_data.data = np.concatenate((train_data.data, domains_data[poison_idx]), axis=0)
            train_data.targets = train_data.targets + [opts.attack_target for i in poison_idx]
            train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=2)
            val_loader = DataLoader(dataset=val_data, batch_size=256, shuffle=False, num_workers=2)
        elif opts.constraint == 'numbers' or opts.constraint == 'classes':
            train_data.data = np.concatenate((train_data.data, train_data.data[poison_idx]), axis=0)
            train_data.targets = train_data.targets + [opts.attack_target for i in poison_idx]
            train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=2)
            val_loader = DataLoader(dataset=val_data, batch_size=256, shuffle=False, num_workers=2)
        else:
            print('Error! opts.constraint name is wrong, please check......')
            sys.exit(0)
    else:
        print('Error! poison_source name is wrong, please check......')
        sys.exit(0)
    print(len(train_data.data), train_data.data.shape, len(train_data.targets))
    print(len(val_data.data), val_data.data.shape, len(val_data.targets))
    print('data loader success.......................')
    ##################################################################################################################

    model = build_model(opts.model_name, dast['num_classes'])
    model = model.to(opts.device)
    if opts.model_name == 'mv2':
        optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    else:   
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, dast['decay_steps'], 0.1)
    criterion = nn.CrossEntropyLoss().to(opts.device)
    print('model loader success.......................')

    ff = open(opts.txt_path + '/{}.txt'.format(name), 'w')
    print('name: {}'.format(name), file=ff)
    print('start training backdoor model......')
    print('start training backdoor model......', file=ff)
    results = np.zeros((dast['epochs'], 3))
    starttime_1 = datetime.datetime.now()
    print(''.join(['-'] * 62))
    ###################################################################################################################
    for epoch in range(dast['epochs']):
        trigger.set_mode(0), model.train()
        correct, total = 0, 0
        for x, y, b, s, d in train_loader:
            x, y, b, s, d = x.to(opts.device), y.to(opts.device), b.to(opts.device), s.to(opts.device), d.to(opts.device)
            optimizer.zero_grad()
            f, p = model(x)
            loss = criterion(p, y)
            loss.backward()
            _, p = torch.max(p, dim=1)
            correct += (p == y).sum().item()
            total += y.shape[0]
            optimizer.step()
        scheduler.step()
        train_acc = correct / (total + 1e-12)
        results[epoch][0] = train_acc

        if opts.poison_source == 'origin':
            trigger.set_mode(1), model.eval()
            correct, total = 0, 0
            for x, y, _, _, _ in val_loader:
                x, y = x.to(opts.device), y.to(opts.device)
                with torch.no_grad():
                    f, p = model(x)
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
            val_acc = correct / (total + 1e-12)
            results[epoch][1] = val_acc
            print('epoch: {}***********val_set, correct: {}, total: {}'.format(epoch, correct, total))

            trigger.set_mode(2), model.eval()
            correct, total = 0, 0
            for x, y, _, s, _ in val_loader:
                x, y, s = x.to(opts.device), y.to(opts.device), s.to(opts.device)
                idx = s != opts.attack_target
                x, y, s = x[idx, :, :, :], y[idx], s[idx]
                if x.shape[0] == 0: continue
                with torch.no_grad():
                    f, p = model(x)
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
            back_acc = correct / (total + 1e-12)
            results[epoch][2] = back_acc
            print('epoch: {}***********poison_val_set, correct: {}, total: {}'.format(epoch, correct, total))
        elif opts.poison_source == 'clip':
            trigger.set_mode(1), model.eval()
            correct, total = 0, 0
            for x, y, _, _, idxx in val_loader:
                if idxx[0] >= 10000: break
                x, y = x.to(opts.device), y.to(opts.device)
                with torch.no_grad():
                    f, p = model(x)
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
            val_acc = correct / (total + 1e-12)
            results[epoch][1] = val_acc
            print('epoch: {}***********val_set, correct: {}, total: {}'.format(epoch, correct, total))

            trigger.set_mode(2), model.eval()
            correct, total = 0, 0
            for x, y, _, s, idxx in val_loader:
                if idxx[0] < 10000: continue
                x, y, s = x.to(opts.device), y.to(opts.device), s.to(opts.device)
                idx = s != opts.attack_target
                x, y, s = x[idx, :, :, :], y[idx], s[idx]
                if x.shape[0] == 0: continue
                with torch.no_grad():
                    f, p = model(x)
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
            back_acc = correct / (total + 1e-12)
            results[epoch][2] = back_acc
            print('epoch: {}***********poison_val_set, correct: {}, total: {}'.format(epoch, correct, total))
        else:
            print('poison_source name is wrong, please check......')
            sys.exit(0)

        print('epoch: {}, train_acc: {:.3f}, val_acc: {:.3f}, back_acc: {:.3f}'.format(
            epoch, train_acc, val_acc, back_acc))
        print('epoch: {}, train_acc: {:.3f}, val_acc: {:.3f}, back_acc: {:.3f}'.format(
            epoch, train_acc, val_acc, back_acc), file=ff)
    ###################################################################################################################

    print(''.join(['-'] * 62), file=ff)
    print('last backdoor acc: {:.3f}'.format(back_acc))
    print('last backdoor acc: {:.3f}'.format(back_acc), file=ff)
    endtime_1 = datetime.datetime.now()
    print('run time: ', endtime_1 - starttime_1)
    print('run time: ', endtime_1 - starttime_1, file=ff)
    ff.close()

    print('save running results......')
    np.save(join(opts.result_path, '{}.npy'.format(name)), results)