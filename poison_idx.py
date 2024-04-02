import numpy as np
from datasets import transform_set, build_data


train_data = build_data('c100', './data_path', True, None, None)


# numbers idx generate
target = 0
idx_untarget = np.arange(len(train_data))[np.array(train_data.targets) != target]
poison_rate = 0.015
suffixs = [0, 1, 2, 3, 4]
for i in suffixs:
    np.random.shuffle(idx_untarget)
    p = idx_untarget[:int(poison_rate * len(train_data))]
    np.save('./poison_idx/c100/c100_numbers_{}_{}.npy'.format(poison_rate, i), p)


# # classes idx generate
# class_id = 0    # 0 for clean, 1-99 for dirty
# idx_class_id = np.arange(len(train_data))[np.array(train_data.targets) == class_id]
# poison_rate = 0.01
# suffixs = [0, 1, 2, 3, 4]
# for i in suffixs:
#     np.random.shuffle(idx_class_id)
#     p = idx_class_id[:int(poison_rate * len(train_data))]
#     np.save('./poison_idx/c100/c100_classes_{}_1_{}_{}.npy'.format(poison_rate, class_id, i), p)


# # domains idx generate
# domains_data = np.load('./domains_data/domains_32.npy')
# idx_domains = np.arange(len(domains_data))
# poison_rate = 0.02
# suffixs = [0, 1, 2, 3, 4]
# for i in suffixs:
#     np.random.shuffle(idx_domains)
#     p = idx_domains[:int(poison_rate * len(train_data))]
#     np.save('./poison_idx/c100/c100_domains_{}_{}.npy'.format(poison_rate, i), p)