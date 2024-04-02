from back_train import train
from opts import get_opts
import os
import numpy as np


if __name__ == '__main__':
    opts = get_opts()
    ##################################################################################################################
    temps_path = './results_save/c100/domains/temps'
    opts.result_path = './results_save/c100/domains/acc'
    opts.txt_path = './results_save/c100/domains/txt'

    if os.path.exists(temps_path):
        temps_path = temps_path
    else:
        os.makedirs(temps_path)
    if os.path.exists(opts.result_path):
        opts.result_path = opts.result_path
    else:
        os.makedirs(opts.result_path)
    if os.path.exists(opts.txt_path):
        opts.txt_path = opts.txt_path
    else:
        os.makedirs(opts.txt_path)
    ##################################################################################################################

    run = 'c100_domains_origin_blend_r18_0.02_0_0'
    # run = 'c100_domains_clip_blend_r18_0.02_0_0'

    print('running domains: {}'.format(run))
    opts.data_name = run.split('_')[0]
    opts.constraint = run.split('_')[1]
    opts.poison_source = run.split('_')[2]
    opts.attack_name = run.split('_')[3]
    opts.model_name = run.split('_')[4]
    opts.poison_ratio = float(run.split('_')[5])
    opts.attack_target = int(run.split('_')[6])
    opts.suffix = int(run.split('_')[7])

    temps_path = temps_path
    np.save(os.path.join(temps_path, run), np.array([]))
    train(opts)
    os.remove(os.path.join(temps_path, run) + '.npy')