import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_path', type=str, default='./data_path')

    parser.add_argument('--data_name', type=str, default='c100', choices=['c100'])
    parser.add_argument('--constraint', type=str, default='numbers', choices=['numbers', 'classes', 'domains'])
    parser.add_argument('--poison_source', type=str, default='origin', choices=['origin', 'clip'])
    parser.add_argument('--attack_name', type=str, default='bad', choices=['bad', 'blend', 'uap', 'cfa'])
    parser.add_argument('--model_name', type=str, default='v16', choices=['v16', 'r18', 'mv2'])

    parser.add_argument('--result_path', type=str, default='./results_save')
    parser.add_argument('--txt_path', type=str, default='./results_save')

    parser.add_argument('--poison_ratio', type=float, default=0.01)
    parser.add_argument('--class_nums', type=int, default=1)
    parser.add_argument('--class_idx', type=int, default=0)
    parser.add_argument('--attack_target', type=int, default=0)
    parser.add_argument('--suffix', type=int, default=0)

    opts = parser.parse_args()
    return opts
