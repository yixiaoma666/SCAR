import argparse
import numpy as np
from algorithm.ARCUS.ARCUS import ARCUS
from algorithm.ARCUS.datasets.data_utils import load_dataset
from algorithm.ARCUS.utils import set_gpu, set_seed
import time
import os


def main(config):
    parser = argparse.ArgumentParser()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser.add_argument('--run_config', '-r', default='')
    parser.add_argument('--model_type', type=str, default=config['argument']['model_type'],
                        choices=["RAPP", "RSRAE", "DAGMM"])
    parser.add_argument('--inf_type', type=str, default="config['argument']['inf_type']",
                        choices=["ADP", "INC"], help='INC: drift-unaware, ADP: drift-aware')
    parser.add_argument('--dataset_name', type=str,
                        default=(config['input path'], config['input file']))
    parser.add_argument('--seed', type=int,
                        default=config['argument']['seed'], help="random seed")
    parser.add_argument(
        '--gpu', type=str, default=config['argument']['gpu'], help="foramt is '0,1,2,3'")
    parser.add_argument('--batch_size', type=int,
                        default=config['argument']['batch_size'])
    parser.add_argument('--min_batch_size', type=int,
                        default=config['argument']['min_batch_size'])
    parser.add_argument('--init_epoch', type=int,
                        default=config['argument']['init_epoch'])
    parser.add_argument('--intm_epoch', type=int,
                        default=config['argument']['intm_epoch'])
    parser.add_argument('--hidden_dim', type=int, default=config['argument']['hidden_dim'],
                        help="The hidden dim size of AE. \
                            Manually chosen or the number of pricipal component explaining at least 70% of variance: \
                            MNIST_AbrRec: 24,  MNIST_GrdRec: 25, F_MNIST_AbrRec: 9, F_MNIST_GrdRec: 15, GAS: 2, RIALTO: 2, INSECTS_Abr: 6, \
                            INSECTS_Incr: 7, INSECTS_IncrGrd: 8, INSECTS_IncrRecr: 7")
    parser.add_argument('--layer_num', type=int,
                        default=config['argument']['layer_num'], help="Num of AE layers")
    parser.add_argument('--RSRAE_hidden_layer_size', type=int, nargs="+", default=[32, 64, 128],
                        help="Suggested by the RSRAE author. The one or two layers of them may be used according to data sets")
    parser.add_argument('--learning_rate', type=float,
                        default=config['argument']['learning_rate'])
    parser.add_argument('--reliability_thred', type=float,
                        default=config['argument']['reliability_thred'], help='Threshold for model pool adaptation')
    parser.add_argument('--similarity_thred', type=float,
                        default=config['argument']['similarity_thred'], help='Threshold for model merging')

    args = parser.parse_args()
    args = set_gpu(args)
    set_seed(args.seed)

    t1 = time.time()
    args, loader = load_dataset(args)
    ARCUS_instance = ARCUS(args)
    returned, auc_hist, anomaly_scores = ARCUS_instance.simulator(loader)
    total_time = time.time()-t1
    with open(f'{config["output path"]}', mode='a+') as f:
        print(f'Algorithm: {config["name"]}\n'
              f'File name: {config["input file"]}\n'
              f'ROC: {np.round(np.mean(auc_hist), 4):.4f}\n'
              f'Running time: {total_time:.4f}\n'
              f'model_type: {config["argument"]["model_type"]}\n'
              f'inf_type: {config["argument"]["inf_type"]}\n'
              f'seed: {config["argument"]["seed"]}\n'
              f'gpu: {config["argument"]["gpu"]}\n'
              f'batch_size: {config["argument"]["batch_size"]}\n'
              f'min_batch_size: {config["argument"]["min_batch_size"]}\n'
              f'init_epoch: {config["argument"]["init_epoch"]}\n'
              f'intm_epoch: {config["argument"]["intm_epoch"]}\n'
              f'hidden_dim: {config["argument"]["hidden_dim"]}\n'
              f'layer_num: {config["argument"]["layer_num"]}\n'
              f'learning_rate: {config["argument"]["learning_rate"]}\n'
              f'reliability_thred: {config["argument"]["reliability_thred"]}\n'
              f'similarity_thred: {config["argument"]["similarity_thred"]}\n',
              end='\n\n',
              file=f)
