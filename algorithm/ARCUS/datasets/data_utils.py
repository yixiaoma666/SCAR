import pandas as pd
import numpy as np
import tensorflow as tf

def load_dataset(args):
    dld = np.loadtxt(f'./{args.dataset_name[0]}/csv/{args.dataset_name[1]}.csv', delimiter=',')
    loader = tf.data.Dataset.from_tensor_slices((dld[:, :-2], dld[:, -2]))
    args.input_dim = loader.element_spec[0].shape[0]
    return args, loader