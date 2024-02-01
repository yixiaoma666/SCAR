# Import modules.
from pysad.evaluation import AUROCMetric
from pysad.models import RobustRandomCutForest
from pysad.utils import ArrayStreamer
from pysad.utils import Data
from tqdm import tqdm
import numpy as np
import os
import time


def main(config):
    dl = np.loadtxt(os.path.join(
        config['input path'], 'csv', config['input file']+".csv"), delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    iterator = ArrayStreamer(shuffle=False)
    model = RobustRandomCutForest(**config['argument'])
    auroc = AUROCMetric()
    t = time.time()
    for x, y in tqdm(iterator.iter(data, label)):
        score = model.fit_score_partial(x)
        auroc.update(y, score)
    total_time = time.time() - t
    with open(f'{config["output path"]}', mode='a+') as f:
        print(f'Algorithm: {config["name"]}\n'
              f'File name: {config["input file"]}\n'
              f'ROC: {auroc.get():.4f}\n'
              f'Running time: {total_time:.4f}\n'
              f'num_trees: {config["argument"]["num_trees"]}\n'
              f'shingle_size: {config["argument"]["shingle_size"]}\n'
              f'tree_size: {config["argument"]["tree_size"]}\n',
              end='\n\n',
              file=f)
