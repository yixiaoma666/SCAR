# Import modules.
from pysad.evaluation import AUROCMetric
from pysad.models import IForestASD
from pysad.utils import ArrayStreamer
from pysad.utils import Data
from tqdm import tqdm
import numpy as np
import os
import time
import warnings


def main(config):
    warnings.filterwarnings("ignore")
    dl = np.loadtxt(os.path.join(
        config['input path'], 'csv', config['input file']+".csv"), delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    iterator = ArrayStreamer(shuffle=False)
    model = IForestASD(**config['argument'])
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
              f'window_size: {config["argument"]["window_size"]}\n',
              end='\n\n',
              file=f)
