# Import modules.
from pysad.evaluation import AUROCMetric
from pysad.models import RSHash
from pysad.utils import ArrayStreamer
from tqdm import tqdm
import numpy as np
import os
import time


def main(config):
    dl = np.loadtxt(os.path.join(
        config['input path'], 'csv', config['input file']+".csv"), delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    iterator = ArrayStreamer(shuffle=False)
    model = RSHash(**dict({'feature_mins': np.array(np.min(data)).reshape(-1),
                           'feature_maxes': np.array(np.max(data)).reshape(-1)}, **config['argument']))
    auroc = AUROCMetric()
    t = time.time()
    for x, y in tqdm(iterator.iter(data, label)):
        score = model.fit_score_partial(x)
        auroc.update(y, -score)
    total_time = time.time() - t
    with open(f'{config["output path"]}', mode='a+') as f:
        print(f'Algorithm: {config["name"]}\n'
              f'File name: {config["input file"]}\n'
              f'ROC: {auroc.get():.4f}\n'
              f'Running time: {total_time:.4f}\n'
              f'sampling_points: {config["argument"]["sampling_points"]}\n'
              f'decay: {config["argument"]["decay"]}\n'
              f'num_components: {config["argument"]["num_components"]}\n'
              f'num_hash_fns: {config["argument"]["num_hash_fns"]}\n',
              end='\n\n',
              file=f)
