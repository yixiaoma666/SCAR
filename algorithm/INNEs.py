from pyod.models.inne import INNE
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import trange
import os
import time
import warnings

class INNEs:
    def __init__(self,
                 X: np.ndarray,
                 psi=2,
                 t=100,
                 W=10,
                 output_index=[-1, 0]) -> None:
        self.X = X
        self.psi = psi
        self.t = t
        self.W = W
        self.output_index = np.array(
            [x + self.W if x < 0 else x for x in output_index])

        self.score_dict = defaultdict(list)

        self.n, self.dim = X.shape[0], X.shape[1]

        self.main()

    def main(self):
        for now in trange(0, self.n-self.W+1):
            detector = INNE(self.t, self.psi)
            detector.fit(self.X[now:now+self.W, :])
            scores = detector.decision_function(self.X[self.output_index+now])
            for i, idx in enumerate(self.output_index):
                self.score_dict[idx+now].append(scores[i])

def main(config):
    warnings.filterwarnings("ignore")
    dl = np.loadtxt(
        f'{os.path.join(config["input path"], "csv", config["input file"]+".csv")}', delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    t = time.time()
    detector = INNEs(data,
                    psi=config['argument']['psi'],
                    t=config['argument']['t'],
                    W=config['argument']['window_size'])
    keys = list(detector.score_dict.keys())
    keys.sort()
    scores = []
    for key in keys:
        scores.append(min(detector.score_dict[key]))
    total_time = time.time() - t

    with open(f'{config["output path"]}', mode='a+') as f:
        print(f'Algorithm: {config["name"]}\n'
              f'File name: {config["input file"]}\n'
              f'ROC: {roc_auc_score(label, scores):.4f}\n'
              f'Running time: {total_time:.4f}\n'
              f'psi: {config["argument"]["psi"]}\n'
              f't: {config["argument"]["t"]}\n'
              f'window_size: {config["argument"]["window_size"]}\n',
              end='\n\n',
              file=f)
    pass


if __name__ == '__main__':
    # files = ['very_easy.csv', 'mix_mnist_023.csv', 'mix_mnist_479.csv', 'mix_COIL20_3toy.csv', 'mix_COIL20_3car.csv', 'shake_breast_far_easy.csv',
    #          'shake_breast_2near.csv', 'shake_Ionosphere_far_easy.csv', 'shake_Ionosphere_2near.csv', 'shake_WDBC_far_easy.csv', 'shake_WDBC_2near_easy.csv',
    #          'norm_long.csv', '3cylinder_big_small.csv']
    files = ['3cylinder_small_big.csv']
    for file in files:
        path = os.path.join('pysad_data', file)
        # path = 'new_data/3cylinder_big_small.csv'

        dl = np.loadtxt(path, delimiter=',')[:]
        data = dl[:, :-2]
        label = dl[:, -2]

        i = 0
        # for psi in ([64, 48, 32, 24, 16, 12, 8, 4, 2]):
        for psi in ([32]):
            for W in [100 * i for i in range(1, 11)]:
            # for W in [100]:
                detector = INNEs(data, psi=psi, t=100,
                                    W=W, output_index=[-1, 0])
                keys = list(detector.score_dict.keys())
                keys.sort()
                scores = []
                for key in keys:
                    scores.append(min(detector.score_dict[key]))
                    
                    
                # np.savetxt(f'new_result/innes_{file[:-4]}_{W}_{i}.csv', scores, delimiter=',')
                i += 1
            # print(psi, end=',')
            # print(roc_auc_score(label, scores))
            
            
            # f = open('result.csv', mode='a+')
            # print(file, psi, roc_auc_score(label, scores), sep=',', file=f, flush=True)
            # f.close()
