import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time
import os
from typing import List


class IDK:
    def __init__(self, X, psi=2, t=100, sample=None) -> None:
        self.X = X
        self.psi = psi
        self.t = t
        self.sample = sample

        self.center_list = self.get_center_radius()
        self.true_center_list = self.X[self.center_list.reshape(-1)]
        self.radius_list, self.feature_map = self.IK_inne()

        self.feature_mean_map = np.mean(self.feature_map, axis=0)

    def get_center_radius(self):
        center_list = []
        if self.sample is None:
            for _ in range(self.t):
                center_list.append(np.random.choice(
                    self.X.shape[0], self.psi, replace=False))
        else:
            center_list = self.sample
        return np.array(center_list).reshape(self.t, -1)

    def IK_inne(self):
        radius_list = []
        output = np.zeros((self.X.shape[0], self.psi*self.t))
        for i in range(self.t):
            sample = self.X[self.center_list[i]]

            tem1 = np.dot(np.square(self.X), np.ones(sample.T.shape))
            tem2 = np.dot(np.ones(self.X.shape), np.square(sample.T))
            p2s = tem1 + tem2 - 2 * np.dot(self.X, sample.T)
            s2s = p2s[self.center_list[i], :]

            row, col = np.diag_indices_from(s2s)
            s2s[row, col] = np.inf
            temp_radius_list = np.min(s2s, axis=0)
            radius_list.append(temp_radius_list)

            p2ns_index = np.argmin(p2s, axis=1)
            p2ns = p2s[range(self.X.shape[0]), p2ns_index]
            ind = p2ns < temp_radius_list[p2ns_index]
            output[ind, (p2ns_index+i*self.psi)[ind]] = 1
        return np.array(radius_list), output

    def IDK_score(self):
        return np.dot(self.feature_map, self.feature_mean_map) / self.t

    def get_fm(self, data):
        data = data.reshape(1, -1)
        output = np.zeros((data.shape[0], self.psi*self.t))

        tem1 = np.dot(np.square(data), np.ones(self.true_center_list.T.shape))
        tem2 = np.dot(np.ones(data.shape), np.square(self.true_center_list.T))
        p2s = tem1 + tem2 - 2 * np.dot(data, self.true_center_list.T)

        p2ns_index = np.argmin(p2s, axis=1)
        p2ns = p2s[range(data.shape[0]), p2ns_index]
        ind = p2ns < self.radius_list.reshape(-1)[p2ns_index]
        output[ind, (p2ns_index)[ind]] = 1
        return output

    def get_given_score(self, index):
        index = [x + self.X.shape[0] if x < 0 else x for x in index]
        given_fm = self.feature_map[index]
        given_score = np.dot(given_fm, self.feature_mean_map) / self.t
        output = dict(zip(index, given_score))
        return output


class IDKs:
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
        self.output_index = output_index

        self.score_dict: dict = dict()
        self.main()

        pass

    def main(self):
        score_dict_list: List[dict] = []
        for _ in tqdm(range(self.t)):
            iidk_single = IDKs_1(
                self.X, self.psi, self.W, self.output_index)
            score_dict_list.append(iidk_single.score_dict)
        for key in score_dict_list[0].keys():
            self.score_dict[key] = [
                sum(x)/self.t for x in zip(*[x[key] for x in score_dict_list])]

    @staticmethod
    def dict_add(d1: dict, d2: dict):
        output = dict()
        for key in d1.keys():
            output[key] = d1[key] + d2[key]
        return output


class IDKs_1:
    def __init__(self,
                 X: np.ndarray,
                 psi: int = 2,
                 W: int = 10,
                 output_index: list = [-1]) -> None:
        self.X = X
        self.psi = psi
        self.W = W
        self.output_index = [x + self.W if x < 0 else x for x in output_index]
        self.now = 0
        self.rebuild_flag = True
        self.fm: np.ndarray
        self.fmm: np.ndarray
        self.detector: IDK

        self.sample_index = self.get_index()
        self.score_dict: dict = dict()
        self.get_score_dict()

    def slide_window(self):
        self.now += 1
        self.rebuild_flag = False
        if not self.all_in_interval(self.sample_index):
            self.rebuild_flag = True
            self.sample_index = self.get_index()

    def random_sample(self,
                      X: np.ndarray,
                      n: int,
                      replace: bool = False):
        return X[np.random.choice(X.shape[0], n, replace=replace)]

    def in_interval(self, x):
        return self.now <= x < self.now + self.W

    def all_in_interval(self, X):
        for each in X:
            if not self.in_interval(each):
                return False
        return True

    def get_index(self):
        return np.random.choice(self.W, self.psi, False) + self.now

    @property
    def stop(self):
        return self.now + self.W > self.X.shape[0]

    def IDK(self):
        if self.rebuild_flag:
            self.detector = IDK(
                self.X[self.now: self.now + self.W], self.psi, 1, self.sample_index - self.now)
            self.fm = self.detector.feature_map
            score = self.fm2score()
            return score
        else:
            new_fm = self.detector.get_fm(self.X[self.now + self.W - 1])
            self.fm = np.concatenate((self.fm, new_fm), axis=0)
            self.fm = np.delete(self.fm, 0, axis=0)
            score = self.fm2score()
            return score

    def test(self):
        print(self.all_in_interval([-1, 0, 1, 2]))

    def fm2score(self):
        self.fmm = np.mean(self.fm, axis=0)
        score = self.fm[self.output_index].dot(self.fmm)
        return dict(zip([x + self.now for x in self.output_index], score))

    def get_score_dict(self):
        while not self.stop:
            temp_score_dict = self.IDK()
            for key, value in temp_score_dict.items():
                if key not in self.score_dict.keys():
                    self.score_dict[key] = [value]
                else:
                    self.score_dict[key].append(value)
            self.slide_window()


def main(config):
    dl = np.loadtxt(
        f'{os.path.join(config["input path"], "csv", config["input file"]+".csv")}', delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    t = time.time()
    detector = IDKs(data,
                    psi=config['argument']['psi'],
                    t=config['argument']['t'],
                    W=config['argument']['window_size'])
    keys = list(detector.score_dict.keys())
    keys.sort()
    scores = []
    for key in keys:
        scores.append(-max(detector.score_dict[key]))
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
