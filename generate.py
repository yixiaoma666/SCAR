from scipy import stats
import json
import numpy as np
import hdf5storage
import argparse
import os

def get_s_distri(distri_arg):
    distri_name = distri_arg['name']
    del distri_arg['name']
    distri: stats.rv_continuous = eval(f'stats.{distri_name}')

    def get_data(size):
        return distri.rvs(**distri_arg, size=size).reshape(-1, 1)
    return get_data


def get_r_distri(path):
    dl = np.loadtxt(path, delimiter=',')
    output = {}
    label = set(dl[:, -1].tolist())
    data_dict = {}
    for each in label:
        data_dict[int(each)] = dl[np.where(dl[:, -1] == each)][:, :-1]
    output = {}

    def get_data(each):
        def _get_data(size):
            return data_dict[int(each)][np.random.choice(np.shape(data_dict[int(each)])[0], size, replace=True)]
        return _get_data
    for each in label:
        output[each] = get_data(each)
    return output


def get_data_from_concept(concept, distri_dict):
    if concept['type'] == 's':
        data_list = []
        for each in concept['distri_list']:
            data_list.append(distri_dict[each](concept['size']))
        data_array = np.array(data_list)
        weight = np.zeros(np.shape(data_array))
        index = np.random.randint(
            len(concept['distri_list']), size=np.shape(weight)[1])
        for i in range(len(concept['distri_list'])):
            weight[i, np.where(index == i), :] = 1

        data: np.ndarray = np.sum(data_array*weight, axis=0)
        label = np.zeros((np.shape(data)[0], 1))
        drift = np.zeros((np.shape(data)[0], 1))
        if concept['ano_type'] == 'mix' and len(concept['ano_list']) != 0:
            ano_data_list = []
            for each in concept['ano_list']:
                ano_data_list.append(distri_dict[each](concept['size']))
            ano_data_array = np.array(ano_data_list)
            ano_weight = np.zeros(np.shape(ano_data_array))
            ano_index = np.random.randint(
                len(concept['ano_list']), size=np.shape(ano_weight)[1])
            for i in range(len(concept['ano_list'])):
                ano_weight[i, np.where(ano_index == i), :] = 1

            ano_data = np.sum(ano_data_array*ano_weight, axis=0)

            all_data = np.array([data, ano_data])

            mix_weight = np.zeros(np.shape(all_data))
            mix_index = np.random.rand(np.shape(mix_weight)[1])

            mix_weight[0, np.where(mix_index > concept['ano_rate']), :] = 1
            mix_weight[1, np.where(mix_index <= concept['ano_rate']), :] = 1

            data = np.sum(all_data*mix_weight, axis=0)
            label[mix_index < concept['ano_rate']] = 1
        elif concept['ano_type'] == 'shake':
            mm_list = np.concatenate((np.min(data, axis=0).reshape(1, -1), np.max(data, axis=0).reshape(1, -1)), axis=0)
            for _n in range(data.shape[0]):
                if np.random.random() < concept['ano_rate']:
                    for _dim in range(data.shape[1]):
                        if np.random.random() < concept['dim_rate']:
                            ano_range = concept['ano_range_list'][np.random.choice(len(concept['ano_range_list']))]
                            shake = (np.random.uniform(min(ano_range), max(ano_range)) - mm_list[0, _dim]) * (mm_list[1, _dim] - mm_list[0, _dim])
                            data[_n, _dim] = shake
                            label[_n, 0] = 1
        drift[0, 0] = 1

        output = np.concatenate((data, label, drift), axis=1)

        return output

    elif concept['type'] == 'g':
        data_list1 = []
        for each in concept['distri_list1']:
            data_list1.append(distri_dict[each](concept['size']))
        data_array1 = np.array(data_list1)
        weight1 = np.zeros(np.shape(data_array1))
        index1 = np.random.randint(
            len(concept['distri_list1']), size=np.shape(weight1)[1])
        for i in range(len(concept['distri_list1'])):
            weight1[i, np.where(index1 == i), :] = 1

        data1 = np.sum(data_array1*weight1, axis=0)

        data_list2 = []
        for each in concept['distri_list2']:
            data_list2.append(distri_dict[each](concept['size']))
        data_array2 = np.array(data_list2)
        weight2 = np.zeros(np.shape(data_array2))
        index2 = np.random.randint(
            len(concept['distri_list2']), size=np.shape(weight2)[1])
        for i in range(len(concept['distri_list2'])):
            weight2[i, np.where(index2 == i), :] = 1

        data2 = np.sum(data_array2*weight2, axis=0)

        normal_data = np.array([data1, data2])
        normal_weight = np.zeros(np.shape(normal_data))
        normal_index = np.random.rand(np.shape(normal_weight)[1])

        rate_index = np.linspace(0, 1, np.shape(normal_weight)[1])
        normal_weight[0, np.where(normal_index > rate_index), :] = 1
        normal_weight[1, np.where(normal_index <= rate_index), :] = 1

        data = np.sum(normal_data*normal_weight, axis=0)
        label = np.zeros((np.shape(data)[0], 1))
        drift = np.zeros((np.shape(data)[0], 1))

        if concept['ano_type'] == 'mix' and len(concept['ano_list']) != 0:
            ano_data_list = []
            for each in concept['ano_list']:
                ano_data_list.append(distri_dict[each](concept['size']))
            ano_data_array = np.array(ano_data_list)
            ano_weight = np.zeros(np.shape(ano_data_array))
            ano_index = np.random.randint(
                len(concept['ano_list']), size=np.shape(ano_weight)[1])
            for i in range(len(concept['ano_list'])):
                ano_weight[i, np.where(ano_index == i), :] = 1

            ano_data = np.sum(ano_data_array*ano_weight, axis=0)

            all_data = np.array([data, ano_data])

            mix_weight = np.zeros(np.shape(all_data))
            mix_index = np.random.rand(np.shape(mix_weight)[1])

            mix_weight[0, np.where(mix_index > concept['ano_rate']), :] = 1
            mix_weight[1, np.where(mix_index <= concept['ano_rate']), :] = 1

            data = np.sum(all_data*mix_weight, axis=0)
            label[mix_index < concept['ano_rate']] = 1
        drift[:, 0] = 1

        output = np.concatenate((data, label, drift), axis=1)

        return output

    elif concept['type'] == 'i':
        data_list1 = []
        for each in concept['distri_list1']:
            data_list1.append(distri_dict[each](concept['size']))
        data_array1 = np.array(data_list1)
        weight1 = np.zeros(np.shape(data_array1))
        index1 = np.random.randint(
            len(concept['distri_list1']), size=np.shape(weight1)[1])
        for i in range(len(concept['distri_list1'])):
            weight1[i, np.where(index1 == i), :] = 1

        data1 = np.sum(data_array1*weight1, axis=0)

        data_list2 = []
        for each in concept['distri_list2']:
            data_list2.append(distri_dict[each](concept['size']))
        data_array2 = np.array(data_list2)
        weight2 = np.zeros(np.shape(data_array2))
        index2 = np.random.randint(
            len(concept['distri_list2']), size=np.shape(weight2)[1])
        for i in range(len(concept['distri_list2'])):
            weight2[i, np.where(index2 == i), :] = 1

        data2 = np.sum(data_array2*weight2, axis=0)

        normal_data = np.array([data1, data2])
        normal_index = np.random.rand(np.shape(normal_data)[1])

        rate_index1 = np.linspace(1, 0, np.shape(normal_data)[1])
        rate_index2 = np.linspace(0, 1, np.shape(normal_data)[1])
        rate_index = np.array([rate_index1, rate_index2])
        rate_index = rate_index[:, :, np.newaxis]
        rate_index = np.repeat(rate_index, np.shape(normal_data)[2], 2)

        data = np.sum(normal_data*rate_index, axis=0)
        label = np.zeros((np.shape(data)[0], 1))
        drift = np.zeros((np.shape(data)[0], 1))

        if concept['ano_type'] == 'mix' and len(concept['ano_list']) != 0:
            ano_data_list = []
            for each in concept['ano_list']:
                ano_data_list.append(distri_dict[each](concept['size']))
            ano_data_array = np.array(ano_data_list)
            ano_weight = np.zeros(np.shape(ano_data_array))
            ano_index = np.random.randint(
                len(concept['ano_list']), size=np.shape(ano_weight)[1])
            for i in range(len(concept['ano_list'])):
                ano_weight[i, np.where(ano_index == i), :] = 1

            ano_data = np.sum(ano_data_array*ano_weight, axis=0)

            all_data = np.array([data, ano_data])

            mix_weight = np.zeros(np.shape(all_data))
            mix_index = np.random.rand(np.shape(mix_weight)[1])

            mix_weight[0, np.where(mix_index > concept['ano_rate']), :] = 1
            mix_weight[1, np.where(mix_index <= concept['ano_rate']), :] = 1

            data = np.sum(all_data*mix_weight, axis=0)
            label[mix_index < concept['ano_rate']] = 1
        drift[:, 0] = 1

        output = np.concatenate((data, label, drift), axis=1)

        return output



def print_info(output):
    print(f'# of instances is {np.shape(output)[0]}')
    print(f'# of anomaly is {np.sum(output[:, -2])}')
    print(
        f'% of anomaly is {np.sum(output[:, -2])/np.shape(output)[0]*100:.3f}%')


def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='config/demo_r_s.json')
    args = parser.parse_args()
    paths = args.path
    if paths.endswith('.json'):
        path, files = '/'.join(paths.split('\\')[:-1]), [paths.split('\\')[-1]]
    else:
        path = paths
        files = os.listdir(paths)
    for file in files:
        with open(os.path.join(path, file)) as f:
            argument = json.loads(f.read())
        if argument['s_r'] == 'synthetic':
            distri_dict = {}
            for key, value in argument['distri_list'].items():
                distri_dict[key] = get_s_distri(value)
            output = np.ndarray((0, 3))
            for concept in argument['streams']:
                output = np.concatenate(
                    (output, get_data_from_concept(concept, distri_dict)), axis=0)
        elif argument['s_r'] == 'real':
            distri_dict = get_r_distri(argument['input_path'])
            output = np.ndarray((0, np.shape(distri_dict[list(distri_dict.keys())[0]](1))[1]+2))
            for concept in argument['streams']:
                output = np.concatenate(
                    (output, get_data_from_concept(concept, distri_dict)), axis=0)
        output[0, -1] = 0
        np.savetxt(os.path.join(argument['output_path'], 'csv', argument['output_filename']+'.csv'), output, delimiter=',')
        hdf5storage.savemat(os.path.join(argument['output_path'], 'mat', argument['output_filename']+'.mat'), {'Y': output[:, :-2], 'L': output[:, -2].reshape(-1, 1), 'C': output[:, -1].reshape(-1, 1)})
        print(f'Output stream data is in {argument["output_path"]}')
        print_info(output)


if __name__ == '__main__':
    main()
