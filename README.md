# Data Generation

If you want to generate data, execute the following code:

```
python generate.py --path generate_config/demo_synthetic.json
```

The `--path` parameter is the streaming data configuration file. If you want to generate multiple streams at once, set the `--path` parameter to a folder.

## Tools Introduction

The tool is able to generate streaming data with a concept drift, and there are two ways to obtain the base distribution:

1. Use the functions in `scipy.stats` as the base distribution
2. Use data from clustered/classified datasets as base distributions

Later use these base distributions to simulate streaming data.


Configure the file in `.json`.

`s_r` is the source of the base distribution, synthetic is using the distribution in `scipy.stats`, and real is using the clustered/classified dataset

The `distri_list` is unique to synthetic, and is a collection of parameters used for distributions in streaming data, the parameters can be found [here](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions)

`input_path` is a parameter unique to real, it is the path to the dataset to be imported.

`output_path` is the path to the streaming data dataset to be exported.

`output_filename` is the filename of the exported streaming data dataset.

The `streams` parameter is the parameter to simulate the streaming data, which is composed of multiple segments of data, where each parameter in the list represents how this segment is generated, where these parameters:

`type` has three values, s, g and i.

When `type` is `s`, the segment has a data length of `size`, normal data is taken with equal probability from the distribution in `distri_list`, and anomaly data is taken with equal probability from the distribution in `ano_list`, and the normal data is replaced with a probability of `ano_rate`. The anomaly label is noted as 1 at the location of the anomaly data replacement, and the drift label is noted as 1 at the beginning of the segment

When `type` is `g`, the segment has a data length of `size`, and the normal data $X_1$, $X_2$ distributions are taken with equal probability from the distributions in `distri_list1` and `distri_list2`, and are followed by the replacement of the $X_2$ with a $[0,\frac{1}{size},\frac{2}{size},\frac{size-1}{size}]$ with probability of replacing $X_1$ to simulate the process of gradual (gradual) drift of the distribution, where the anomalous data are taken with equal probability from the distribution in `ano_list` and the normal data are replaced with probability of `ano_rate`. The location where the anomalous data is replaced is noted as 1 for the anomaly label and 1 for all drift labels in the segment

When `type` is `i`, the length of the segment is `size`, and the normal data $X_1$, $X_2$ distributions are taken with equal probability from the distributions in `distri_list1` and `distri_list2`, and the mixing weight is noted as $weight=[0,\frac{1}{size},\frac{2}{size}. \frac{size-1}{size}]$, and the final data is $(1-weight)\cdot X_1+weight\cdot X_2$, with anomalous data taken with equal probability from the distribution in the `ano_list` and normal data replaced with probability at the `ano_rate`. The location of the anomaly data replacement is noted as 1 for the anomaly label and 1 for all drift labels in the segment.

When `ano_type` is `shake`, outliers will be randomly sampled from a uniform distribution of `ano_range_list`, `dim_rate` is the proportion of dimensions modified, and `ano_rate` is the proportion of outliers.

When `ano_type` is `mix`, outliers are sampled from the `ano_list` and inserted into the normal distribution in proportion to the `ano_rate`.

# Datasets

The dataset is available at [here](https://drive.google.com/drive/folders/1ac3qX7-Amq-zEgBQeWetR1ZE9CSkuAXB?usp=drive_link).


# Test Algorithms

The algorithms **STORM**, **HSTree**, **IForestASD**, **LODA**, **RSHash**, **xStream**, **RRCF**, **Memstream**, **ARCUS**, **IDKs**, and **INNEs** are tested in a Windows environment.

The algorithm configuration file is composed of `.json` files, an array containing multiple objects, each object corresponding to the run configuration of an algorithm. When `run_algorithm.py` is executed, the configured algorithms are executed sequentially.

```
pip install -r requirements.txt
```

```
python run_algorithm.py -r demo_config.json
```

The algorithm **Mstream** is tested in a Linux environment.


Here is the algorithm configuration file:

## STORM

STORM's default configuration file and parameters

```json
    {
        "name": "STORM",
        "input path": "your input path",
        "input file": "your input file",
        "output path": "your output path",
        "argument":{
            "window_size": 100,
            "max_radius": 0.1
        }
    }
```

## HSTree

HSTree's default configuration file and parameters

```json
    {
        "name": "HSTree",
        "input path": "your input path",
        "input file": "your input file",
        "output path": "your output path",
        "argument":{
            "window_size": 100,
            "num_trees": 25,
            "max_depth": 15
        }
    }
```

## IForestASD

IForestASD's default configuration file and parameters

```json
    {
        "name": "IForestASD",
        "input path": "your input path",
        "input file": "your input file",
        "output path": "your output path",
        "argument":{
            "window_size": 2048
        }
    }
```

## LODA

LODA's default configuration file and parameters

```json
    {
        "name": "LODA",
        "input path": "your input path",
        "input file": "your input file",
        "output path": "your output path",
        "argument":{
            "num_bins": 10,
            "num_random_cuts": 100
        }
    }
```

## RSHash

RSHash's default configuration file and parameters

```json
    {
        "name": "RSHash",
        "input path": "your input path",
        "input file": "your input file",
        "output path": "your output path",
        "argument":{
            "sampling_points": 1000,
            "decay": 0.015,
            "num_components": 100,
            "num_hash_fns": 1
        }
    }
```

## xStream

xStream's default configuration file and parameters

```json
    {
        "name": "xStream",
        "input path": "your input path",
        "input file": "your input file",
        "output path": "your output path",
        "argument":{
            "num_components": 100,
            "n_chains": 100,
            "depth": 25,
            "window_size": 25
        }
    }
```

## RRCF

RRCF's default configuration file and parameters

```json
    {
        "name": "RRCF",
        "input path": "your input path",
        "input file": "your input file",
        "output path": "your output path",
        "argument":{
            "num_trees": 4,
            "shingle_size": 4,
            "tree_size": 256
        }
    }
```

## Memstream

Memstream's default configuration file and parameters

```json
    {
        "name": "Memstream",
        "input path": "your input path",
        "input file": "your input file",
        "output path": "your output path",
        "argument":{
            "beta": 4,
            "memlen": 4
        }
    }
```

## IDKs

IDKs's default configuration file and parameters

```json
    {
        "name": "IDKs",
        "input path": "your input path",
        "input file": "your input file",
        "output path": "your output path",
        "argument":{
            "psi": 8,
            "t": 100,
            "window_size": 100
        }
    }
```

## INNEs

INNEs's default configuration file and parameters

```json
    {
        "name": "INNEs",
        "input path": "your input path",
        "input file": "your input file",
        "output path": "your output path",
        "argument":{
            "psi": 8,
            "t": 100,
            "window_size": 100
        }
    }
```

## ARCUS

ARCUS's default configuration file and parameters

```json
    {
        "name": "ARCUS",
        "input path": "stream_data",
        "input file": "demo_real_shake",
        "output path": "result",
        "argument":{
            "model_type": "RAPP",
            "inf_type": "ADP",
            "seed": 0,
            "gpu": "0",
            "batch_size": 512,
            "min_batch_size": 32,
            "init_epoch": 5,
            "intm_epoch": 1,
            "hidden_dim": 24,
            "layer_num": 3,
            "learning_rate": 1e-4,
            "reliability_thred": 0.95,
            "similarity_thred": 0.80
        }
    }
```

## Mstream

See [here](https://github.com/Stream-AD/MStream)

# Experiment Evaluation

| methods    | AUC_ROC      |
|------------|--------------|
| LODA       | 0.500041892  |
| Memstream  | 0.528944595  |
| Mstream    | 0.601215278  |
| RSHash     | 0.693941892  |
| HSTree     | 0.769993243  |
| STORM      | 0.850366216  |
| xStream    | 0.904577027  |
| IForestASD | 0.938587838  |
| RRCF       | 0.963112162  |
| IForests   | 0.96848      |
| LOFs       | 0.970574324  |
| IDKs       | 0.970804054  |
| INNEs      | 0.972421622  |

