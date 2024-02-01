import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', '-r',
                        default='demo_config.json',
                        type=str,
                        help='Path of running algorithm configuration')
    args = parser.parse_args()
    with open(args.run_config) as f:
        configs = json.loads(f.read())
    for config in configs:
        if config['name'] == 'STORM':
            from algorithm import STORM
            STORM.main(config)
        elif config['name'] == 'HSTree':
            from algorithm import HSTree
            HSTree.main(config)
        elif config['name'] == 'IForestASD':
            from algorithm import IForestASD
            IForestASD.main(config)
        elif config['name'] == 'LODA':
            from algorithm import LODA
            LODA.main(config)
        elif config['name'] == 'RSHash':
            from algorithm import RSHash
            RSHash.main(config)
        elif config['name'] == 'xStream':
            from algorithm import xStream
            xStream.main(config)
        elif config['name'] == 'RRCF':
            from algorithm import RRCF
            RRCF.main(config)
        elif config['name'] == 'Memstream':
            from algorithm import Memstream
            Memstream.main(config)
        elif config['name'] == 'IDKs':
            from algorithm import IDKs
            IDKs.main(config)
        elif config['name'] == 'INNEs':
            from algorithm import INNEs
            INNEs.main(config)
        elif config['name'] == 'ARCUS':
            from algorithm.ARCUS import run_ARCUS
            run_ARCUS.main(config)



if __name__ == '__main__':
    main()