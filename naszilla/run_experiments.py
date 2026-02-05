import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np
import copy

from naszilla.params import *
from naszilla.nas_benchmarks import Nasbench101, Nasbench201, Nasbench301
from naszilla.nas_algorithms import run_nas_algorithm

def run_experiments(args, save_dir):
    """
    Run NAS algorithms based on the provided arguments and save the results.
    Args:
        args: Parsed command-line arguments containing experiment configurations.
        save_dir: Directory to save the experiment results.
    """

    # set up arguments
    trials = args.trials
    queries = args.queries
    out_file = args.output_filename
    save_specs = args.save_specs
    metann_params = meta_neuralnet_params(args.metann_params)
    ss = args.search_space
    dataset = args.dataset
    mf = args.mf
    algorithm_params = algo_params(args.algo_params, queries=queries)
    num_algos = len(algorithm_params)
    logging.info(algorithm_params)

    # deep copy of metann_params for each algorithm
    mp = copy.deepcopy(metann_params) # a dict of surrogate model hyperparameters (epochs, batch size, etc)

    # initialize search space
    if ss == 'nasbench_101':
        search_space = Nasbench101(mf=mf)
    elif ss == 'nasbench_201':
        search_space = Nasbench201(dataset=dataset)
    elif ss == 'nasbench_301':
        search_space = Nasbench301()
    else:
        print('Invalid search space')
        raise NotImplementedError()

    # Outer loop: Repeat the whole comparison for multiple trials times
    for i in range(trials):
        results = []
        val_results = []
        walltimes = []
        run_data = []

        # Inner loop: run each NAS algorithm (random, evolution, bananas, etc)
        for j in range(num_algos):
            print('\n* Running NAS algorithm: {}'.format(algorithm_params[j]))
            starttime = time.time()
            # this line runs the nas algorithm and returns the result
            result, val_result, run_datum = run_nas_algorithm(algorithm_params[j], search_space, mp)

            result = np.round(result, 5)
            val_result = np.round(val_result, 5)

            # remove unnecessary dict entries that take up space,
            for d in run_datum:
                if not save_specs:
                    d.pop('spec')
                for key in ['encoding', 'adj', 'path', 'dist_to_min']:
                    if key in d:
                        d.pop(key)

            # add walltime, results, run_data
            walltimes.append(time.time()-starttime)
            results.append(result)
            val_results.append(val_result)
            run_data.append(run_datum)

        # print and pickle results
        filename = os.path.join(save_dir, '{}_{}.pkl'.format(out_file, i))
        print('\n* Trial summary: (params, results, walltimes)')
        print(algorithm_params)
        print(ss)
        print(results)
        print(walltimes)
        print('\n* Saving to file {}'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump([algorithm_params, metann_params, results, walltimes, run_data, val_results], f)
            f.close()

def main(args):

    # make save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    algo_params = args.algo_params
    save_path = save_dir + '/' + algo_params + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run_experiments(args, save_path)
    

if __name__ == "__main__":
    """
    The function of each argument:
    --trials: number of trials to run the NAS algorithm
    --queries: number of queries/evaluations each NAS algorithm gets
    --search_space: which NAS benchmark to use (nasbench_101, nasbench_201, or nasbench_301), default 'nasbench_101'
    --dataset: which dataset to use (cifar10, 100, or imagenet for nasbench201), default 'cifar10'
    --mf: whether to use multi-fidelity (only for nasbench101), means using different epochs for training, default false
    --metann_params: which parameters to use for the meta neural net surrogate model, predefined options are 'standard' and 'diverse', default 'standard'
    --algo_params: which parameters to use for the NAS algorithm, default 'simple_algos'
    --output_filename: name of output files, default 'round'
    --save_dir: name of save directory, default 'results_output'
    --save_specs: whether to save the architecture specs in the output files, default false
    """
    parser = argparse.ArgumentParser(description='Args for BANANAS experiments')
    parser.add_argument('--trials', type=int, default=500, help='Number of trials')
    parser.add_argument('--queries', type=int, default=150, help='Max number of queries/evaluations each NAS algorithm gets')
    parser.add_argument('--search_space', type=str, default='nasbench_101', help='nasbench_101, _201, or _301')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, 100, or imagenet (for nasbench201)')
    parser.add_argument('--mf', type=bool, default=False, help='Multi fidelity: true or false (for nasbench101)')
    parser.add_argument('--metann_params', type=str, default='standard', help='which parameters to use')
    parser.add_argument('--algo_params', type=str, default='simple_algos', help='which parameters to use')
    parser.add_argument('--output_filename', type=str, default='round', help='name of output files')
    parser.add_argument('--save_dir', type=str, default='results_output', help='name of save directory')
    parser.add_argument('--save_specs', type=bool, default=False, help='save the architecture specs')    

    args = parser.parse_args()
    main(args)
