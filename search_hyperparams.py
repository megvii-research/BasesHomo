"""Peform hyperparemeters search"""

import argparse
import collections
import itertools
import os
import sys

from common import utils
from experiment_dispatcher import dispatcher, tmux

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments', help='Directory containing params.json')
parser.add_argument('--id', default=1, type=int, help="Experiment id")


def launch_training_job(exp_dir, exp_name, session_name, param_pool_dict, params, start_id=0):
    # Partition tmux windows automatically
    tmux_ops = tmux.TmuxOps()
    # Combining hyper-parameters and experiment ID automatically
    task_manager = dispatcher.Enumerate_params_dict(task_thread=0, if_single_id_task=True, **param_pool_dict)

    num_jobs = len([v for v in itertools.product(*param_pool_dict.values())])
    exp_cmds = []

    for job_id in range(num_jobs):
        param_pool = task_manager.get_thread(ind=job_id)
        for hyper_params in param_pool:
            job_name = 'exp_{}'.format(job_id + start_id)
            for k in hyper_params.keys():
                params.dict[k] = hyper_params[k]

            params.dict['model_dir'] = os.path.join(exp_dir, exp_name, job_name)
            model_dir = params.dict['model_dir']

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # Write parameters in json file
            json_path = os.path.join(model_dir, 'params.json')
            params.save(json_path)

            # Launch training with this config
            cmd = 'rlaunch --cpu={} --memory={} --gpu={} -- python train.py --model_dir {}'.format(params.cpu, params.memory, params.gpu, model_dir)
            exp_cmds.append(cmd)

    tmux_ops.run_task(exp_cmds, task_name=exp_name, session_name=session_name)


def experiment():
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')   
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    if args.id == 1:
        # e.g. model and logs will be stored under 'experiment_learning_rate'
        name = "learning_rate"
        session_name = 'exp'  # tmux session name, need pre-create
        start_id = 0
        exp_name = 'experiment_{}'.format(name)
        param_pool_dict = collections.OrderedDict()
        param_pool_dict['learning_rate'] = [0.0005, 0.001]
    else:
        raise NotImplementedError

    launch_training_job(args.parent_dir, exp_name, session_name, param_pool_dict, params, start_id)


if __name__ == "__main__":
    experiment()
