import os
import sys
import logging
import argparse
from mteb import MTEB
from InstructorEmbedding import INSTRUCTOR
import random, numpy as np, torch 

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None,type=str)
    parser.add_argument('--output_dir', default=None,type=str)
    parser.add_argument('--task_name', default=None,type=str)
    parser.add_argument('--cache_dir', default=None,type=str)
    parser.add_argument('--result_file', default=None,type=str)
    parser.add_argument('--prompt', default=None,type=str)
    parser.add_argument('--split', default='test',type=str)
    parser.add_argument('--batch_size', default=128,type=int)
    parser.add_argument('--samples_per_label', default=16,type=int)
    parser.add_argument('--n_experiments', default=1,type=int)
    parser.add_argument('--seed', default=25,type=int)
    parser.add_argument('--checkRobustness', default=True,type=bool)
    parser.add_argument('--robustnessSamples', default=5,type=int)
    parser.add_argument('--checkMultiLinguality', default=True,type=bool)
    
    args = parser.parse_args()

    if not args.result_file.endswith('.txt') and not os.path.isdir(args.result_file):
        os.makedirs(args.result_file,exist_ok=True)

    #seeds = [30,42,89,120,26,19]
    # from tqdm import tqdm
    # from functools import partialmethod
    #
    # tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    seed = args.seed
    set_seed(seed)
    model = INSTRUCTOR(args.model_name,cache_folder=args.cache_dir)
    evaluation = MTEB(tasks=[args.task_name],task_langs=["en"])
    evaluation.run(model, output_folder=args.output_dir, eval_splits=[args.split],args=args,)

    print("--DONE--")
