import torch
import torch.nn as nn
from statistics import mean, stdev
import sys
import os
import argparse
import logging

from src.tools.tools import get_default_device
from src.models.ensemble import Ensemble
from src.models.model_selector import select_model
from src.data.data_selector import select_data
from src.training.batch_trainer import BatchTrainer as Trainer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path_base', type=str, required=True, help='e.g. experiments/trained_models/my_model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. rt')
    commandLineParser.add_argument('--bs', type=int, default=8, help="Specify batch size")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--num_seeds', type=int, default=1, help="Specify number of seeds for model to load")
    commandLineParser.add_argument('--num_classes', type=int, default=2, help="Specify number of classes")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the test data
    data = select_data(args, train=False)
    dl = Trainer.prep_dl(select_model(args.model_name), data, bs=args.bs, shuffle=False)

    # Load models
    model_paths = [f'{args.model_path_base}{i}.th' for i in range(1, args.num_seeds+1)]
    ens_model = Ensemble(args.model_name, model_paths, device, num_labels=args.num_classes)

    # Evaluate
    criterion = nn.CrossEntropyLoss().to(device)
    accs = ens_model.eval(dl, criterion, device)

    if len(model_paths) > 1:
        acc_mean = mean(accs)
        acc_std = stdev(accs)
        print(f'{len(model_paths)} models\nOverall {acc_mean:.3f}+-{acc_std:.3f}')
    else:
        print(f'Accuracy Single Seed {accs}')