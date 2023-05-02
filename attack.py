'''
    Attack specified model with specified adversarial attack method.

    Output: Saves as json file a list over all attacked data sample, with each entry being
            a dict with keys: 
                - 'text'
                - 'pred_label'
                - 'label'
                - 'att_text'
                - 'att_pred_label'
'''

import sys
import os
import argparse
import json

from src.models.model_selector import select_model
from src.data.data_selector import select_data
from src.attack.attacker import Attacker

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_path', type=str, required=True, help='where to save .json file')
    commandLineParser.add_argument('--model_path', type=str, required=True, help='trained model path')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. roberta-base')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. sst')
    commandLineParser.add_argument('--num_classes', type=int, default=2, help="Specify number of classes in data")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--data_part', type=str, default='test', choices=['train', 'val', 'test'], help='select data part to attack')
    commandLineParser.add_argument('--attack_method', type=str, default='textfooler', help="Specify attack method")
    commandLineParser.add_argument('--batch', action='store_true', help='apply attack to batch of the data')
    commandLineParser.add_argument('--start', type=int, required=False, default=0, help='start of batch')
    commandLineParser.add_argument('--end', type=int, required=False, default=1000, help='end of batch')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load model
    model = select_model(args.model_name, model_path=args.model_path, num_labels=args.num_classes)
    # model.to(device)
   
    # Load the relevant data
    if args.data_part == 'test':
        data = select_data(args, train=False)
    elif args.data_part == 'val':
        data, _ = select_data(args, train=True)
    else:
        _, data = select_data(args, train=True)
    if args.batch:
        data = data[args.start:args.end]


    # Attack all samples
    att_data = Attacker.attack_all(data, model, method=args.attack_method)

    # Save
    with open(args.out_path, 'w') as f:
        json.dump(att_data, f)