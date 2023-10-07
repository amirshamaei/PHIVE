import argparse
import json
import sys

import engine as eng

def main(args):
    file = open('training_report.csv', 'w+')
    json_file_path = args.json_file

    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())

    for run in contents['runs']:
        if args.ens_id != None:
            run["ens_id"] = args.ens_id
        if args.cuda != None:
            run["gpu"] = args.cuda
        if run["version"] in ["1/"]:
            engine = eng . Engine(run)
            if args.mode == 'train':
                engine.dotrain(run["ens_id"])
            if args.mode == 'test':
                engine.dotest(args.test_path,args.mask_path,args.affine_path)
    file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=False, type=str, help="Path to JSON file",default="runs/exp46.json")
    parser.add_argument("--ens_id", required=False, type=int, help="Ensemble ID",default=None)
    parser.add_argument("--mode", required=False, type=str, help="train or test",default='test')
    parser.add_argument("--cuda", required=False, type=str, help="which cuda", default='cuda:0')
    parser.add_argument("--test_path", nargs='+', required=False, type=str, help="which dataset")
    parser.add_argument("--mask_path", nargs='+', required=False, type=str, help="which mask")
    parser.add_argument("--affine_path", nargs='+', required=False, type=str, help="which affine")
    args = parser.parse_args()

    main(args)