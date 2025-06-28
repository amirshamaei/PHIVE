import argparse
import json

from engine import Engine


def inference(args):
    json_file_path = args.json_file

    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())

    for run in contents['runs']:
        if args.ens_id != None:
            run["ens_id"] = args.ens_id
        if args.cuda != None:
            run["gpu"] = args.cuda

        engine = Engine(run)
        engine.inference(args.test_path,args.mask_path,args.affine_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=False, type=str, help="Path to JSON file",default="runs/for_ice.json")
    parser.add_argument("--ens_id", required=False, type=int, help="Ensemble ID",default=0)
    parser.add_argument("--cuda", required=False, type=str, help="which cuda", default='cpu')
    parser.add_argument("--test_path", nargs='+', required=False, type=str, help="which dataset",default=['data/new_data/MS_Better_Patient.npy'])
    parser.add_argument("--mask_path", nargs='+', required=False, type=str, help="which mask",default=['data/new_data/MS_Better_Patient_mask.npy'])
    parser.add_argument("--affine_path", nargs='+', required=False, type=str, help="which affine",default=['data/new_data/csi_template_mod.nii'])
    args = parser.parse_args()

    inference(args)
