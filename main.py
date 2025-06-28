"""
MRSI_fit - Main Training and Testing Script
===========================================

This script serves as the main entry point for training and testing the MRSI (Magnetic Resonance 
Spectroscopic Imaging) fitting model. It provides a command-line interface for running experiments
defined in JSON configuration files.

The script supports:
- Training deep learning models for MRSI spectral fitting
- Testing trained models on new data
- Ensemble training with multiple model instances
- GPU acceleration support
- Flexible configuration through JSON files

Author: [Your Name]
Date: [Date]
"""

import argparse
import json
import sys

import engine as eng

def main(args):
    """
    Main function that orchestrates the training or testing process.
    
    Args:
        args: Parsed command line arguments containing configuration parameters
    """
    # Initialize training report file
    file = open('training_report.csv', 'w+')
    json_file_path = args.json_file

    # Load experiment configuration from JSON file
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())

    # Process each run defined in the configuration
    for run in contents['runs']:
        # Override ensemble ID if specified
        if args.ens_id != None:
            run["ens_id"] = args.ens_id
        # Override CUDA device if specified
        if args.cuda != None:
            run["gpu"] = args.cuda
        # Override conditional test parameter if specified
        if args.cond_test != None:
            run["cond_test"] = args.cond_test
        
        # Process runs with version "1/"
        if run["version"] in ["1/"]:
            # Initialize the training engine with run parameters
            engine = eng.Engine(run)
            
            # Execute training or testing based on mode
            if args.mode == 'train':
                engine.dotrain(run["ens_id"])
            if args.mode == 'test':
                engine.dotest(args.test_path, args.mask_path, args.affine_path)
    
    file.close()

if __name__ == '__main__':
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="MRSI_fit: Deep Learning-based MRSI Spectral Fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model using default configuration
  python main.py --mode train
  
  # Train with specific ensemble ID and GPU
  python main.py --mode train --ens_id 1 --cuda cuda:1
  
  # Test a trained model
  python main.py --mode test --test_path data/test.npy --mask_path data/mask.npy
        """
    )
    
    # Define command line arguments
    parser.add_argument("--json_file", required=False, type=str, 
                       help="Path to JSON configuration file", 
                       default="runs/exp101.json")
    parser.add_argument("--ens_id", required=False, type=int, 
                       help="Ensemble ID for multi-model training", 
                       default=0)
    parser.add_argument("--cond_test", required=False, type=int, 
                       help="Conditional test parameter", 
                       default=0)
    parser.add_argument("--mode", required=False, type=str, 
                       help="Operation mode: 'train' or 'test'", 
                       default='train')
    parser.add_argument("--cuda", required=False, type=str, 
                       help="CUDA device specification (e.g., 'cuda:0')", 
                       default='cuda:0')
    parser.add_argument("--test_path", nargs='+', required=False, type=str, 
                       help="Path(s) to test data file(s)")
    parser.add_argument("--mask_path", nargs='+', required=False, type=str, 
                       help="Path(s) to mask file(s)")
    parser.add_argument("--affine_path", nargs='+', required=False, type=str, 
                       help="Path(s) to affine transformation file(s)")
    
    args = parser.parse_args()

    main(args)