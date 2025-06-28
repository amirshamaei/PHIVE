"""
Basic Usage Example for MRSI_fit
================================

This example demonstrates how to use the MRSI_fit framework for training
and testing MRSI spectral fitting models.
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Engine
from Model import Encoder_Model


def create_basic_config():
    """
    Create a basic configuration for MRSI fitting.
    
    Returns:
        dict: Configuration dictionary
    """
    config = {
        "description": "Basic MRSI fitting example",
        "parent_root": "experiments/",
        "child_root": "basic_example/",
        "version": "1/",
        "data_dir": "data/training_data.npy",
        "data_dir_ny": "data/training_data_processed.npy",
        "basis_dir": "data/basis_set.mat",
        "test_data_root": "test_data",
        "max_epoch": 50,
        "batchsize": 32,
        "numOfSample": 1000,
        "t_step": 0.0005,
        "trnfreq": 123.2,
        "nauis": 1,
        "save": True,
        "tr": 0.004,
        "betas": [0.001],
        "depths": [4],
        "ens": 1,
        "met_name": ["NAA", "Cr", "Cho", "Ins"],
        "wr": [False, 0],
        "data_name": "data",
        "numOfSig": 4,
        "sigLen": 2048,
        "truncSigLen": 1024,
        "MM": False,
        "MM_f": [],
        "MM_d": [],
        "MM_a": [],
        "MM_plot": False,
        "pre_plot": False,
        "basis_need_shift": [False, 0],
        "aug_params": [0.1, 0.1, 0.1, 0.1],
        "tr_prc": 0.8,
        "in_shape": "complex",
        "enc_type": "conv",
        "banorm": True,
        "reg_wei": 0.001,
        "data_conj": False,
        "test_nos": 100,
        "quality_filt": False,
        "test_name": "test_data",
        "beta_step": 0.001,
        "MM_type": "param",
        "MM_dir": None,
        "MM_constr": False,
        "comp_freq": False,
        "sim_order": ["test", "simulations/"],
        "gpu": "cuda:0",
        "num_of_workers": 0,
        "domain": "time",
        "zero_fill": [False, 1024],
        "fbound": [4.0, 1.0],
        "spline": False,
        "numofsplines": 10,
        "dropout": 0.1,
        "decode": False,
        "freeze_enc": False,
        "freeze_dec": False,
        "MM_model": "lorntz",
        "MM_fd_constr": False,
        "cond_max": 1,
        "simulated": False,
        "test_load": False,
        "reload_test": True
    }
    return config


def example_training():
    """
    Example of training an MRSI fitting model.
    """
    print("=== MRSI_fit Training Example ===")
    
    # Create configuration
    config = create_basic_config()
    
    # Save configuration to file
    os.makedirs("runs", exist_ok=True)
    with open("runs/example_config.json", "w") as f:
        json.dump({"runs": [config]}, f, indent=2)
    
    print("Configuration saved to runs/example_config.json")
    
    # Initialize engine
    print("Initializing training engine...")
    engine = Engine(config)
    
    # Prepare data (this would normally load your actual data)
    print("Preparing data...")
    # Note: In a real scenario, you would have actual data files
    # engine.data_prep()
    
    # Train model
    print("Starting training...")
    # engine.dotrain(ens_id=0)
    
    print("Training example completed!")
    print("Note: This is a demonstration. Actual training requires data files.")


def example_testing():
    """
    Example of testing an MRSI fitting model.
    """
    print("=== MRSI_fit Testing Example ===")
    
    # Create configuration
    config = create_basic_config()
    
    # Initialize engine
    print("Initializing testing engine...")
    engine = Engine(config)
    
    # Example test paths (these would be your actual data files)
    test_path = "data/test_data.npy"
    mask_path = "data/mask.npy"
    affine_path = "data/affine.nii"
    
    print(f"Test data path: {test_path}")
    print(f"Mask path: {mask_path}")
    print(f"Affine path: {affine_path}")
    
    # Test model
    print("Starting testing...")
    # engine.dotest(test_path, mask_path, affine_path)
    
    print("Testing example completed!")
    print("Note: This is a demonstration. Actual testing requires trained models and data files.")


def example_model_creation():
    """
    Example of creating an MRSI fitting model.
    """
    print("=== MRSI_fit Model Creation Example ===")
    
    # Create a simple parameter object for demonstration
    class SimpleParams:
        def __init__(self):
            self.t = [[0.0005 * i for i in range(1024)]]
            self.t_step = 0.0005
            self.org_truncSigLen = 1024
            self.basisset = np.random.randn(1024, 4).astype('complex64')
            self.numOfSig = 4
            self.MM = False
            self.MM_constr = False
            self.parameters = {
                'spline': False,
                'numofsplines': 10,
                'dropout': 0.1,
                'decode': False,
                'freeze_enc': False,
                'freeze_dec': False,
                'MM_model': 'lorntz',
                'domain': 'time',
                'zero_fill': [False, 1024],
                'fbound': [4.0, 1.0],
                'cond_max': 1,
                'gpu': 'cuda:0'
            }
            self.in_shape = 'complex'
            self.truncSigLen = 1024
            self.trnfreq = 123.2
    
    # Create model parameters
    import numpy as np
    params = SimpleParams()
    
    # Create model
    print("Creating MRSI fitting model...")
    model = Encoder_Model(
        depth=4,
        beta=0.001,
        tr_wei=1.0,
        i=0,
        param=params
    )
    
    print(f"Model created successfully!")
    print(f"Model type: {type(model)}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model


if __name__ == "__main__":
    print("MRSI_fit Basic Usage Examples")
    print("=" * 40)
    
    # Run examples
    example_model_creation()
    print()
    example_training()
    print()
    example_testing()
    
    print("\nAll examples completed!")
    print("For actual usage, please:")
    print("1. Prepare your data files")
    print("2. Configure your experiment parameters")
    print("3. Run training with: python main.py --mode train --json_file runs/your_config.json")
    print("4. Run testing with: python main.py --mode test --test_path your_data.npy") 