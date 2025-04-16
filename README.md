# Piecewise Constant Spectral Graph Neural Network

This repository is the official implementation of the model in the following paper:

    @article{piecon,
      author       = {A. Martirosyan, Vahan, and H. Giraldo, Jhony, and D. Malliaros, Fragkiskos},
      title        = {Piecewise Constant Spectral Graph Neural Network},
      journal      = {Transactions on Machine Learning Research},
      year         = {2025}
    }

## Reproduce Our Results

You should first run preprocess_data.py to do eigenvalue decomposition and save the eigenvalues and eigenvectors in generated files for each dataset.

    python preprocess_data.py

To reproduce the results of PieCoN for each dataset:

    python piecon.py --dataset $dataset --runs 10

To run with a custom seed:

    python piecon.py --dataset $dataset --seed 523

The hyperparameters for each dataset are available in configs/piecon_config.yaml
    
