# Diversity Experiments
This repository contains the implementation of the following methods:

- L-BRDiv
- BRDiv
- LIPO


All methods share the same implementation. The only difference lies in the different weights associated to the optimised loss functions within each method.

# Important Hyperparameters
All the list of relevant hyperparameters can be found in ``teammate_generation_experiments.yaml`` in the **configs** folder. Among these hyperparameters, these are the important parameters that should be considered in hyperparameter tuning:

- ``lr``: Learning rate.
- ``tau``: Weights related to soft target value network change.
- ``tolerance_factor``: Tau for L-BRDiv
- ``sp-rew-weight``: Weights associated to maximising the self-play performance of populations.
- ``xp-loss-weights``: Weights associated to XP matrix-based diversity optimisation.
- ``entropy-regularizer-loss``: Weight associated to entropy.
- ``num-populations``

The other hyperparameters are less important and can be ignored for hyperparameter training.

# Running Our Proposed Approach

To run our proposed approach, run the code with ``--jsd-weights=0``, ``--with-lagrange=True``, and ``--with-lipo=False``.

# Running BRDiv

To run our proposed approach, run the code with ``--jsd-weights=0``, ``--with-lagrange=False``, and ``--with-lipo=False``.

# Running LIPO

To run our proposed approach, run the code with ``--jsd-weights=0``, ``--with-lagrange=False``, and ``--with-lipo=True``.

# Installation
Create a clean conda env (can be skipped if you want to install on top of an existing conda/virtualenv)
```bash
conda create -n <env name> python=3.11
```
First, install the necessary libraries to run the code.
```bash
pip3 install wandb
wandb login
<Type in shared wandb password>
pip3 install pathlib chardet
pip3 install gym==0.26.0
pip3 install hydra-core==1.3.2
pip3 install nashpy==0.0.21
pip3 install torch torchvision torchaudio
pip3 install moviepy
pip3 install pettingzoo
pip3 install agilerl
```

Install melting pot from our forked version
```bash
git submodule init
git submodule update
```

Afterwards, go to every folder in the ```envs``` folder and type the following command to install the environments:
```bash
pip install -e .
```

# Running Experiments

To run experiments, go to the MAACBasedOptim and run the following command:
```bash
python run.py -cn teammate_generation_experiments --multirun hydra=hydra_slurm ++run.seed=1,2,3,4,5
```
This will run experiments with 5 seeds.

# Visualizing Results
To visualize results, use the codes in the ```Visualization``` folder. Then, tag the experiments that will be reported using the same tags in wandb.
To visualize AHT results, use the following commands:
```bash
python wan_script.py
```

Meanwhile to visualize the change in Lagrange multipliers, use the following commands:
```bash
python wan_script_lagrange.py
```
