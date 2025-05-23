# A Temporary Replication of *Learning a Generalized Physical Face Model From Data*

This repository if a replication of paper [Learning a Generalized Physical Face Model From Data](https://dl.acm.org/doi/abs/10.1145/3658189).

## Setup

### uv

You can install uv directly from PyPI:
```bash
pip install uv
```

Or, you can go to [the official uv repository](https://pypi.org/project/uv/) to learn more about how to install and use it.

To install related libraries, you can use `sync`:
```bash
uv sync
```

### PyTorch

You can visit [PyTorch website](https://pytorch.org/get-started/locally/) to check how to install PyTorch.

### FLAME

Clone [the official FLAME_PyTorch repository](https://github.com/soubhiksanyal/FLAME_PyTorch):
```bash
mkdir dependencies
cd dependencies
git clone https://github.com/soubhiksanyal/FLAME_PyTorch.git
```

Install `FLAME_PyTorch`:
```bash
cd FLAME_PyTorch
uv pip install .
```

Be aware of [chumpy]() here, you should install a newer version manually to avoid errors:
```bash
uv pip install git+https://github.com/mattloper/chumpy.git --no-build-isolation
```

### psbody-mesh

Install the `Boost` Library:
```bash
sudo apt-get install libboost-dev
```

Clone the [repository](https://github.com/MPI-IS/mesh):
```bash
cd dependencies
git clone https://github.com/MPI-IS/mesh.git
```

And after replacing `--install-option` with `--config-settings` in line 7 of `Makefile`, install `psbody-mesh`:
```bash
cd mesh
uv pip install .
```

### Models

You can get the FLAME model from [FLAME website](https://flame.is.tue.mpg.de/).

After downloading the model, unzip it and place them into the `./model` folder, and rename it as `generic_model.pkl`.

You should also download embeddings from [RingNet project](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model). Copy it inside the model folder as well.

Now you should be able to run the script.

## Usage

Not yet finished, to be implemented in the future.
