# A Temporary Replication of *Learning a Generalized Physical Face Model From Data*

This repository is a replication of paper [Learning a Generalized Physical Face Model From Data](https://dl.acm.org/doi/abs/10.1145/3658189).

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
mkdir libs
cd libs
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

### Models

You can get the FLAME model from [FLAME website](https://flame.is.tue.mpg.de/).

After downloading the model, unzip it and place them into the `./model/flame` folder.

You should also download embeddings from [RingNet project](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model). Copy it inside the flame folder as well.

For SCULPTOR, you can get the model from [SCULPTOR repository](https://github.com/sculptor2022/sculptor). Place `paradict.npy` into the `./model/sculptor` folder.

## Usage

Not yet finished, to be implemented in the future.
