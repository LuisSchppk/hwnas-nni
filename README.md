# Project Setup Guide

This repository depends on [MNSIM-2.0](https://github.com/thu-nics/MNSIM-2.0/tree/master) and other Python packages. Follow the instructions below to set up your environment.

---

## Step 1: Install Python Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Step 2: Download MNSIM-2.0

Clone the MNSIM-2.0 repository:

```bash
git clone https://github.com/thu-nics/MNSIM-2.0.git
```

> Ensure you are in the correct working directory before cloning.

---

##️ Step 3: Make MNSIM Installable

Add a `pyproject.toml` or `setup.py` to the root of the `MNSIM` directory.

### Option A: `setup.py`

Create a file `MNSIM/setup.py` with the following contents:

```python
from setuptools import setup, find_packages

setup(
    name="mnsim",
    version="2.0",
    packages=find_packages(),
    install_requires=[],
)
```

> This makes MNSIM installable as a Python package.

---

## Step 4: Install MNSIM in Editable Mode

From the `MNSIM` directory:

```bash
cd MNSIM
pip install -e .
```

This installs MNSIM in **editable mode**, so any changes to the source code take effect immediately without reinstalling.

---

## Done
