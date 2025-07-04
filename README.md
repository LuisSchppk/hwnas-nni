# Project Setup Guide

This repository depends on [MNSIM-2.0](https://github.com/thu-nics/MNSIM-2.0/tree/master) and other Python packages. Follow the instructions below to set up your environment.

---

## Step 1: Install Python Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
CUDA Version 11.8 was used.
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

### Option: `pyproject.toml`

Create a file `MNSIM-2.0/pyproject.toml` with the following contents:

```python
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "mnsim"
version = "2.0.0"
description = "Packaging of third party MNSIM-2.0 for easy installation"
dependencies = []

[tool.setuptools.packages.find]
where = ["."]
```

> This makes MNSIM-2.0 installable as a Python package.

---

## Step 4: Install MNSIM in Editable Mode

From the `MNSIM` directory:

```bash
cd MNSIM-2.0
pip install -e .
```

This installs MNSIM in **editable mode**, so any changes to the source code take effect immediately without reinstalling.

---

## Step 5:

## Done
