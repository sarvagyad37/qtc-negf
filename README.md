# Quantum Transport Calculator

A Python-based quantum transport calculator that combines DFT (Density Functional Theory) and NEGF (Non-Equilibrium Green's Function) methods to calculate electronic transport properties of molecular systems.

## Features

- DFT calculations using PySCF
- Quantum transport calculations using Green's function formalism
- Transmission function calculation
- Current-voltage characteristics
- Density matrix calculation
- Optimized for Apple Silicon (M1/M2)
- Multi-threaded calculations
- Comprehensive logging and result visualization

## Requirements

- Python 3.8 or later
- MacOS (optimized for Apple Silicon) or Linux
- 8GB+ RAM recommended

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sarvagyad37/qtc-ngef.git
cd qtc-ngef
```

2. Create and activate virtual environment:

```bash
python -m venv qtc-venv
source qtc-venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the calculator:

```bash
python run_transport.py scatteringco.xyz
```

