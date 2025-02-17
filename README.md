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
- Optional ECP (Effective Core Potential) for Au atoms

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

### Calculation Workflow

The calculator follows this workflow for quantum transport calculations:

1. **DFT Calculation**
   - Reads molecular geometry from XYZ file
   - Sets up basis sets (DZP for all atoms, optional ECP for Au)
   - Performs SCF calculation using B3LYP-D3 functional
   - Generates Hamiltonian and overlap matrices

2. **Green's Function Setup**
   - Constructs self-energy matrices for electrodes
   - Handles coupling between molecule and electrodes
   - Sets up energy grid for transmission calculation

3. **Transmission Calculation**
   - Calculates retarded Green's function G(E)
   - Computes transmission function T(E) = Tr[Γ₁GΓ₂G†]
   - Uses parallel processing for energy points

4. **Current Calculation**
   - Integrates transmission over energy window
   - Includes Fermi function for temperature effects
   - Calculates I-V characteristics

5. **Results Processing**
   - Saves all matrices and calculation results
   - Generates transmission and I-V plots
   - Creates detailed calculation log

### Key Equations

- **Green's Function**: G(E) = [(E + iη)S - H - Σ₁ - Σ₂]⁻¹
- **Transmission**: T(E) = Tr[Γ₁GΓ₂G†]
- **Current**: I(V) = (2e/h)∫T(E)[f₁(E) - f₂(E)]dE

### Parameters

- Energy range: [-5.0, 5.0] eV
- Temperature: 300K
- Voltage range: [-2.0, 2.0] V
- Integration points: 1000 (energy), 401 (voltage)

### Basic Usage

Run the calculator with all-electron calculation (no ECP):

```bash
python run_transport.py molecule.xyz
```

### Using ECP for Au atoms

To use Effective Core Potential (ECP) for Au atoms:

```bash
python run_transport.py molecule.xyz --use-ecp
```

### Specifying Output Directory

```bash
python run_transport.py molecule.xyz -o results_directory --use-ecp
```

## Output Structure

The program creates a timestamped output directory containing:
```
output_molecule_YYYYMMDD_HHMMSS/
├── calculation.log          # Detailed calculation log
├── molecule.xyz            # Copy of input geometry
├── hamiltonian.npy         # DFT Hamiltonian matrix
├── overlap.npy             # Overlap matrix
├── transmission.npy        # Transmission function data
├── current.npy             # Current-voltage data
├── density_matrix.npy      # Density matrix
├── results.json           # Calculation parameters and metadata
└── transport_results.png   # Plots of transmission and I-V characteristics
```

## Calculation Options

### All-Electron Calculation
- Uses DZP (Double Zeta plus Polarization) basis set for all atoms
- Treats all electrons explicitly
- More computationally intensive, especially for Au atoms

### ECP Calculation (--use-ecp)
- Uses DZP basis set with LANL2DZ ECP for Au atoms
- Treats only valence electrons for Au (11 electrons)
- More efficient for systems with Au atoms
- Includes relativistic effects through ECP

## Performance Notes

- Utilizes multiple CPU cores for parallel calculations
- Optimized for Apple Silicon with Metal acceleration (if PyTorch is available)
- Memory usage scales with system size
- ECP calculations are generally faster for systems with Au atoms
- Calculation time depends on:
  - System size
  - Basis set choice (all-electron vs. ECP)
  - Energy grid resolution
  - Voltage points
  - Available computational resources

## Troubleshooting

1. Memory Issues:
   - Use ECP for large systems with Au atoms
   - Reduce energy_points or voltage_points in TransportParameters
   - Increase chunk_size for larger systems
   - Monitor memory usage during calculation

2. Convergence Issues:
   - Check calculation.log for warnings
   - Try switching between ECP and all-electron calculations
   - Adjust broadening parameter (eta)
   - Verify input structure

## Citation

If you use this code in your research, please cite:
[Add your citation information]

## License

[Add your license information]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

