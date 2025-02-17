#!/usr/bin/env python3
# -15745.28 eV
import numpy as np
from pyscf import gto, dft, scf
import logging
from typing import Tuple, Optional
import os
import platform

HARTREE_TO_EV = 27.211386245988  # Conversion factor from Hartree to eV

class DFTHamiltonian:
    """
    Class for performing DFT-based Hamiltonian calculations using PySCF.
    Uses double-zeta plus polarization (DZP) basis set.
    """
    
    def __init__(self, xyz_file: str, charge: int = 0, spin: int = 0, use_ecp: bool = False):
        """
        Initialize DFT calculator with system parameters.
        Optimized for Apple Silicon (M1/M2) processors.
        
        Args:
            xyz_file (str): Path to .xyz file containing molecular geometry
            charge (int): Total charge of the system
            spin (int): Spin multiplicity (2S + 1)
            use_ecp (bool): Whether to use ECP for Au atoms (default: False)
        """
        self.xyz_file = xyz_file
        self.charge = charge
        self.spin = spin
        self.use_ecp = use_ecp
        self.mol = None
        self.mf = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configure threading based on platform
        if platform.processor() == 'arm':
            # Optimize for Apple Silicon (M1/M2)
            self.logger.info("Configuring for Apple Silicon...")
            
            # Set memory limit
            os.environ['PYSCF_MAX_MEMORY'] = '32000'  # MB
            
            # Configure Apple Accelerate framework
            os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
            
            # PySCF native threading (avoid OpenMP)
            from pyscf.lib import num_threads
            num_threads(8)  # Set number of threads for native threading
            
            # Try to enable Metal acceleration
            try:
                import torch
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    os.environ['PYSCF_USE_METAL'] = '1'
                    self.logger.info("Metal acceleration enabled")
                    self.logger.info(f"PyTorch MPS device available: {torch.backends.mps.is_available()}")
                else:
                    self.logger.info("Metal acceleration not available")
            except ImportError:
                self.logger.info("PyTorch not available, using CPU only")
        else:
            # For non-Apple Silicon platforms
            self.logger.info("Configuring for standard CPU...")
            
            # Use standard threading configuration
            os.environ['OMP_NUM_THREADS'] = '8'
            os.environ['MKL_NUM_THREADS'] = '8'
            
            from pyscf.lib import num_threads
            num_threads(8)

    def read_xyz(self) -> list:
        """
        Read atomic coordinates from .xyz file.
        
        Returns:
            list: List of atomic symbols and coordinates
        """
        if not os.path.exists(self.xyz_file):
            raise FileNotFoundError(f"XYZ file not found: {self.xyz_file}")
            
        atoms = []
        try:
            with open(self.xyz_file, 'r') as f:
                n_atoms = int(f.readline())
                _ = f.readline()  # Skip comment line
                for _ in range(n_atoms):
                    line = f.readline().strip().split()
                    atom = [line[0], float(line[1]), float(line[2]), float(line[3])]
                    atoms.append(atom)
        except Exception as e:
            self.logger.error(f"Error reading XYZ file: {str(e)}")
            raise
            
        return atoms

    def setup_molecule(self) -> None:
        """
        Set up molecular system using PySCF's gto module.
        Uses DZP for non-Au atoms, and LANL2DZ with ECP for Au when ECP is enabled.
        """
        try:
            atoms = self.read_xyz()
            
            # Calculate total electrons
            total_electrons = 0
            basis_dict = {}
            
            # Set up basis sets and count electrons
            for atom in atoms:
                symbol = atom[0]
                atomic_number = gto.elements.charge(symbol)
                
                if symbol == 'Au' and self.use_ecp:
                    # For Au with ECP: use LANL2DZ basis and ECP
                    basis_dict[symbol] = 'lanl2dz'
                    total_electrons += 11  # 11 valence electrons with LANL2DZ ECP
                else:
                    # For all other atoms or Au without ECP: use DZP
                    basis_dict[symbol] = 'dzp'
                    total_electrons += atomic_number
            
            total_electrons -= self.charge
            default_spin = total_electrons % 2
            
            # Log configuration details
            if self.use_ecp:
                self.logger.info(f"System has {total_electrons} electrons (with LANL2DZ ECP for Au)")
                self.logger.info("Basis sets:")
                for atom, basis in basis_dict.items():
                    self.logger.info(f"  {atom}: {basis}")
            else:
                self.logger.info(f"System has {total_electrons} electrons (all-electron calculation)")
                self.logger.info("Using DZP basis for all atoms")
            
            self.logger.info(f"Using spin = {default_spin} (multiplicity = {default_spin + 1})")
            
            # Set up molecule
            mol_args = {
                'atom': atoms,
                'basis': basis_dict if self.use_ecp else 'dzp',
                'charge': self.charge,
                'spin': default_spin,
                'verbose': 3,
                'max_memory': 16000,
                'symmetry': True
            }
            
            # Add ECP for Au when requested
            if self.use_ecp:
                mol_args['ecp'] = {'Au': 'lanl2dz'}
                self.logger.info("Using LANL2DZ ECP for Au atoms")
            
            self.mol = gto.M(**mol_args)
            self.mol.build()
            
        except Exception as e:
            self.logger.error(f"Error setting up molecule: {str(e)}")
            raise

    def run_dft(self) -> None:
        """
        Perform high-accuracy DFT calculation optimized for Au-CO-Au system.
        Uses hybrid functional with dispersion correction.
        """
        try:
            self.mf = dft.RKS(self.mol)
            
            # Use B3LYP with empirical dispersion correction
            self.mf.xc = 'b3lyp-d3'
            
            # Higher accuracy grid for better integration
            self.mf.grids.level = 5  # Increased from 4
            self.mf.grids.prune = None  # No grid pruning for accuracy
            
            # Tighter convergence criteria
            self.mf.conv_tol = 1e-8  # Increased from 1e-6
            self.mf.conv_tol_grad = 1e-5  # Gradient convergence
            self.mf.max_cycle = 150  # Increased max cycles
            
            # SCF stability optimization
            self.mf.direct_scf = True
            self.mf.diis_space = 12
            self.mf.level_shift = 0.1  # Small level shift for stability
            
            # DIIS with damping for better convergence
            from pyscf.scf import DIIS
            self.mf = scf.addons.fast_newton(self.mf)
            self.mf.diis_start_cycle = 1
            self.mf.damp = 0.2  # Conservative damping
            
            # Run SCF with initial guess from superposition of atomic densities
            self.mf.init_guess = 'atom'
            self.logger.info("Starting SCF calculation with high accuracy settings...")
            self.mf.kernel()
            
            # Log converged energies
            energy_hartree = self.mf.e_tot
            energy_ev = energy_hartree * HARTREE_TO_EV
            self.logger.info(f"Converged total energy: {energy_hartree:.8f} Hartree")
            self.logger.info(f"Converged total energy: {energy_ev:.8f} eV")
            
            # Check SCF convergence and stability
            if not self.mf.converged:
                self.logger.warning("SCF did not converge! Trying with alternative scheme...")
                
                # Try with different convergence scheme
                self.mf.diis = DIIS()
                self.mf.level_shift = 0.25
                self.mf.damp = 0.5
                self.mf.kernel()
                
                if not self.mf.converged:
                    self.logger.warning("SCF still did not converge! Results may be unreliable.")
            
            # Perform stability analysis
            stability = self.mf.stability()[0]
            if stability > 1e-5:
                self.logger.warning(f"Wavefunction stability index: {stability:.2e}")
                self.logger.warning("The solution might not be the ground state")
            
        except Exception as e:
            self.logger.error(f"Error in DFT calculation: {str(e)}")
            raise

    def get_hamiltonian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate and return the Hamiltonian and overlap matrices.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Hamiltonian and overlap matrices
        """
        if self.mf is None:
            raise RuntimeError("Must run DFT calculation before getting Hamiltonian")
            
        try:
            # Get Fock matrix (Hamiltonian)
            fock = self.mf.get_fock()
            # Get overlap matrix
            overlap = self.mf.get_ovlp()
            
            return fock, overlap
            
        except Exception as e:
            self.logger.error(f"Error getting Hamiltonian: {str(e)}")
            raise

    def calculate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform complete DFT calculation workflow with M1 optimization.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Hamiltonian and overlap matrices
        """
        import time
        start_time = time.time()
        
        self.setup_molecule()
        self.run_dft()
        result = self.get_hamiltonian()
        
        end_time = time.time()
        self.logger.info(f"Calculation completed in {end_time - start_time:.2f} seconds")
        
        return result


def main():
    """
    Example usage of DFTHamiltonian class.
    """
    xyz_file = "molecule.xyz"
    
    try:
        # Example with all-electron calculation
        self.logger.info("Running all-electron calculation...")
        dft_calc = DFTHamiltonian(xyz_file, use_ecp=False)
        hamiltonian, overlap = dft_calc.calculate()
        np.save('hamiltonian_full.npy', hamiltonian)
        np.save('overlap_full.npy', overlap)
        print("All-electron calculation completed")
        
        # Example with ECP for Au
        self.logger.info("\nRunning calculation with ECP for Au...")
        dft_calc_ecp = DFTHamiltonian(xyz_file, use_ecp=True)
        hamiltonian_ecp, overlap_ecp = dft_calc_ecp.calculate()
        np.save('hamiltonian_ecp.npy', hamiltonian_ecp)
        np.save('overlap_ecp.npy', overlap_ecp)
        print("ECP calculation completed")
        
        print("\nCalculations completed successfully!")
        print(f"Matrix shapes: {hamiltonian.shape}")
        print("Results saved to:")
        print("- All-electron: hamiltonian_full.npy, overlap_full.npy")
        print("- With ECP: hamiltonian_ecp.npy, overlap_ecp.npy")
        
    except Exception as e:
        logging.error(f"Calculation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 