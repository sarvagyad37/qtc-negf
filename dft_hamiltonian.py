#!/usr/bin/env python3
# -15745.28 eV
import numpy as np
from pyscf import gto, dft, scf
import logging
from typing import Tuple, Optional
import os
import platform

class DFTHamiltonian:
    """
    Class for performing DFT-based Hamiltonian calculations using PySCF.
    Uses double-zeta plus polarization (DZP) basis set.
    """
    
    def __init__(self, xyz_file: str, charge: int = 0, spin: int = 0):
        """
        Initialize DFT calculator with system parameters.
        Optimized for Apple Silicon (M1/M2) processors.
        
        Args:
            xyz_file (str): Path to .xyz file containing molecular geometry
            charge (int): Total charge of the system
            spin (int): Spin multiplicity (2S + 1)
        """
        self.xyz_file = xyz_file
        self.charge = charge
        self.spin = spin
        self.mol = None
        self.mf = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configure threading for Apple Silicon
        if platform.processor() == 'arm':
            # Optimize for M1/M2
            os.environ['PYSCF_MAX_MEMORY'] = '32000'  # MB
            os.environ['VECLIB_MAXIMUM_THREADS'] = '8' # Optimize Accelerate framework
            
            # Remove OpenMP settings as it's not available
            if 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']
            if 'MKL_NUM_THREADS' in os.environ:
                del os.environ['MKL_NUM_THREADS']
            
            # Use Apple's native threading
            from pyscf.lib import num_threads
            num_threads(8)  # Set number of threads for native threading
            
            self.logger.info("Configured for Apple Silicon native threading")
            
            # Try to enable Metal acceleration if available
            try:
                import torch
                self.logger.info(f"PyTorch version: {torch.__version__}")
                self.logger.info(f"MPS available: {torch.backends.mps.is_available()}")
                self.logger.info(f"MPS built: {torch.backends.mps.is_built()}")
                
                if torch.backends.mps.is_available():
                    os.environ['PYSCF_USE_METAL'] = '1'
                    self.logger.info("Metal acceleration enabled")
                else:
                    self.logger.warning("MPS is not available. Metal acceleration disabled")
                    if not torch.backends.mps.is_built():
                        self.logger.warning("PyTorch was not built with MPS support")
            except ImportError as e:
                self.logger.warning(f"PyTorch import failed: {str(e)}")
                self.logger.info("PyTorch not available, Metal acceleration disabled")

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
        Optimized for Au metal with relativistic effects.
        Uses LANL2DZ with ECP for Au and DZP for other atoms.
        """
        try:
            atoms = self.read_xyz()
            
            # Calculate total electrons and set up basis
            total_electrons = 0
            basis_dict = {}
            
            # Special basis set handling
            for atom in atoms:
                symbol = atom[0]
                atomic_number = gto.elements.charge(symbol)
                
                if symbol == 'Au':
                    # Special treatment for Au
                    basis_dict[symbol] = 'lanl2dz'
                    # Count only valence electrons for Au (11 valence electrons with LANL2DZ ECP)
                    total_electrons += 11
                else:
                    basis_dict[symbol] = 'dzp'
                    total_electrons += atomic_number
            
            total_electrons -= self.charge
            default_spin = total_electrons % 2
            
            self.logger.info(f"System has {total_electrons} electrons (with ECP for Au)")
            self.logger.info(f"Using spin = {default_spin} (multiplicity = {default_spin + 1})")
            self.logger.info("Basis sets:")
            for atom, basis in basis_dict.items():
                self.logger.info(f"  {atom}: {basis}")
            
            # Define ECP for Au
            ecp_dict = {'Au': 'lanl2dz'}
            
            self.mol = gto.M(
                atom=atoms,
                basis=basis_dict,
                charge=self.charge,
                spin=default_spin,
                verbose=3,
                max_memory=8000,
                symmetry=True,
                ecp=ecp_dict,  # Specific ECP for Au
            )
            self.mol.build()
            
        except Exception as e:
            self.logger.error(f"Error setting up molecule: {str(e)}")
            raise

    def run_dft(self) -> None:
        """
        Perform DFT calculation optimized for Au metal systems and Apple Silicon.
        Uses B3LYP functional with relativistic corrections.
        """
        try:
            self.mf = dft.RKS(self.mol)
            self.mf.xc = 'b3lyp'
            
            # Performance optimizations for Au systems on M1
            self.mf.grids.level = 4
            self.mf.grids.prune = None
            self.mf.conv_tol = 1e-6
            self.mf.max_cycle = 100
            self.mf.direct_scf = True
            self.mf.diis_space = 12
            
            # Use DIIS convergence acceleration with damping
            from pyscf.scf import DIIS
            self.mf = scf.addons.fast_newton(self.mf)
            self.mf.diis_start_cycle = 1
            self.mf.damp = 0.3
            
            # Enable density fitting with thread optimization
            from pyscf import df
            self.mf = self.mf.density_fit()
            self.mf.with_df.max_memory = 32000  # MB
            
            # Additional M1 optimizations
            self.mf._gen_integrals_cache = True  # Cache integrals
            self.mf.init_guess = 'atom'          # Better initial guess
            
            self.logger.info("Starting SCF calculation with M1 optimizations...")
            self.mf.kernel()
            
            if not self.mf.converged:
                self.logger.warning("SCF did not converge! Trying with alternative scheme...")
                self.mf.diis = DIIS()
                self.mf.level_shift = 0.25
                self.mf.kernel()
                
                if not self.mf.converged:
                    self.logger.warning("SCF still did not converge! Results may be unreliable.")
            
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
    # Example usage
    xyz_file = "scatteringco.xyz"  # Replace with your xyz file
    
    try:
        dft_calc = DFTHamiltonian(xyz_file)
        hamiltonian, overlap = dft_calc.calculate()
        
        print("Calculation completed successfully!")
        print(f"Hamiltonian matrix shape: {hamiltonian.shape}")
        print(f"Overlap matrix shape: {overlap.shape}")

        # put hamiltonian and overlap into a file
        # Save matrices to numpy files
        np.save('hamiltonian.npy', hamiltonian)
        np.save('overlap.npy', overlap)
        print(f"Saved Hamiltonian and overlap matrices to hamiltonian.npy and overlap.npy")

    except Exception as e:
        logging.error(f"Calculation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 