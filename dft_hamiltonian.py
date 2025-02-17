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
        Uses DZP (Double Zeta plus Polarization) basis set for all atoms.
        Optionally uses ECP for Au atoms.
        """
        try:
            atoms = self.read_xyz()
            
            # Calculate total electrons
            total_electrons = 0
            
            # Count electrons based on ECP usage
            for atom in atoms:
                symbol = atom[0]
                atomic_number = gto.elements.charge(symbol)
                
                if symbol == 'Au' and self.use_ecp:
                    total_electrons += 11  # 11 valence electrons with ECP
                else:
                    total_electrons += atomic_number  # All electrons
            
            total_electrons -= self.charge
            default_spin = total_electrons % 2
            
            # Log electron count and basis information
            if self.use_ecp:
                self.logger.info(f"System has {total_electrons} electrons (with ECP for Au)")
                self.logger.info("Using DZP basis with LANL2DZ ECP for Au")
            else:
                self.logger.info(f"System has {total_electrons} electrons (all-electron calculation)")
                self.logger.info("Using DZP basis for all atoms (including all electrons for Au)")
            
            self.logger.info(f"Using spin = {default_spin} (multiplicity = {default_spin + 1})")
            
            # Set up molecule with or without ECP
            mol_args = {
                'atom': atoms,
                'basis': 'dzp',
                'charge': self.charge,
                'spin': default_spin,
                'verbose': 3,
                'max_memory': 16000,
                'symmetry': True
            }
            
            # Add ECP if requested
            if self.use_ecp:
                mol_args['ecp'] = {'Au': 'lanl2dz'}
            
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
    xyz_file = "scatteringco.xyz"
    
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