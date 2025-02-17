#!/usr/bin/env python3

import numpy as np
from scipy import linalg
from scipy.integrate import simpson
import logging
from typing import Tuple, Optional, List
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os

# Physical constants
KB = 8.617333262e-5  # Boltzmann constant in eV/K
HBAR = 6.582119569e-16  # Reduced Planck constant in eV⋅s
E_CHARGE = 1.602176634e-19  # Elementary charge in Coulombs
H_PLANCK = 4.135667696e-15  # Planck constant in eV⋅s

@dataclass
class TransportParameters:
    """Parameters for transport calculations"""
    energy_range: Tuple[float, float] = (-5.0, 5.0)
    energy_points: int = 1000
    temperature: float = 300.0  # Kelvin
    eta: float = 1e-3  # Broadening parameter in eV
    voltage_range: Tuple[float, float] = (-2.0, 2.0)
    voltage_points: int = 401
    fermi_energy: float = 0.0  # Fermi energy in eV

class QuantumTransport:
    """
    Quantum transport calculations using Green's function formalism.
    """
    def __init__(self, params: TransportParameters = TransportParameters()):
        self.params = params
        self.hamiltonian = None
        self.overlap = None
        self.energy_grid = None
        self.voltage_grid = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize results storage
        self.transmission = None
        self.current = None
        self.density_matrix = None
        
        # Add caching for frequently used calculations
        self._green_cache = {}
        self._self_energy_cache = {}
        self.chunk_size = 50  # Process energy points in chunks
        
        # Enable numpy multithreading
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["MKL_NUM_THREADS"] = "8"
        np.set_printoptions(precision=6, suppress=True)
        
        # Increase precision and set numerical parameters
        self.numerical_params = {
            'matrix_precision': np.complex128,
            'real_precision': np.float64,
            'cache_decimals': 8,  # Increased from 6
            'svd_cutoff': 1e-12,  # For matrix inversion stability
            'min_determinant': 1e-30,  # For matrix condition checking
            'integration_rtol': 1e-8,  # Relative tolerance for integration
            'integration_atol': 1e-12  # Absolute tolerance for integration
        }
        
        # Configure numpy for high precision
        np.set_printoptions(precision=8, suppress=True)
        
    def load_matrices(self, hamiltonian_file: str = 'hamiltonian.npy', 
                     overlap_file: str = 'overlap.npy') -> None:
        """Load Hamiltonian and overlap matrices from files"""
        try:
            self.hamiltonian = np.load(hamiltonian_file)
            self.overlap = np.load(overlap_file)
            self.logger.info(f"Loaded matrices: H shape {self.hamiltonian.shape}")
        except Exception as e:
            self.logger.error(f"Error loading matrices: {str(e)}")
            raise

    def setup_energy_grid(self) -> None:
        """Set up energy grid for calculations"""
        self.energy_grid = np.linspace(
            self.params.energy_range[0],
            self.params.energy_range[1],
            self.params.energy_points
        )
        self.voltage_grid = np.linspace(
            self.params.voltage_range[0],
            self.params.voltage_range[1],
            self.params.voltage_points
        )

    def fermi_function(self, energy: float, mu: float) -> float:
        """Calculate Fermi-Dirac distribution"""
        return 1.0 / (1.0 + np.exp((energy - mu) / (KB * self.params.temperature)))

    def calculate_self_energy(self, energy: float) -> Tuple[np.ndarray, np.ndarray]:
        """High-precision self-energy calculation with caching"""
        cache_key = round(energy, self.numerical_params['cache_decimals'])
        if cache_key in self._self_energy_cache:
            return self._self_energy_cache[cache_key]
        
        n = self.hamiltonian.shape[0]
        gamma = self.params.eta
        
        # Use high precision complex numbers
        sigma_l = -1j * gamma * np.eye(n, dtype=self.numerical_params['matrix_precision'])
        sigma_r = -1j * gamma * np.eye(n, dtype=self.numerical_params['matrix_precision'])
        
        # Check matrix conditioning
        if np.linalg.cond(sigma_l) > 1/self.numerical_params['min_determinant']:
            self.logger.warning(f"Self-energy matrices poorly conditioned at E={energy:.6f}")
        
        self._self_energy_cache[cache_key] = (sigma_l, sigma_r)
        return sigma_l, sigma_r

    def calculate_greens_function(self, energy: float) -> np.ndarray:
        """High-precision Green's function calculation with stability checks"""
        cache_key = round(energy, self.numerical_params['cache_decimals'])
        if cache_key in self._green_cache:
            return self._green_cache[cache_key]
        
        try:
            sigma_l, sigma_r = self.calculate_self_energy(energy)
            z = (energy + 1j * self.params.eta) * self.overlap
            g_inv = z - self.hamiltonian - sigma_l - sigma_r
            
            # Check matrix condition number
            cond = np.linalg.cond(g_inv)
            if cond > 1/self.numerical_params['min_determinant']:
                self.logger.warning(f"Poor matrix conditioning at E={energy:.6f}, cond={cond:.2e}")
                
                # Use SVD with cutoff for better stability
                u, s, vh = np.linalg.svd(g_inv)
                s[s < self.numerical_params['svd_cutoff']] = self.numerical_params['svd_cutoff']
                g = vh.conj().T @ np.diag(1/s) @ u.conj().T
            else:
                g = np.linalg.solve(g_inv, np.eye(g_inv.shape[0]))
            
            self._green_cache[cache_key] = g
            return g
        except np.linalg.LinAlgError as e:
            self.logger.error(f"Matrix inversion failed at E={energy:.6f}: {str(e)}")
            return np.zeros_like(self.hamiltonian, dtype=self.numerical_params['matrix_precision'])

    def calculate_gamma(self, sigma: np.ndarray) -> np.ndarray:
        """Calculate broadening matrix"""
        return 1j * (sigma - sigma.conj().T)

    def process_energy_chunk(self, energies: np.ndarray) -> List[float]:
        """Process energy chunk with high precision and stability checks"""
        results = []
        for energy in energies:
            try:
                g = self.calculate_greens_function(energy)
                sigma_l, sigma_r = self.calculate_self_energy(energy)
                gamma_l = self.calculate_gamma(sigma_l)
                gamma_r = self.calculate_gamma(sigma_r)
                
                # Check matrix properties
                if np.any(np.isnan(g)) or np.any(np.isinf(g)):
                    raise ValueError(f"Invalid Green's function at E={energy:.6f}")
                
                temp = gamma_l @ g @ gamma_r @ g.conj().T
                transmission = np.real(np.trace(temp))
                
                # Validate transmission value
                if transmission < 0 or np.isnan(transmission) or np.isinf(transmission):
                    self.logger.warning(f"Invalid transmission {transmission} at E={energy:.6f}")
                    transmission = 0.0
                
                results.append(transmission)
            except Exception as e:
                self.logger.error(f"Error processing energy E={energy:.6f}: {str(e)}")
                results.append(0.0)
        return results

    def calculate_density_matrix(self) -> np.ndarray:
        """
        Calculate density matrix using parallel energy integration.
        """
        num_threads = min(os.cpu_count(), 8)
        
        def integrand(energy: float) -> np.ndarray:
            g = self.calculate_greens_function(energy)
            sigma_l, sigma_r = self.calculate_self_energy(energy)
            gamma_l = self.calculate_gamma(sigma_l)
            gamma_r = self.calculate_gamma(sigma_r)
            
            spectral = g @ (gamma_l + gamma_r) @ g.conj().T
            return spectral * self.fermi_function(energy, self.params.fermi_energy)

        # Parallel calculation over energy grid
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            chunk_size = len(self.energy_grid) // num_threads
            chunks = [self.energy_grid[i:i + chunk_size] for i in range(0, len(self.energy_grid), chunk_size)]
            
            futures = []
            for chunk in chunks:
                future = executor.submit(lambda x: [integrand(E) for E in x], chunk)
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                results.extend(future.result())
        
        # Integrate over energy
        density_matrix = simpson(results, self.energy_grid) / (2 * np.pi)
        return density_matrix

    def calculate_current(self, voltage: float) -> float:
        """High-precision current calculation with error checking"""
        try:
            mu_l = self.params.fermi_energy + voltage/2
            mu_r = self.params.fermi_energy - voltage/2
            
            # Vectorized calculation with high precision
            delta_f = self.fermi_function(self.energy_grid, mu_l) - \
                     self.fermi_function(self.energy_grid, mu_r)
            
            # Check for numerical overflow in Fermi functions
            if np.any(np.isnan(delta_f)) or np.any(np.isinf(delta_f)):
                self.logger.error(f"Numerical overflow in Fermi function at V={voltage:.6f}")
                return 0.0
            
            # High precision integration
            integrand = self.transmission * delta_f
            current = simpson(integrand, self.energy_grid, 
                             rtol=self.numerical_params['integration_rtol'],
                             atol=self.numerical_params['integration_atol'])
            
            # Scale with high precision constants
            scaled_current = (2 * E_CHARGE * E_CHARGE / H_PLANCK) * current
            
            # Check result validity
            if np.isnan(scaled_current) or np.isinf(scaled_current):
                raise ValueError(f"Invalid current value at V={voltage:.6f}")
            
            return scaled_current
        except Exception as e:
            self.logger.error(f"Current calculation failed at V={voltage:.6f}: {str(e)}")
            return 0.0

    def run_calculation(self) -> None:
        """Optimized calculation with chunked processing"""
        self.logger.info("Starting transport calculations...")
        
        try:
            # Initial setup
            self.setup_energy_grid()
            num_threads = min(os.cpu_count(), 8)
            self.logger.info(f"System size: {self.hamiltonian.shape[0]}x{self.hamiltonian.shape[0]}")
            
            # Split energy grid into chunks for better cache utilization
            energy_chunks = np.array_split(self.energy_grid, 
                                         max(1, len(self.energy_grid) // self.chunk_size))
            
            # Calculate transmission with chunked parallel processing
            self.logger.info("Calculating transmission...")
            self.transmission = np.zeros(len(self.energy_grid))
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(self.process_energy_chunk, chunk): i 
                          for i, chunk in enumerate(energy_chunks)}
                
                completed_chunks = 0
                for future in futures:
                    try:
                        chunk_idx = futures[future]
                        start_idx = chunk_idx * self.chunk_size
                        chunk_results = future.result(timeout=300)
                        
                        # Store results
                        end_idx = min(start_idx + len(chunk_results), len(self.transmission))
                        self.transmission[start_idx:end_idx] = chunk_results
                        
                        completed_chunks += 1
                        self.logger.info(f"Transmission progress: {completed_chunks/len(energy_chunks)*100:.1f}%")
                    except Exception as e:
                        self.logger.error(f"Chunk {chunk_idx} failed: {str(e)}")
            
            # Clear caches to free memory
            self._green_cache.clear()
            self._self_energy_cache.clear()
            
            # Calculate current (reuse transmission results)
            self.logger.info("Calculating I-V characteristics...")
            self.current = np.zeros(len(self.voltage_grid))
            
            def calculate_current_for_voltage(voltage: float) -> float:
                try:
                    mu_l = self.params.fermi_energy + voltage/2
                    mu_r = self.params.fermi_energy - voltage/2
                    
                    # Vectorized calculation of Fermi function difference
                    delta_f = self.fermi_function(self.energy_grid, mu_l) - \
                             self.fermi_function(self.energy_grid, mu_r)
                    
                    # Vectorized integration
                    integrand = self.transmission * delta_f
                    current = simpson(integrand, self.energy_grid)
                    return (2 * E_CHARGE * E_CHARGE / H_PLANCK) * current
                except Exception as e:
                    self.logger.error(f"Error at V={voltage:.3f}V: {str(e)}")
                    return 0.0
            
            # Parallel current calculation
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(calculate_current_for_voltage, V): i 
                          for i, V in enumerate(self.voltage_grid)}
                
                for future in futures:
                    try:
                        idx = futures[future]
                        self.current[idx] = future.result(timeout=60)
                        if (idx + 1) % (len(self.voltage_grid) // 5) == 0:
                            self.logger.info(f"I-V progress: {(idx + 1)/len(self.voltage_grid)*100:.1f}%")
                    except Exception as e:
                        self.logger.error(f"Current calculation failed: {str(e)}")
            
            # Save and plot results
            self.save_results()
            self.plot_results()
            
            self.logger.info("\nFinal Results:")
            self.logger.info(f"Max transmission: {np.max(self.transmission):.3e}")
            self.logger.info(f"Max current: {np.max(np.abs(self.current)):.3e} A")
            
        except Exception as e:
            self.logger.error(f"Calculation failed: {str(e)}")
            raise

    def save_results(self) -> None:
        """Save calculation results"""
        self.logger.info("Saving results to files...")
        
        np.save('transmission.npy', self.transmission)
        np.save('current.npy', self.current)
        np.save('density_matrix.npy', self.density_matrix)
        
        metadata = {
            'energy_range': self.params.energy_range,
            'temperature': self.params.temperature,
            'eta': self.params.eta,
            'voltage_range': self.params.voltage_range,
            'fermi_energy': self.params.fermi_energy,
            'matrix_dimensions': self.hamiltonian.shape[0],
            'transmission_max': float(np.max(self.transmission)),
            'current_max': float(np.max(np.abs(self.current))),
        }
        
        with open('results.json', 'w') as f:
            json.dump(metadata, f, indent=4)

    def plot_results(self) -> None:
        """Plot transmission and I-V characteristics"""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot transmission
        ax1.plot(self.energy_grid, self.transmission)
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Transmission')
        ax1.set_title('Transmission Function')
        
        # Plot I-V characteristic
        ax2.plot(self.voltage_grid, self.current)
        ax2.set_xlabel('Voltage (V)')
        ax2.set_ylabel('Current (A)')
        ax2.set_title('I-V Characteristic')
        
        plt.tight_layout()
        plt.savefig('transport_results.png')
        plt.close()

def main():
    """Main execution function"""
    try:
        # Initialize transport calculator
        params = TransportParameters()
        calculator = QuantumTransport(params)
        
        # Load matrices
        calculator.load_matrices()
        
        # Run calculations
        calculator.run_calculation()
        
        # Plot results
        calculator.plot_results()
        
    except Exception as e:
        logging.error(f"Calculation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 