#!/usr/bin/env python3

import os
import sys
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from dft_hamiltonian import DFTHamiltonian
from quantum_transport import QuantumTransport, TransportParameters

class TransportCalculation:
    """
    Orchestrates DFT and quantum transport calculations.
    """
    def __init__(self, xyz_file: str, output_dir: str = None):
        self.xyz_file = xyz_file
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            molecule_name = Path(xyz_file).stem
            self.output_dir = f"output_{molecule_name}_{timestamp}"
        else:
            self.output_dir = output_dir
            
        # Set up logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging to both file and console"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler(f"{self.output_dir}/calculation.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        
        # Root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    def run_dft(self):
        """Run DFT calculation"""
        self.logger.info("Starting DFT calculation...")
        
        try:
            dft_calc = DFTHamiltonian(self.xyz_file)
            hamiltonian, overlap = dft_calc.calculate()
            
            # Save matrices to output directory
            np.save(f"{self.output_dir}/hamiltonian.npy", hamiltonian)
            np.save(f"{self.output_dir}/overlap.npy", overlap)
            
            self.logger.info("DFT calculation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"DFT calculation failed: {str(e)}")
            raise
            
    def run_transport(self):
        """Run quantum transport calculation"""
        self.logger.info("Starting transport calculation...")
        
        try:
            # Set up transport parameters
            params = TransportParameters(
                energy_range=(-5.0, 5.0),
                energy_points=1000,
                temperature=300.0,
                voltage_range=(-2.0, 2.0),
                voltage_points=401
            )
            
            # Initialize transport calculator
            calculator = QuantumTransport(params)
            
            # Load matrices from output directory
            calculator.load_matrices(
                hamiltonian_file=f"{self.output_dir}/hamiltonian.npy",
                overlap_file=f"{self.output_dir}/overlap.npy"
            )
            
            # Run calculation
            calculator.run_calculation()
            
            # Move results to output directory
            result_files = ['transmission.npy', 'current.npy', 
                          'density_matrix.npy', 'results.json',
                          'transport_results.png']
            
            for file in result_files:
                if os.path.exists(file):
                    shutil.move(file, f"{self.output_dir}/{file}")
            
            self.logger.info("Transport calculation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Transport calculation failed: {str(e)}")
            raise
            
    def run_full_calculation(self):
        """Run complete DFT + transport calculation workflow"""
        self.logger.info(f"Starting calculation for {self.xyz_file}")
        self.logger.info(f"Results will be saved in {self.output_dir}")
        
        try:
            # Copy input XYZ file to output directory
            shutil.copy2(self.xyz_file, f"{self.output_dir}/")
            
            # Run calculations
            self.run_dft()
            self.run_transport()
            
            self.logger.info("All calculations completed successfully")
            self.logger.info(f"Results are available in {self.output_dir}")
            
        except Exception as e:
            self.logger.error("Calculation failed")
            raise

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run quantum transport calculations")
    parser.add_argument("xyz_file", help="Input XYZ file")
    parser.add_argument("--output", "-o", help="Output directory (optional)")
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.xyz_file):
        print(f"Error: File {args.xyz_file} not found")
        sys.exit(1)
    
    try:
        # Run calculation
        calc = TransportCalculation(args.xyz_file, args.output)
        calc.run_full_calculation()
        
    except Exception as e:
        print(f"Calculation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 