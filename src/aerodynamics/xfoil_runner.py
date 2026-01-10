"""
XFOIL Runner for Aerodynamic Analysis.
"""

import subprocess
import os
import numpy as np
from pathlib import Path
import tempfile
import time

class XFoilRunner:
    """
    Handles interaction with XFOIL for aerodynamic analysis of airfoils.
    """

    def __init__(self, xfoil_path: str = "xfoil", timeout: float = 10.0):
        """
        Initialize the XFoil runner.

        Parameters
        ----------
        xfoil_path : str, optional
            Path to the XFOIL executable. Default is "xfoil".
        timeout : float, optional
            Timeout in seconds for sub-process execution. Default is 10.0.
        """
        self.xfoil_path = xfoil_path
        self.timeout = timeout

    def analyze(self, coordinates: np.ndarray, reynolds: float, alpha: float, iter_limit: int = 100) -> dict:
        """
        Perform aerodynamic analysis using XFOIL.

        Parameters
        ----------
        coordinates : np.ndarray
            (N, 2) array of X, Y coordinates.
        reynolds : float
            Reynolds number.
        alpha : float
            Angle of attack in degrees.
        iter_limit : int, optional
            Iteration limit for valid viscous solution. Default is 100.

        Returns
        -------
        dict
            Dictionary containing 'CL', 'CD', and 'CM'. Returns None for values if analysis fails.
        """
        # I'm creating temporary files here. I need to be careful with the working directory 
        # because XFOIL tends to write files to the CWD. Also, XFOIL can be finicky 
        # with long paths or spaces, so I'll run it directly inside the temp folder.
        
        original_cwd = os.getcwd()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            airfoil_file = temp_path / "airfoil.dat"
            results_file = "results.pol" # Relative path for XFOIL command
            
            # Write coordinates used for analysis
            self._write_coordinates(coordinates, airfoil_file)
            
            # I'm setting up the XFOIL input commands. 
            # I run everything inside the temp directory to avoid any path issues.
            commands = [
                f"LOAD {airfoil_file.name}",
                "OPER",
                f"VISC {reynolds}",
                f"ITER {iter_limit}",
                "PACC",
                f"{results_file}",  # Output polar file
                "",                 # Dump file (empty -> no dump)
                f"ALFA {alpha}",
                "PACC",             # Turn off accumulation
                "QUIT"
            ]
            
            input_str = "\n".join(commands)
            
            try:
                # Execute XFOIL
                result = subprocess.run(
                    [self.xfoil_path],
                    input=input_str,
                    text=True,
                    capture_output=True,
                    cwd=temp_path,  # Run inside the temp folder
                    timeout=self.timeout
                )
                
                if result.returncode != 0:
                    # XFOIL often returns non-zero codes even if it doesn't crash explicitly,
                    # so I'll just print the warning for now.
                    print(f"XFOIL warning/error: {result.stderr}")

                # Parse results
                cl, cd, cm = self._parse_results(temp_path / results_file, alpha)
                return {'CL': cl, 'CD': cd, 'CM': cm}

            except subprocess.TimeoutExpired:
                print("XFOIL analysis timed out.")
                return {'CL': None, 'CD': None, 'CM': None}
            except Exception as e:
                print(f"XFOIL execution error: {e}")
                return {'CL': None, 'CD': None, 'CM': None}

    def _write_coordinates(self, coordinates: np.ndarray, filepath: Path):
        """Write coordinates to an XFOIL-compatible text file."""
        with open(filepath, 'w') as f:
            f.write("Generated Airfoil\n")
            for x, y in coordinates:
                f.write(f" {x:.6f}    {y:.6f}\n")

    def _parse_results(self, result_path: Path, target_alpha: float) -> tuple:
        """
        Parse the XFOIL polar file to find CL and CD.
        """
        if not result_path.exists():
            return None, None

        try:
            with open(result_path, 'r') as f:
                lines = f.readlines()
            
            # XFOIL Polar file format typically starts with header lines
            # Data usually starts after the line starting with "------"
            # Format:  alpha    CL        CD       CDp       CM     Top_Xcp  Bot_Xcp
            
            data_start = False
            for line in lines:
                if "------" in line:
                    data_start = True
                    continue
                
                if data_start and line.strip():
                    parts = line.split()
                    try:
                        alpha = float(parts[0])
                        # Checking if this row matches my target alpha (approximate match)
                        if abs(alpha - target_alpha) < 0.01:
                            cl = float(parts[1])
                            cd = float(parts[2])
                            cm = float(parts[4])
                            return cl, cd, cm
                    except (ValueError, IndexError):
                        continue
                        
            return None, None, None
            
        except Exception:
            return None, None, None
