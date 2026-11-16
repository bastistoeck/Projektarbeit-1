### run DOS calculation for tmdc at special points only

import subprocess
import sys

parameter_sets = [
    (0.0, 0.0), # gamma
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (1/3.0, 1/3.0), # K (one of them)
]

for i, (param1, param2) in enumerate(parameter_sets):

    command = [
        "mpiexec",
        "-np", "4", #"python",
        sys.executable,
        "dos_numpy_tmdc.py",
        "--k_x", str(param1),
        "--k_y", str(param2)
    ]

    print(f"\nStarting DOS calculation {i+1}/{len(parameter_sets)}")
    print(f"Parameters: k_x_reduced = {param1}, k_y_reduced = {param2}")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(f"DOS calculation {i+1} failed!")
        print(result.stderr)
        break


print("\nDONE")