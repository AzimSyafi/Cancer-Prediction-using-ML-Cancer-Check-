"""
Writes requirements.txt with the exact library versions installed on this computer.

Run it once before uploading to GitHub:
    c:\\python313\\python.exe make_requirements.py

The saved models only load with the same scikit-learn and imbalanced-learn
versions they were trained with, so the hosted app must use these exact versions.
"""

import os
from importlib.metadata import version, PackageNotFoundError

PACKAGES = ["streamlit", "pandas", "numpy", "scikit-learn", "imbalanced-learn",
            "joblib", "plotly", "fpdf2", "matplotlib"]

lines = []
for name in PACKAGES:
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print(f"Not installed: {name}. Install it with: python -m pip install {name}")

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
print("Wrote requirements.txt:")
print("\n".join(lines))
