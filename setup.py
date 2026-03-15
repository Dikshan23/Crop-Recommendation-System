import os
import shutil

# ── Paths ─────────────────────────────────────────────────────────────────────
base      = "/home/prajwalmac/7th Semester/Project/Project Demo/AgroTree"
datas_dir = "/home/prajwalmac/7th Semester/Project/Project Demo/datas"

# ── 1. Ensure all folders exist ───────────────────────────────────────────────
folders = [
    "data/processed",
    "notebooks",
    "src",
    "results/plots",
    "results/metrics",
]
for folder in folders:
    os.makedirs(os.path.join(base, folder), exist_ok=True)

# ── 2. File mapping: (absolute source, relative destination) ──────────────────
moves = [
    # CSV dataset
    (os.path.join(datas_dir, "Crop_recommendation.csv"),
     "data/processed/Crop_recommendation.csv"),

    # Notebooks (if still in root)
    (os.path.join(base, "eda.ipynb"),       "notebooks/eda.ipynb"),
    (os.path.join(base, "script.ipynb"),    "notebooks/script.ipynb"),

    # script.py from datas
    (os.path.join(datas_dir, "script.py"),  "src/script.py"),
]

# ── 3. Move files ──────────────────────────────────────────────────────────────
moved, skipped = [], []

for src, dst_rel in moves:
    dst = os.path.join(base, dst_rel)

    # Skip if already in destination
    if os.path.abspath(src) == os.path.abspath(dst):
        skipped.append(f"  ↷  {dst_rel}  (already in place)")
        continue

    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        moved.append(f"  ✔  {os.path.basename(src)}  →  {dst_rel}")
    else:
        skipped.append(f"  ✘  {os.path.basename(src)}  (not found, skipped)")

# ── 4. Write missing files ────────────────────────────────────────────────────

# README.md
readme_path = os.path.join(base, "README.md")
if not os.path.exists(readme_path):
    with open(readme_path, "w") as f:
        f.write("# AgroTree — Crop Recommendation System\n")
    moved.append("  ✔  README.md  →  created")

# requirements.txt
req_path = os.path.join(base, "requirements.txt")
if not os.path.exists(req_path):
    with open(req_path, "w") as f:
        f.write(
            "numpy==2.4.1\n"
            "pandas==2.3.3\n"
            "matplotlib==3.10.8\n"
            "seaborn==0.13.2\n"
            "scikit-learn\n"
            "ipykernel\n"
            "jupyter_client\n"
            "jupyter_core\n"
            "graphviz==0.21\n"
            "pillow==12.1.1\n"
        )
    moved.append("  ✔  requirements.txt  →  created")

# .gitignore
gitignore_path = os.path.join(base, ".gitignore")
with open(gitignore_path, "w") as f:
    f.write(
        "# Python cache\n__pycache__/\n*.pyc\n\n"
        "# Virtual environments\nvenv/\n.env/\n\n"
        "# Jupyter notebook checkpoints\n.ipynb_checkpoints\n\n"
        "# VS Code settings\n.vscode/\n\n"
        "# System files\n.DS_Store\nThumbs.db\n\n"
        "# Data files\n*.csv\n*.xlsx\n\n"
        "# Large model files\n*.pkl\n*.joblib\n*.h5\n\n"
        "# Results/outputs\nresults/\n"
    )
moved.append("  ✔  .gitignore  →  written")

# ── 5. Clean up empty datas/ folder ──────────────────────────────────────────
if os.path.isdir(datas_dir) and not os.listdir(datas_dir):
    os.rmdir(datas_dir)
    print(f"Removed empty folder: {datas_dir}")

# ── 6. Report ─────────────────────────────────────────────────────────────────
print("\n=== AgroTree Setup Complete ===\n")
if moved:
    print("Done:")
    print("\n".join(moved))
if skipped:
    print("\nSkipped:")
    print("\n".join(skipped))

print("\nFinal structure:")
for root, dirs, files in os.walk(base):
    dirs[:] = sorted([d for d in dirs if d not in ["__pycache__", ".git"]])
    level  = root.replace(base, "").count(os.sep)
    indent = "    " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in sorted(files):
        print(f"{indent}    {f}")