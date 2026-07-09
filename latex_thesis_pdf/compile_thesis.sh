#!/bin/bash

# Define the temporary python script name
PY_SCRIPT="temp_compile.py"

# Write the python compilation script
cat << 'EOF' > "$PY_SCRIPT"
import subprocess
import sys
import os

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        # bibtex often returns non-zero even with formatting issues, but pdflatex failure is more severe.
        if cmd[0] != 'bibtex':
            print("--- Output ---")
            print(result.stdout)
            print("--- Error ---")
            print(result.stderr)
            sys.exit(result.returncode)
    return result

if __name__ == "__main__":
    tex_file = "thesis.tex"
    
    if not os.path.exists(tex_file):
        print(f"Error: {tex_file} not found!")
        sys.exit(1)
        
    print("--- Starting compilation process ---")
    
    # 1. First pdflatex run to generate .aux
    run_command(["pdflatex", "-interaction=nonstopmode", tex_file])
    
    # 2. bibtex run to process bibliography
    run_command(["bibtex", tex_file.replace(".tex", "")])
    
    # 3. Second pdflatex run to include bibliography and update references
    run_command(["pdflatex", "-interaction=nonstopmode", tex_file])
    
    # 4. Third pdflatex run to get cross-references perfectly right
    run_command(["pdflatex", "-interaction=nonstopmode", tex_file])
    
    print("--- Compilation finished successfully ---")

EOF

# Make sure we use python3
if command -v python3 &>/dev/null; then
    python3 "$PY_SCRIPT"
else
    # Fallback to just python if python3 is not explicitly available
    python "$PY_SCRIPT"
fi

# Clean up the temporary python script
rm -f "$PY_SCRIPT"
