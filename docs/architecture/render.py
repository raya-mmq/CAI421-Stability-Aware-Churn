"""Render the final architecture diagram (needs Graphviz `dot` on PATH).
    python docs/architecture/render.py
Writes final_architecture.svg (editable vector) and final_architecture.png (300 dpi) next to the .dot."""
import re, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
dot = HERE / "final_architecture.dot"
subprocess.run(["dot", "-Tpng", "-Gdpi=300", str(dot), "-o", str(HERE / "final_architecture.png")], check=True)
svg = subprocess.run(["dot", "-Tsvg", str(dot)], check=True, capture_output=True, text=True).stdout
svg = re.sub(r'font-family="Inter[^"]*"', 'font-family="Inter, Helvetica, Arial, sans-serif"', svg)
(HERE / "final_architecture.svg").write_text(svg)
print("wrote", HERE / "final_architecture.png", "and", HERE / "final_architecture.svg")
