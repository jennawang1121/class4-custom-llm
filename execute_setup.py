"""Execute the setup notebook and preserve outputs, including errors."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import sys

root = Path(__file__).resolve().parent
path = root / (sys.argv[1] if len(sys.argv) > 1 else 'setup_10steps.ipynb')
nb = nbformat.read(path, as_version=4)
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(nb, km=km, timeout=600, resources={'metadata': {'path': str(root)}})
def progress(cell, cell_index, **kwargs):
    print(f'Executing cell {cell_index}', flush=True)
client.on_cell_start = progress
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print(f'Saved executed notebook: {path}', flush=True)
