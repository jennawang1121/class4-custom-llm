"""Rerun a formal experiment in a new isolated folder, preserving saved evidence."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent

def prepare(experiment, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ['nanogpt_model.py', 'NANOGPT_LICENSE', 'run_evals.py', 'chat.py']:
        shutil.copy2(ROOT/name, destination/name)
    shutil.copytree(ROOT/'evals', destination/'evals')
    (destination/'corpus').mkdir()
    if experiment == 'expanded':
        for name in ['grammar.txt', 'spatial_relations.txt']:
            shutil.copy2(ROOT/'corpus'/name, destination/'corpus'/name)
    nb = nbformat.read(ROOT/f'{experiment}_3000steps.ipynb', as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    target = destination/'custom_llm.ipynb'
    nbformat.write(nb, target)
    return nb, target

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['starter', 'expanded'])
    parser.add_argument('--prepare-only', action='store_true', help='Create isolated files without training; open its notebook in Jupyter/VS Code.')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = ROOT/'reproductions'/f'{args.experiment}_{stamp}'
    nb, target = prepare(args.experiment, destination)
    print(f'Isolated notebook: {target}', flush=True)
    if args.prepare_only:
        return
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(nb, km=km, timeout=600, resources={'metadata': {'path': str(destination)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, target)
    print(f'Completed; notebook and fresh llm_runs are in {destination}', flush=True)

if __name__ == '__main__':
    main()
