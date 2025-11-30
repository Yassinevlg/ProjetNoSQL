import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
import sys

NOTEBOOK = Path('train_sign_language_cnn.ipynb')
OUT = Path('executed_partial.ipynb')
MAX_CELL = 6  # execute up to cell 6 (imports, config, dataset download)

if not NOTEBOOK.exists():
    print(f'Notebook not found: {NOTEBOOK}', file=sys.stderr)
    sys.exit(2)

nb = nbformat.read(str(NOTEBOOK), as_version=4)
# take first MAX_CELL cells
nb_subset = nbformat.v4.new_notebook()
nb_subset['cells'] = nb['cells'][:MAX_CELL]
nb_subset['metadata'] = nb.get('metadata', {})

print(f'Executing first {MAX_CELL} cells of {NOTEBOOK}...')

try:
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb_subset, {'metadata': {'path': '.'}})
    nbformat.write(nb_subset, str(OUT))
    print('Partial execution finished, written to', OUT)
except Exception as e:
    print('Partial execution failed:', e, file=sys.stderr)
    try:
        nbformat.write(nb_subset, str(OUT))
        print('Partial output (with errors) written to', OUT)
    except Exception:
        pass
    sys.exit(1)
