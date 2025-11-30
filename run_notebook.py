import asyncio
import sys
import traceback
from pathlib import Path

# Ensure Windows selector event loop policy to avoid Proactor add_reader issues
try:
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

from nbformat import read, write, NO_CONVERT
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK = Path('train_sign_language_cnn.ipynb')
OUT = Path('executed_train.ipynb')

if not NOTEBOOK.exists():
    print(f'ERROR: {NOTEBOOK} not found', file=sys.stderr)
    sys.exit(2)

print(f'Executing notebook: {NOTEBOOK} -> {OUT}')

try:
    nb = read(str(NOTEBOOK), as_version=4)
    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    write(nb, str(OUT))
    print('Execution finished, written to', OUT)
except Exception as e:
    print('Notebook execution failed:', file=sys.stderr)
    traceback.print_exc()
    # try to write partial output if available
    try:
        write(nb, str(OUT))
        print('Partial output written to', OUT)
    except Exception:
        pass
    sys.exit(1)
