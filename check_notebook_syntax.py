import nbformat
import ast
from pathlib import Path

nb_path = Path('train_sign_language_cnn.ipynb')
if not nb_path.exists():
    print('Notebook not found:', nb_path)
    raise SystemExit(2)

nb = nbformat.read(str(nb_path), as_version=4)
errors = []
for i, cell in enumerate(nb.cells, start=1):
    if cell.cell_type != 'code':
        continue
    source = ''.join(cell.get('source', []))
    try:
        ast.parse(source)
    except SyntaxError as e:
        # show context lines
        src_lines = source.splitlines()
        lineno = e.lineno or 0
        start = max(0, lineno-3)
        end = min(len(src_lines), lineno+2)
        context = '\n'.join(f'{j+1:4d}: {ln}' for j, ln in enumerate(src_lines[start:end], start=start))
        errors.append((i, e, context))

if not errors:
    print('No syntax errors detected in code cells.')
else:
    print(f'Detected {len(errors)} syntax error(s):')
    for cell_idx, err, ctx in errors:
        print('\n---- Cell', cell_idx, '----')
        print('SyntaxError:', err.msg)
        print('At line:', err.lineno)
        print('Context:')
        print(ctx)

# exit non-zero if errors found
if errors:
    raise SystemExit(1)
