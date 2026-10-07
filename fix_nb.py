import json

with open('Gated_Attention_YOLOv8n_Research_Pipeline (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('patched_source' in line for line in cell['source']):
        cell['source'] = [
            'import inspect\n',
            'import contextlib\n',
            'import ast\n',
            'import ultralytics.nn.tasks as tasks\n',
            '\n',
            'original_source = inspect.getsource(tasks.parse_model)\n',
            '\n',
            'injection = """\n',
            '        elif m is C2f_Attention:\n',
            '            c1, c2 = ch[f], args[0]\n',
            '            args = [c1, c2, *args[1:]]\n',
            '            args.insert(2, n)\n',
            '            n = 1\n',
            '        elif m is GatedFusion:\n',
            '            c1a, c1b = (ch[x] for x in f)\n',
            '            c2 = args[0]\n',
            '            args = [c1a, c1b, c2]\n',
            '"""\n',
            '\n',
            'patched_source = original_source.replace("elif m is Concat:", injection.lstrip("\\n") + "        elif m is Concat:")\n',
            'patched_source = patched_source.replace("def parse_model(", "def parse_model_patched(")\n',
            '\n',
            'exec(compile(patched_source, "<parse_model_patched>", "exec"), tasks.__dict__)\n',
            'tasks.parse_model = tasks.__dict__["parse_model_patched"]\n',
            'print("parse_model dynamically patched for custom layers:", tasks.parse_model)\n'
        ]

with open('Gated_Attention_YOLOv8n_Research_Pipeline_Fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
