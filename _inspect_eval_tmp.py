import json
from pathlib import Path
root = Path(r'c:\Users\LNV\Workspace\PACLIC_2026\ViCoQA\qwen2.5-1.5b\LoRa')
for p in sorted(root.glob('eval_preds_*_test.json')):
    data = json.load(open(p, encoding='utf-8'))
    print(p.name, 'n=', len(data), 'keys=', list(data[0].keys()) if data else None)
    em = 100*sum(x['em'] for x in data)/len(data)
    f1 = 100*sum(x['f1'] for x in data)/len(data)
    print(f'  EM={em:.4f} F1={f1:.4f}')
