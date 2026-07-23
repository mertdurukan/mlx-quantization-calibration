from datasets import load_dataset

SPECS = [
  ("ARC-Easy", lambda: load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")),
  ("ARC-Chal", lambda: load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")),
  ("HellaSwag", lambda: load_dataset("Rowan/hellaswag", split="validation")),
  ("PIQA", lambda: load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)),
]
for name, fn in SPECS:
    try:
        d = fn()
        ex = d[0]
        print(f"OK   {name:10} n={len(d):6}  alanlar={list(ex.keys())}")
    except Exception as e:
        print(f"YOK  {name:10} {type(e).__name__}: {str(e)[:80]}")
