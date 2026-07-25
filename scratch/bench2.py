from datasets import load_dataset
for name, args in [("ARC-Easy",("allenai/ai2_arc","ARC-Easy")), ("MMLU",("cais/mmlu","all"))]:
    try:
        d = load_dataset(*args, split="test")
        ex = d[0]
        print(f"OK {name:10} n={len(d):6} alanlar={list(ex.keys())}")
        print(f"   ornek: {str(ex)[:220]}")
    except Exception as e:
        print(f"YOK {name}: {type(e).__name__}: {str(e)[:80]}")
