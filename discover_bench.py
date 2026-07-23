from huggingface_hub import list_repo_files
CANDS = [
  ("allenai/ai2_arc", "ARC"),
  ("Rowan/hellaswag", "HellaSwag"),
  ("ybisk/piqa", "PIQA"),
  ("aps/super_glue", "BoolQ/SuperGLUE"),
  ("cais/mmlu", "MMLU"),
]
for rid, name in CANDS:
    try:
        fs = list_repo_files(rid, repo_type="dataset")
        print(f"  OK   {name:16} {rid:24} {len(fs)} dosya")
    except Exception as e:
        print(f"  YOK  {name:16} {rid:24} {type(e).__name__}")
