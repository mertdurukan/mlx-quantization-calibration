.PHONY: setup verify test prompt-freeze pilot reproduce paper

PYTHON := ./.venv/bin/python

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

verify:
	$(PYTHON) -c "\
import shutil; \
from huggingface_hub import snapshot_download; \
from mlx_lm import load, convert; \
import mlx.core as mx; \
import numpy as np; \
SRC = 'mlx-community/Qwen2.5-0.5B-Instruct-bf16'; \
snapshot_download(SRC); \
OUT = '/tmp/mlxq_verify'; \
shutil.rmtree(OUT, ignore_errors=True); \
convert(hf_path=SRC, mlx_path=OUT, quantize=True, q_bits=4, q_group_size=64, q_mode='affine'); \
m, tok = load(OUT); \
ids = tok.encode('Question: What is 2+2?' + chr(10) + 'A) 3' + chr(10) + 'B) 4' + chr(10) + 'Answer:'); \
lg = m(mx.array([ids]))[0, -1, :]; \
lp = lg - mx.logsumexp(lg); \
print('logprob read ok, sum(exp)=', float(mx.exp(lp).sum())); \
conf = np.array([0.9, 0.6, 0.8, 0.95]); \
correct = np.array([True, False, True, True]); \
order = np.argsort(conf); \
bins = np.array_split(order, 2); \
e = sum(abs(correct[b].mean() - conf[b].mean()) * len(b) for b in bins) / len(conf); \
print('ece computed ok:', e); \
shutil.rmtree(OUT, ignore_errors=True)"

test:
	$(PYTHON) -m pytest tests/ -q

prompt-freeze:
	$(PYTHON) -c "import hashlib; print(hashlib.sha256(open('prompts/mc_letter.txt','rb').read()).hexdigest())"

pilot:
	$(PYTHON) -m src.runner --pilot

reproduce: test
	$(PYTHON) -m src.runner
	$(PYTHON) -m src.analyze

# Derive the arXiv LaTeX package from paper.md (the single source) into
# build/arxiv/. Needs pandoc and a LaTeX distribution; see ARXIV.md.
paper:
	$(PYTHON) scripts/build_paper.py
