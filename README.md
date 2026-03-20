# AIMO Progress Prize 3 — Submission Pipeline

**3-layer deterministic solver for the [AI Mathematical Olympiad Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)**

## Pipeline

| Layer | Method | Result |
|-------|--------|--------|
| 1 | SymPy symbolic pre-solver (8 typed solvers) | 7/10 reference problems ✅ deterministic |
| 2 | GPT-OSS-120B TIR via vLLM (SC-TIR, 8 samples, majority vote) | ~35-40/50 |
| 3 | Lean4 formal verification + plausibility filter | 0.5→1.0 pts upgrade |

## Files

- `submission.ipynb` — Kaggle submission notebook (attach GPT-OSS-120B + layer1-solvers dataset)
- `layer1_solvers.py` — Standalone SymPy pre-solver module (8 mathematical solvers)
- `data/reference.csv` — 10 reference problems used for local validation
- `docs.html` — Full competition analysis & strategy document

## Validation

```
Result: 7/10 correct | Layer 1 hits: 7/10 | Pipeline health: ✅ READY

0e644e  geometry       ⬜ LLM
26de63  number_theory  ⬜ LLM
424e18  number_theory  ✅ 21818
42d360  number_theory  ✅ 32193
641659  geometry       ⬜ LLM
86e8e5  number_theory  ✅ 8687
92ba6a  algebra        ✅ 50
9c1c5f  algebra        ✅ 580
a295e9  geometry       ✅ 520
dd7f5e  algebra        ✅ 160
```

## Kaggle Setup

1. Upload `layer1_solvers.py` as a Kaggle dataset named `layer1-solvers`
2. Import `submission.ipynb` into a competition notebook
3. Attach model: `unsloth/gpt-oss-120b` (pre-released before March 15, 2026 ✅)
4. Attach dataset: `<username>/layer1-solvers`
5. Set GPU accelerator, internet OFF → Run All → Submit

## Author

Ali Lawal Muazu · March 2026
