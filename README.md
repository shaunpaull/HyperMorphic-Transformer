# HyperMorphic-Transformer
HyperMorphic Transformer

# Biomorphic Transformer — HyperMorphic Optimizer Suite

**One-file, reproducible benchmark** for evaluating a **Biomorphic (Φ/Ψ + εₕ + trust-blend)** optimizer on a
calibration‑sensitive **sequence modeling** task with a tiny **causal Transformer**.

> TL;DR: This repo shows how a biologically‑inspired optimizer (HyperMorphic, “HM”) improves **loss, calibration (ECE)**, and **trajectory stability** under noise & shift—using clean ablations, stats, sensitivity scans, and optional plots.

---

## ✨ What’s inside

- **`biomorphic_transformer.py`** — the entire project in **one script**:
  - **Model**: small causal **Transformer** (2 layers by default).
  - **Task**: synthetic next‑token prediction with **regime switch** + **domain‑shifted validation** (great for calibration stress‑tests).
  - **Optimizers**:
    - Baselines: **Adam**, **AdamClip**.
    - **HyperMorphic (HM)** variants:
      - `HM_Static` — Φ + Ψ + **εₕ** + trust‑blend (fixed params).
      - `HM_Bio` — as above **with biomorphic controller** (smooth schedules for εₕ, base \(b\), α, β, λ).
      - Ablations: `HM_no_eps` (no εₕ), `HM_phi_only` (Φ only), `HM_psi_only` (Ψ only).
  - **Metrics**: validation **loss**, **ECE** (Expected Calibration Error), **update alignment** (↑ good), **flip‑rate** (↓ good), **delta‑norm variance** (↓ good).
  - **Statistics**: **95% CI**, **paired t‑test** and **Wilcoxon** vs `HM_Bio`.
  - **Sensitivity scans**: sweep **(εₕ, b, α, β, λ)**; optional plots.

---

## 🧠 Core idea (Biological coherence)

The HyperMorphic pipeline maps raw gradients through two bounded transforms and a trust‑blend:

- **Φ (log‑compression)**: \( \Phi_{\varepsilon,b}(x)=\mathrm{sgn}(x)\,\log_b\!\big(1+\frac{|x|}{\varepsilon}\big) \)  
  Contracts large/low‑SNR signals; controlled by **εₕ** and base **b**.

- **Ψ (bounded fold)**: \( \Psi_{\alpha,\beta}(x)=\sin\!\big(\beta\,\tanh(\alpha x)\big) \)  
  Binds updates inside \([-1,1]\) with slope control via **α, β**.

- **Trust‑blend (λ)**: \( d_t \propto \lambda\,\hat u + (1-\lambda)\,d_{t-1} \)  
  Caps per‑step turning, reducing flip‑rate & variance.

- **Biomorphic controller**: Smoothly modulates \((\varepsilon,b,\alpha,\beta,\lambda)\) over time; acts like a **homeostatic governor**—more contractive in noisy phases, looser in stable phases.

**Why it helps:** Under noise and shift, HM acts like **adaptive temperature scaling** on logits (shrinks overconfidence) while **bounding curvature** of the optimization path → **ECE ↓ with stable loss**.

---

## ⚙️ Installation

```bash
# Python 3.9+ recommended
pip install torch pandas matplotlib
# (pandas/matplotlib are optional but used for CSVs and plots)
```

> CPU is fine. GPU is optional. No external datasets required.

---

## 🚀 Quick start

Run baselines + HM variants with **10 seeds**, compute **95% CIs** + **paired tests**, and write CSVs:

```bash
python biomorphic_transformer.py \
  --seeds 10 --epochs 3 --device cpu \
  --opts Adam AdamClip HM_Static HM_Bio HM_no_eps HM_phi_only HM_psi_only \
  --out_prefix HM_TinyTX
```

Outputs (CSV):
- `HM_TinyTX_PerSeed.csv` — per‑seed metrics
- `HM_TinyTX_Summary_95CI.csv` — mean ± 95% CI for each metric
- `HM_TinyTX_Stats.csv` — paired t / Wilcoxon vs **HM_Bio** on loss/ECE/alignment

---

## 🔎 Sensitivity scans & plots

Scan parameter ranges and build loss/ECE curves with 95% CIs:

```bash
python biomorphic_transformer.py \
  --seeds 6 --epochs 2 --device cpu \
  --do_scans --scan_params eps_h base_b alpha beta lam \
  --plots \
  --out_prefix HM_TinyTX
```

Additional figures (if `--plots`):
- `HM_TinyTX_loss_by_opt.png` — mean val loss by optimizer (+CI)
- `HM_TinyTX_ece_by_opt.png` — mean ECE by optimizer (+CI)
- `HM_TinyTX_loss_vs_<param>.png`, `HM_TinyTX_ece_vs_<param>.png` — sensitivity curves (+CI)

---

## 📐 Metrics (how they’re computed)

- **Loss** — average negative log‑likelihood on all but last timestep.
- **ECE** — 10‑bin reliability gap between accuracy & confidence. Lower is **better**.
- **Update alignment** — cosine similarity between successive parameter deltas (↑ smoother path).
- **Flip‑rate** — fraction of coordinates with sign flips between successive updates (↓ fewer reversals).
- **Delta‑norm variance** — variance of \(\|\Delta\theta\|\) over steps (↓ stabler magnitudes).

---

## 🧪 Ablations & baselines

| Label         | Description                                      |
|---------------|--------------------------------------------------|
| `Adam`        | Standard Adam                                    |
| `AdamClip`    | Adam + global grad clipping (1.0)                |
| `HM_Static`   | Φ + Ψ + εₕ + trust‑blend (fixed params)          |
| `HM_Bio`      | HM + **biomorphic controller** (adaptive params) |
| `HM_no_eps`   | HM without εₕ floor                              |
| `HM_phi_only` | HM with Φ only (no Ψ fold)                       |
| `HM_psi_only` | HM with Ψ only (no Φ compression)                |

**What to expect (qualitative):**
- **HM_Bio** typically improves **loss** and **ECE** vs baselines and `HM_Static`.
- Removing **εₕ** tends to **worsen ECE** (overconfidence returns).
- **Φ only / Ψ only** underperform **Φ+Ψ** (the combo is the point).
- Mid‑high **λ** reduces flip‑rate and increases alignment (bounded curvature).

> Your exact numbers will vary by seed; use the included **95% CIs** and **paired tests** for rigor.

---

## 🧩 Command reference (selected)

```bash
# change model size
--d_model 64 --d_ff 128 --n_heads 4 --n_layers 2

# dataset size
--n_train 2000 --n_val 1000 --seq_len 32 --vocab 16

# HM hyperparameters
--lr_hm 2.5e-3 --eps_h 2e-3 --base_b 2.0 --alpha 1.1 --beta 0.9 --lam 0.75

# choose which optimizers to run
--opts Adam HM_Bio

# plotting
--plots
```

Run `python biomorphic_transformer.py -h` for full CLI.

---

## 📊 Interpreting results

1. Open `*_Summary_95CI.csv` — compare **loss** and **ECE** means + **95% CI** across optimizers.  
2. Check `*_Stats.csv` — **paired t / Wilcoxon** vs `HM_Bio` (lower p‑values → stronger evidence).  
3. Inspect stability metrics:
   - **alignment ↑**, **flip‑rate ↓**, **delta‑norm variance ↓** → smoother learning dynamics.
4. If running scans, find the **εₕ** region where **ECE** dips (calibration sweet‑spot) without hurting loss.

---

## 🧪 Reproducibility notes

- **Deterministic**: seeds are set for Python, NumPy, and PyTorch; pass `--seeds N --base_seed S` for paired experiments.
- **CPU/GPU**: results should be similar on CPU; GPU nondeterminism can add tiny noise unless fully pinned.

---

## 🧠 Theory cheat‑sheet (why HM helps)

- **Contraction:** Φ∘Ψ has global Lipschitz constant \(L \le \alpha\beta/(\varepsilon\ln b)\) → more contraction when εₕ is larger or \(b\) is higher.  
- **Curvature bound:** trust‑blend caps per‑step turning by \( \arcsin(\lambda) \) → fewer sign flips & less variance.  
- **Calibration link:** bounded, adaptive shrinkage of logit steps behaves like **temperature scaling** where it matters → **ECE ↓** with decisions often unchanged.

For a full write‑up (statements & proof sketches), see the “Biological coherence” notes in the project discussion.

---

## ❓ FAQ

- **Do I need a dataset?** No — the script generates a synthetic regime‑switching corpus.
- **Can I plug HM into my model?** Yes — the HM optimizer is a drop‑in `torch.optim.Optimizer`‑style class.
- **Why ECE?** Because deployment safety needs calibrated probabilities, not just low loss.

---

## 📄 License

MIT. See your repository’s `LICENSE` file (recommended).

---

## 🙌 Citation

If you use this in academic or industrial work, please cite:

```
@software{BiomorphicTransformer2025,
  title   = {Biomorphic Transformer: HyperMorphic Optimizer Suite},
  author  = {Gerrard, Shaun Paul and Collaborators},
  year    = {2025},
  url     = {https://github.com/<your-org>/biomorphic-transformer}
}
```

---

## 🤝 Acknowledgments

Thanks to the communities exploring **calibration**, **robust optimization**, and **bio‑inspired control**—this project stands on many shoulders.

---

**Go build.** 🔧 **Go measure.** 📈 **Go calibrate.** 🎯
