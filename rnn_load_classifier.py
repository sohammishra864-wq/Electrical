"""RNN electrical load classification (CP / CC / CZ) on RL, RC, RLC (+R, L, C, LC) circuit data.

Run:  python rnn_load_classifier.py
Input:  load_dataset_baseline_v3.xlsx  (read-only)
Output: outputs/  -> figures (PNG), trained model (.pt), RNN_Load_Classification_Results.xlsx

Pipeline: circuit data -> preprocessing -> feature extraction -> RNN/LSTM/GRU -> evaluation -> Excel.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
from torch import nn
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import FormulaRule
from openpyxl.utils import get_column_letter

SEED = 42
HERE = Path(__file__).parent
DATA = HERE / "load_dataset_baseline_v3.xlsx"
OUT = HERE / "outputs"
FIG = OUT / "figures"
CLASSES = ["CP", "CC", "CZ"]          # dataset's third class is CZ (constant impedance) - see report
CLASS_NAME = {"CP": "Constant Power", "CC": "Constant Current", "CZ": "Constant Impedance"}
LEVELS = [0.90, 0.95, 1.00, 1.05]     # voltage-sweep test levels (L0..L3)
# validated categorical palette (dataviz reference, slots 1-3; all-pairs safe)
COL = {"CP": "#2a78d6", "CC": "#eb6834", "CZ": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "savefig.dpi": 130, "savefig.bbox": "tight", "axes.axisbelow": True})

UNITS = {"Vrms": "Vrms (V)", "Irms": "Irms (A)", "P": "Active Power P (W)", "Q1": "Reactive Power Q (var)",
         "S": "Apparent Power S (VA)", "PF": "Power Factor (-)", "phi_deg": "Phase Angle (deg)",
         "Zmag": "Impedance |Z| (ohm)", "frequency_Hz": "Frequency (Hz)", "THD_V": "THD_V (pu)",
         "THD_I": "THD_I (pu)", "condition": "Condition"}
for k, lv in enumerate(LEVELS):
    UNITS |= {f"V_L{k}": f"V @{lv:.0%} (V)", f"I_L{k}": f"I @{lv:.0%} (A)", f"P_L{k}": f"P @{lv:.0%} (W)"}


# ---------------------------------------------------------------- 1. data
def load_data():
    d = pd.read_excel(DATA, sheet_name="Dataset")
    s = pd.read_excel(DATA, sheet_name="Source link")[["sample_id", "frequency_used_Hz", "condition"]]
    d = d.merge(s.rename(columns={"frequency_used_Hz": "frequency_Hz"}), on="sample_id", validate="1:1")
    assert len(d) == 16800 and d.isna().sum().sum() == 0, "dataset shape/NaN check failed"
    assert set(d.behaviour) == set(CLASSES)
    return d


# ---------------------------------------------------------------- 2. circuit models
def series_circuit(R, L, C, f):
    """Series R-L-C branch, steady state: Z = R + j(wL - 1/wC). C=None -> no capacitor."""
    w = 2 * np.pi * f
    X = w * L - (0 if C is None else 1 / (w * C))
    Z = complex(R, X)
    return abs(Z), np.degrees(np.angle(Z))


def rlc_from_sample(row, kind):
    """Back out R, L, C from a sample's |Z| and phase angle.
    ponytail: RLC split uses X_L = 2|X|, X_C = |X| (inductive) or the mirror (capacitive) - any split
    with the same net X gives the same 50 Hz steady state; only the transient shape depends on it."""
    w = 2 * np.pi * row.frequency_Hz
    R = row.Zmag * np.cos(np.radians(row.phi_deg))
    X = row.Zmag * np.sin(np.radians(row.phi_deg))
    if kind == "RL":
        return R, abs(X) / w, None
    if kind == "RC":
        return R, 0.0, 1 / (w * abs(X))
    XL, XC = (2 * abs(X), abs(X)) if X >= 0 else (abs(X), 2 * abs(X))
    return R, XL / w, 1 / (w * XC)


def switch_on_transient(R, L, C, Vrms, f, t_end=0.1):
    """Switch-on current of a series R/L/C branch from KVL:  v(t) = R i + L di/dt + q/C,  i = dq/dt."""
    w, Vm = 2 * np.pi * f, np.sqrt(2) * Vrms
    if C is None:                                        # RL:  L di/dt = v - R i
        rhs = lambda t, y: [(Vm * np.sin(w * t) - R * y[0]) / L]
        y0 = [0.0]
    elif L == 0:                                         # RC:  R dq/dt = v - q/C
        rhs = lambda t, y: [(Vm * np.sin(w * t) - y[0] / C) / R]
        y0 = [0.0]
    else:                                                # RLC: state [q, i]
        rhs = lambda t, y: [y[1], (Vm * np.sin(w * t) - R * y[1] - y[0] / C) / L]
        y0 = [0.0, 0.0]
    t = np.linspace(0, t_end, 4000)
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t, max_step=1e-4, method="LSODA")
    if C is not None and L == 0:                         # RC: i = (v - q/C)/R
        i = (Vm * np.sin(w * t) - sol.y[0] / C) / R
    else:
        i = sol.y[-1]
    return t, Vm * np.sin(w * t), i


def waveform(Vrms, Irms, phi_deg, f, thd_v, thd_i, t):
    """Rebuild v(t), i(t) from the measured RMS / phase / THD (the dataset ships features, not raw samples).
    ponytail: THD spread over 3rd/5th/7th with fixed weights 0.7/0.5/0.3 - true harmonic split unknown."""
    w, phi = 2 * np.pi * f, np.radians(phi_deg)
    h = np.array([0.7, 0.5, 0.3]); h /= np.linalg.norm(h)
    v = np.sin(w * t) + sum(thd_v * a * np.sin(n * w * t) for a, n in zip(h, (3, 5, 7)))
    i = np.sin(w * t - phi) + sum(thd_i * a * np.sin(n * (w * t - phi)) for a, n in zip(h, (3, 5, 7)))
    v *= Vrms * np.sqrt(2) / np.sqrt(1 + thd_v ** 2)    # keep the RMS equal to the measured value
    i *= Irms * np.sqrt(2) / np.sqrt(1 + thd_i ** 2)
    return v, i


# ---------------------------------------------------------------- 3. features
def extract_features(d):
    """Per-load electrical features. Sweep exponents follow the ZIP model:
    I ~ V^nI, P ~ V^nP  ->  CP: nI=-1, nP=0 | CC: nI=0, nP=1 | CZ: nI=+1, nP=2."""
    lnV = np.log(d[[f"V_L{k}" for k in range(4)]].values)
    lnI = np.log(d[[f"I_L{k}" for k in range(4)]].values)
    lnP = np.log(d[[f"P_L{k}" for k in range(4)]].values)
    slope = lambda y: ((lnV - lnV.mean(1, keepdims=True)) * (y - y.mean(1, keepdims=True))).sum(1) / \
                      ((lnV - lnV.mean(1, keepdims=True)) ** 2).sum(1)
    f = d[["sample_id", "circuit", "condition", "behaviour", "Vrms", "Irms", "P", "Q1", "S", "PF",
           "phi_deg", "Zmag", "frequency_Hz", "THD_V", "THD_I"]].copy()
    f["dV_sweep_pct"] = 100 * (d.V_L3 - d.V_L0) / d.V_L2
    f["dI_sweep_pct"] = 100 * (d.I_L3 - d.I_L0) / d.I_L2
    f["dP_sweep_pct"] = 100 * (d.P_L3 - d.P_L0) / d.P_L2
    f["nI_current_exponent"] = slope(lnI)
    f["nP_power_exponent"] = slope(lnP)
    f["freq_dev_Hz"] = d.frequency_Hz - 50.0
    return f


def build_sequences(d):
    """RNN input: one time step per sweep level (90 -> 95 -> 100 -> 105 % voltage).
    Each step = [V, I, P, V/V_100, I/I_100, P/P_100]  ->  shape (N, 4, 6)."""
    steps = []
    for k in range(4):
        V, I, P = d[f"V_L{k}"].values, d[f"I_L{k}"].values, d[f"P_L{k}"].values
        steps.append(np.stack([V, I, P, V / d.V_L2.values, I / d.I_L2.values, P / d.P_L2.values], 1))
    return np.stack(steps, 1).astype(np.float32)


# ---------------------------------------------------------------- 4. model
class LoadRNN(nn.Module):
    """Input (4 steps x 6) -> 2-layer recurrent layer (64 units) -> Dense 32 ReLU -> Dense 3 (softmax in loss)."""
    def __init__(self, cell, n_in, hidden=64, layers=2, p=0.2):
        super().__init__()
        self.rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell](
            n_in, hidden, layers, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 3))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])                    # last time step summarises the whole sweep


def train(cell, Xtr, ytr, Xva, yva, epochs=150, patience=20, bs=128):
    torch.manual_seed(SEED)
    m = LoadRNN(cell, Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t, ytr_t, Xva_t, yva_t = map(torch.tensor, (Xtr, ytr, Xva, yva))
    hist, best, best_state, wait = [], np.inf, None, 0
    g = torch.Generator().manual_seed(SEED)
    for ep in range(1, epochs + 1):
        m.train()
        idx = torch.randperm(len(Xtr_t), generator=g)
        tl, tc = 0.0, 0
        for b in range(0, len(idx), bs):
            j = idx[b:b + bs]
            opt.zero_grad()
            out = m(Xtr_t[j]); loss = loss_fn(out, ytr_t[j]); loss.backward(); opt.step()
            tl += loss.item() * len(j); tc += (out.argmax(1) == ytr_t[j]).sum().item()
        m.eval()
        with torch.no_grad():
            vo = m(Xva_t); vl = loss_fn(vo, yva_t).item(); va = (vo.argmax(1) == yva_t).float().mean().item()
        hist.append({"model": cell, "epoch": ep, "train_loss": tl / len(Xtr_t), "val_loss": vl,
                     "train_acc": tc / len(Xtr_t), "val_acc": va})
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(best_state)                       # early stopping: keep best-validation weights
    return m, pd.DataFrame(hist)


def predict(m, X):
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(X)), 1).numpy()


def scores(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    acc = accuracy_score(y, p)
    return {"Accuracy (%)": 100 * acc, "Precision (%)": 100 * pr, "Recall (%)": 100 * rc,
            "F1-score (%)": 100 * f1, "Classification Error (%)": 100 * (1 - acc)}


# ---------------------------------------------------------------- 5. figures
def save(fig, name):
    p = FIG / name
    fig.savefig(p); plt.close(fig)
    return p


def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 4.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2)
    stages = [("Circuit\nSimulation", "RL / RC / RLC\n(+R, L, C, LC)"),
              ("Dataset\nGeneration", "16,800 loads\nvoltage sweep"),
              ("Data\nPreprocessing", "split 70/15/15\nscale on train"),
              ("Feature\nExtraction", "RMS, P, Q, S\nPF, f, nI, nP"),
              ("RNN\nClassifier", "4 steps x 6\nGRU/LSTM/RNN"),
              ("CP / CC / CZ\nClassification", "softmax\n3 classes"),
              ("Performance\nEvaluation", "Acc, P, R, F1\nconfusion"),
              ("Excel Tables\n& Graphs", "9 sheets\n+ figures")]
    w, gap = 1.35, 0.27
    for i, (title, sub) in enumerate(stages):
        x = 0.15 + i * (w + gap)
        hl = i == 4
        ax.add_patch(FancyBboxPatch((x, 1.5), w, 1.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#2a78d6" if hl else "#eef3fb", ec="#2a78d6", lw=1.5))
        ax.text(x + w / 2, 2.65, title, ha="center", va="center", fontsize=9.5, weight="bold",
                color="white" if hl else INK)
        ax.text(x + w / 2, 1.95, sub, ha="center", va="center", fontsize=8, color="white" if hl else INK2)
        if i < len(stages) - 1:
            ax.annotate("", (x + w + gap - 0.02, 2.3), (x + w + 0.02, 2.3),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5))
    ax.text(6.5, 3.75, "Project architecture - RNN electrical load classification", ha="center",
            fontsize=13, weight="bold")
    ax.text(6.5, 0.8, "RNN detail:  Input (4 sweep steps x [V, I, P, V/V100, I/I100, P/P100])  ->  "
            "Recurrent layer 1 (64)  ->  Recurrent layer 2 (64)  ->  Dense 32 + ReLU  ->  Dense 3 + Softmax",
            ha="center", fontsize=9, color=INK2)
    ax.text(6.5, 0.35, "Loss: cross-entropy   Optimizer: Adam (lr 0.001)   Early stopping on validation loss",
            ha="center", fontsize=9, color=INK2)
    return save(fig, "01_architecture.png")


def sweep_signals(row, cyc=3, n=128):
    """v(t), i(t), p(t) during the 90/95/100/105 % sweep, cyc cycles per level."""
    f = row.frequency_Hz
    ts, vs, is_ = [], [], []
    for k in range(4):
        t = np.arange(cyc * n) / (n * f)
        v, i = waveform(row[f"V_L{k}"], row[f"I_L{k}"], row.phi_deg, f, row.THD_V, row.THD_I, t)
        ts.append(t + k * cyc / f); vs.append(v); is_.append(i)
    t, v, i = map(np.concatenate, (ts, vs, is_))
    return t * 1000, v, i, v * i


def fig_sweeps(d):
    picks = {c: d[(d.circuit == "RLC") & (d.behaviour == c) & (d.condition == "normal")].iloc[0] for c in CLASSES}
    paths = []
    for qi, (qname, ylabel, fname) in enumerate([("Voltage", "v(t) (V)", "02_voltage_vs_time.png"),
                                                 ("Current", "i(t) (A)", "03_current_vs_time.png"),
                                                 ("Instantaneous power", "p(t) (W)", "04_power_vs_time.png")]):
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        top = axs[0].secondary_xaxis("top")
        top.set_xticks([30, 90, 150, 210], [f"supply {lv:.0%}" for lv in LEVELS]); top.tick_params(length=0)
        for ax, c in zip(axs, CLASSES):
            r = picks[c]
            t, v, i, p = sweep_signals(r)
            y = (v, i, p)[qi]
            ax.plot(t, y, color=COL[c], lw=1.2)
            if qi == 2:   # average power per level
                for k in range(4):
                    ax.hlines(r[f"P_L{k}"], k * 60, (k + 1) * 60, color=INK, lw=1.5, ls="--")
            for k in range(1, 4):
                ax.axvline(k * 60, color=INK2, lw=0.6)
            ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
            ax.set_title(f"{c} - {CLASS_NAME[c]}  (RLC load {r.sample_id})", loc="left", fontsize=10,
                         pad=22 if c == CLASSES[0] else 6)
        axs[-1].set_xlabel("Time (ms)")
        fig.suptitle(f"{qname} vs time during the voltage-sweep test" +
                     ("  (dashed = average active power P)" if qi == 2 else ""), weight="bold")
        fig.tight_layout()
        paths.append(save(fig, fname))
    return paths


def fig_waveforms(d):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, circ in zip(axs, ["RL", "RC", "RLC"]):
        r = d[(d.circuit == circ) & (d.condition == "normal")].iloc[0]
        t = np.linspace(0, 2 / r.frequency_Hz, 600)
        v, i = waveform(r.Vrms, r.Irms, r.phi_deg, r.frequency_Hz, r.THD_V, r.THD_I, t)
        ax.plot(t * 1000, v / v.max(), color=COL["CP"], label=f"Voltage (peak {v.max():.0f} V)")
        ax.plot(t * 1000, i / i.max(), color=COL["CC"], label=f"Current (peak {i.max():.1f} A)")
        ax.set_title(f"{circ} circuit  -  phase angle {r.phi_deg:+.1f} deg\n"
                     f"({'current lags' if r.phi_deg > 0 else 'current leads'} voltage)", fontsize=10)
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Normalised amplitude (pu of peak)")
        ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.suptitle("Voltage and current waveforms (2 cycles, 50 Hz)", weight="bold"); fig.tight_layout()
    return save(fig, "05_voltage_current_waveforms.png")


def fig_transients(d):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, circ in zip(axs, ["RL", "RC", "RLC"]):
        r = d[(d.circuit == circ) & (d.condition == "normal")].iloc[0]
        R, L, C = rlc_from_sample(r, circ)
        Zc, ph = series_circuit(R, L, C, r.frequency_Hz)
        assert abs(Zc - r.Zmag) / r.Zmag < 1e-6, "circuit model does not reproduce sample |Z|"
        t, v, i = switch_on_transient(R, L, C, r.Vrms, r.frequency_Hz)
        ax.plot(t * 1000, i, color=COL["CC"], lw=1.4, label="Current i(t)")
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Current (A)")
        lab = f"R={R:.1f} ohm" + (f", L={L*1000:.1f} mH" if L else "") + (f", C={C*1e6:.0f} uF" if C else "")
        ax.set_title(f"{circ} switch-on transient\n{lab}, |Z|={Zc:.1f} ohm", fontsize=9.5)
    fig.suptitle("Circuit simulation models: series RL / RC / RLC solved from KVL (solve_ivp)", weight="bold")
    fig.tight_layout()
    return save(fig, "06_circuit_transients.png")


def fig_features(f):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axs[0]
    for c in CLASSES:
        s = f[f.behaviour == c]
        ax.scatter(s.nI_current_exponent, s.nP_power_exponent, s=8, alpha=0.35, color=COL[c],
                   edgecolors="none", label=f"{c} - {CLASS_NAME[c]}")
    for c, (x, y) in {"CP": (-1, 0), "CC": (0, 1), "CZ": (1, 2)}.items():
        ax.scatter([x], [y], marker="X", s=120, color=COL[c], edgecolors=SURF, linewidths=2)
        ax.annotate(f"ideal {c}", (x, y), (8, -12), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Current exponent nI  (I ~ V^nI)"); ax.set_ylabel("Power exponent nP  (P ~ V^nP)")
    ax.set_title("How current and power respond to the voltage sweep", fontsize=10)
    ax.legend(fontsize=8, frameon=False, markerscale=2)
    ax = axs[1]
    cols = ["PF", "THD_I", "dI_sweep_pct", "nI_current_exponent"]
    names = ["Power factor", "THD_I (pu)", "dI over sweep (%)", "nI exponent"]
    z = (f[cols] - f[cols].mean()) / f[cols].std()
    pos = np.arange(len(cols))
    for k, c in enumerate(CLASSES):
        bp = ax.boxplot([z.loc[f.behaviour == c, col].clip(-4, 4) for col in cols], positions=pos + (k - 1) * 0.25,
                        widths=0.2, patch_artist=True, showfliers=False)
        for b in bp["boxes"]: b.set(facecolor=COL[c], edgecolor=COL[c], alpha=0.8)
        for el in ("medians", "whiskers", "caps"):
            for ln in bp[el]: ln.set(color=INK2, lw=1)
        ax.plot([], [], color=COL[c], lw=6, label=c)
    ax.set_xticks(pos, names); ax.set_ylabel("Standardised value (z-score)")
    ax.set_title("Feature distributions by class (only the sweep features separate them)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Feature analysis", weight="bold"); fig.tight_layout()
    return save(fig, "07_feature_analysis.png")


def fig_training(hist, best):
    paths = []
    for metric, ylabel, fname, title in [("acc", "Accuracy (%)", "08_rnn_accuracy.png", "RNN model accuracy"),
                                         ("loss", "Cross-entropy loss", "09_rnn_loss.png", "RNN training and validation loss")]:
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        for ax, (cell, h) in zip(axs, hist.groupby("model", sort=False)):
            k = 100 if metric == "acc" else 1
            ax.plot(h.epoch, k * h[f"train_{metric}"], color=COL["CP"], label="Training")
            ax.plot(h.epoch, k * h[f"val_{metric}"], color=COL["CC"], label="Validation")
            ax.set_title(f"{cell}" + ("  (selected)" if cell == best else ""), fontsize=10)
            ax.set_xlabel("Epoch")
            if metric == "acc": ax.set_ylim(80, 100.5)
        axs[0].set_ylabel(ylabel); axs[0].legend(frameon=False)
        fig.suptitle(title, weight="bold"); fig.tight_layout()
        paths.append(save(fig, fname))
    return paths


def fig_confusion(cm, title, fname):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    pct = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100); ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)", ha="center", va="center", fontsize=10,
                    color="white" if pct[i, j] > 60 else INK)
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class"); ax.set_title(title, fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)
    return save(fig, fname)


def fig_actual_vs_pred(res, n=120):
    r = res.head(n)
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ya = r.actual.map(CLASSES.index); yp = r.predicted.map(CLASSES.index)
    ax.scatter(range(n), ya, s=70, facecolors="none", edgecolors=INK2, lw=1.2, label="Actual")
    ax.scatter(range(n), yp, s=18, color=COL["CP"], label="Predicted")
    wrong = np.where(ya != yp)[0]
    ax.scatter(wrong, yp.iloc[wrong], s=60, marker="x", color="#e34948", label=f"Misclassified ({len(wrong)})")
    ax.set_yticks(range(3), CLASSES); ax.set_xlabel(f"Test sample index (first {n} of {len(res)})")
    ax.set_title("Actual vs predicted classification on unseen test loads", weight="bold", loc="left")
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0)); ax.set_ylim(-0.5, 2.5)
    return save(fig, "11_actual_vs_predicted.png")


def fig_comparison(model_tbl, per_class, by_group):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    x = np.arange(len(model_tbl)); vals = model_tbl["Accuracy (%)"].values
    ax.bar(x, vals, color=COL["CP"], width=0.55)
    for xi, v in zip(x, vals): ax.text(xi, v + 0.1, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.axhline(95, color="#e34948", ls="--", lw=1); ax.text(-0.4, 95.15, "95 % target", fontsize=8, color=INK2)
    ax.set_xticks(x, [m.split(" (")[0].replace("Logistic regression", "LogReg (features)") for m in model_tbl.Model],
                  fontsize=8.5); ax.set_ylim(90, 100.8); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Test accuracy (%)"); ax.set_title("Model comparison (test set)", fontsize=10)
    ax = axs[1]
    m = ["Precision (%)", "Recall (%)", "F1-score (%)"]
    for k, c in enumerate(CLASSES):
        ax.bar(np.arange(3) + (k - 1) * 0.27, per_class.loc[c, m], width=0.25, color=COL[c], label=c)
    ax.set_xticks(range(3), ["Precision", "Recall", "F1-score"]); ax.set_ylim(90, 100.8); ax.grid(axis="x", visible=False)
    ax.set_title("Per-class metrics (selected model)", fontsize=10); ax.legend(frameon=False, ncol=3, fontsize=8)
    ax = axs[2]
    g = by_group
    ax.barh(range(len(g)), g["Accuracy (%)"], color=COL["CZ"], height=0.6)
    for yi, v in enumerate(g["Accuracy (%)"]): ax.text(v + 0.05, yi, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(g)), g.Group, fontsize=8.5); ax.set_xlim(90, 101); ax.invert_yaxis()
    ax.set_xlabel("Test accuracy (%)"); ax.set_title("Accuracy by circuit and by condition", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Classification performance comparison", weight="bold"); fig.tight_layout()
    return save(fig, "12_performance_comparison.png")


# ---------------------------------------------------------------- 6. Excel
HDR_FILL = PatternFill("solid", fgColor="1F4E78")
HDR_FONT = Font(bold=True, color="FFFFFF")
THIN = Border(*[Side(style="thin", color="BFBFBF")] * 4)


def style_sheet(ws, title=None, header_row=1, widths=None):
    for c in ws[header_row]:
        c.fill, c.font, c.border = HDR_FILL, HDR_FONT, THIN
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[header_row].height = 32
    ws.freeze_panes = ws.cell(header_row + 1, 1)
    for i, col in enumerate(ws.iter_cols(min_row=header_row, max_row=min(ws.max_row, header_row + 50)), 1):
        w = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[get_column_letter(i)].width = (widths or {}).get(i, min(max(10, w * 0.9 + 2), 28))
    if ws.max_row > header_row:
        ws.auto_filter.ref = f"A{header_row}:{get_column_letter(ws.max_column)}{ws.max_row}"


def write_excel(path, frames, images):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, (df, startrow) in frames.items():
            if isinstance(df, list):        # several tables stacked on one sheet
                r = 0
                for caption, t in df:
                    pd.DataFrame({caption: []}).to_excel(xw, sheet_name=name, startrow=r, index=False)
                    t.to_excel(xw, sheet_name=name, startrow=r + 1, index=False)
                    r += len(t) + 4
            else:
                df.to_excel(xw, sheet_name=name, startrow=startrow, index=False)
        pd.DataFrame({"Graphical Analysis - all figures (also saved as PNG in outputs/figures)": []}).to_excel(
            xw, sheet_name="Graphical Analysis", index=False)

    wb = load_workbook(path)
    for name, (df, startrow) in frames.items():
        ws = wb[name]
        if isinstance(df, list):            # style each stacked table's caption + header
            r = 1
            for caption, t in df:
                ws.cell(r, 1).font = Font(bold=True, size=12, color="1F4E78")
                for c in ws[r + 1]:
                    if c.value is not None:
                        c.fill, c.font, c.border = HDR_FILL, HDR_FONT, THIN
                        c.alignment = Alignment(horizontal="center", wrap_text=True)
                for row in ws.iter_rows(min_row=r + 2, max_row=r + 1 + len(t), max_col=t.shape[1]):
                    for c in row:
                        c.border = THIN
                        if isinstance(c.value, float): c.number_format = "0.00"
                r += len(t) + 4
            for i in range(1, ws.max_column + 1):
                ws.column_dimensions[get_column_letter(i)].width = 22
        else:
            style_sheet(ws, header_row=startrow + 1)
            for row in ws.iter_rows(min_row=startrow + 2):
                for c in row:
                    if isinstance(c.value, float): c.number_format = "0.0000"
    # Actual vs Predicted: highlight misclassified rows
    ws = wb["Actual vs Predicted"]
    last = get_column_letter(ws.max_column)
    corr_col = get_column_letter([c.value for c in ws[1]].index("Correct") + 1)
    ws.conditional_formatting.add(f"A2:{last}{ws.max_row}", FormulaRule(
        formula=[f'${corr_col}2="NO"'], fill=PatternFill("solid", fgColor="F8CBAD")))
    # Confusion matrix image next to the tables
    ws = wb["Confusion Matrix"]
    ws.add_image(XLImage(str(images["cm"])), "F2")
    # Graphical Analysis: every figure, stacked with a caption
    ws = wb["Graphical Analysis"]
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    row = 3
    for caption, p in images["all"]:
        ws.cell(row, 1, caption).font = Font(bold=True, size=11)
        img = XLImage(str(p))
        scale = min(1.0, 1000 / img.width)
        img.width, img.height = img.width * scale, img.height * scale
        ws.add_image(img, f"A{row + 1}")
        row += int(img.height / 20) + 4
    wb.move_sheet("Graphical Analysis", offset=0)
    wb.move_sheet("Graphical Analysis", offset=len(wb.sheetnames) - 1)
    wb.save(path)


# ---------------------------------------------------------------- main
def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    FIG.mkdir(parents=True, exist_ok=True)
    d = load_data()
    print(f"Loaded {len(d)} loads:", d.groupby("behaviour").size().to_dict())

    # --- split FIRST (stratified on class x circuit x condition) -> no leakage from test into scaling/training
    strat = d.behaviour + "_" + d.circuit + "_" + d.condition
    idx_tr, idx_tmp = train_test_split(d.index, test_size=0.30, stratify=strat, random_state=SEED)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=strat[idx_tmp], random_state=SEED)
    assert not (set(idx_tr) & set(idx_va) | set(idx_tr) & set(idx_te) | set(idx_va) & set(idx_te))
    split = pd.Series("train", index=d.index); split[idx_va] = "validation"; split[idx_te] = "test"
    print("Split:", split.value_counts().to_dict())

    y = d.behaviour.map(CLASSES.index).values.astype(np.int64)
    seq = build_sequences(d)
    N, T, F = seq.shape
    scaler = StandardScaler().fit(seq[idx_tr].reshape(len(idx_tr), -1))      # fit on TRAIN only
    Xs = scaler.transform(seq.reshape(N, -1)).reshape(N, T, F).astype(np.float32)
    feats = extract_features(d)

    # --- train the three recurrent cells, choose by VALIDATION accuracy (test set untouched until the end)
    models, hists, val_acc = {}, [], {}
    for cell in ["RNN", "LSTM", "GRU"]:
        m, h = train(cell, Xs[idx_tr], y[idx_tr], Xs[idx_va], y[idx_va])
        models[cell], val_acc[cell] = m, accuracy_score(y[idx_va], predict(m, Xs[idx_va]).argmax(1))
        hists.append(h)
        print(f"{cell:5s} epochs={len(h):3d}  val_acc={100*val_acc[cell]:.2f}%")
    hist = pd.concat(hists, ignore_index=True)
    best = max(val_acc, key=val_acc.get)
    torch.save({"state_dict": models[best].state_dict(), "cell": best, "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_, "classes": CLASSES}, OUT / f"rnn_{best.lower()}_model.pt")

    # --- feature-based baseline (same split, scaler fit on train) for comparison
    fcols = ["nI_current_exponent", "nP_power_exponent", "dI_sweep_pct", "dP_sweep_pct", "PF", "THD_I", "THD_V"]
    fs = StandardScaler().fit(feats.loc[idx_tr, fcols])
    lr = LogisticRegression(max_iter=3000).fit(fs.transform(feats.loc[idx_tr, fcols]), y[idx_tr])
    lr_pred = lr.predict(fs.transform(feats.loc[idx_te, fcols]))

    # --- final evaluation on the unseen TEST set
    rows = []
    for cell in models:
        p = predict(models[cell], Xs[idx_te]).argmax(1)
        rows.append({"Model": f"{cell} (4x6 sequence)", "Selected": "YES" if cell == best else "",
                     "Val accuracy (%)": 100 * val_acc[cell], **scores(y[idx_te], p)})
    rows.append({"Model": "Logistic regression (features)", "Selected": "", "Val accuracy (%)": np.nan,
                 **scores(y[idx_te], lr_pred)})
    model_tbl = pd.DataFrame(rows)

    prob = predict(models[best], Xs[idx_te]); pred = prob.argmax(1); yt = y[idx_te]
    cm = confusion_matrix(yt, pred, labels=[0, 1, 2])
    rep = classification_report(yt, pred, target_names=CLASSES, output_dict=True, digits=4)
    per_class = pd.DataFrame({c: {"Precision (%)": 100 * rep[c]["precision"], "Recall (%)": 100 * rep[c]["recall"],
                                  "F1-score (%)": 100 * rep[c]["f1-score"], "Support (loads)": rep[c]["support"]}
                              for c in CLASSES}).T
    res = d.loc[idx_te, ["sample_id", "circuit", "condition"]].reset_index(drop=True)
    res["actual"] = [CLASSES[i] for i in yt]; res["predicted"] = [CLASSES[i] for i in pred]
    for k, c in enumerate(CLASSES): res[f"P({c})"] = prob[:, k].round(4)
    res["confidence"] = prob.max(1).round(4); res["Correct"] = np.where(yt == pred, "YES", "NO")

    grp = []
    for col, vals in [("circuit", ["R", "L", "C", "RL", "RC", "LC", "RLC"]), ("condition", ["normal", "tough"])]:
        for v in vals:
            s = res[res[col] == v]
            grp.append({"Group": f"{col}: {v}", "Test loads": len(s),
                        "Accuracy (%)": 100 * (s.actual == s.predicted).mean(),
                        "Errors": int((s.actual != s.predicted).sum())})
    by_group = pd.DataFrame(grp)

    s = scores(yt, pred)
    print(f"\nSelected model: {best}   TEST  " + "  ".join(f"{k}={v:.2f}" for k, v in s.items()))
    print("Confusion matrix (rows=actual CP,CC,CZ):\n", cm)

    # --- figures
    figs = [("Project architecture / block diagram", fig_architecture())]
    sw = fig_sweeps(d)
    figs += [("Voltage vs time (voltage-sweep test)", sw[0]), ("Current vs time", sw[1]), ("Power vs time", sw[2]),
             ("Voltage and current waveforms - RL, RC, RLC", fig_waveforms(d)),
             ("Circuit simulation models - switch-on transients", fig_transients(d)),
             ("Feature analysis", fig_features(feats))]
    tr = fig_training(hist, best)
    figs += [("RNN model accuracy", tr[0]), ("RNN training and validation loss", tr[1])]
    cm_png = fig_confusion(cm, f"Confusion matrix - {best} (test set, n={len(yt)})", "10_confusion_matrix.png")
    figs += [("Confusion matrix", cm_png), ("Actual vs predicted classification", fig_actual_vs_pred(res)),
             ("Classification performance comparison", fig_comparison(model_tbl, per_class, by_group))]

    # --- Excel
    ds_cols = ["sample_id", "behaviour", "circuit", "condition", "Vrms", "Irms", "P", "Q1", "S", "PF", "phi_deg",
               "Zmag", "frequency_Hz", "THD_V", "THD_I"] + [f"{q}_L{k}" for k in range(4) for q in "VIP"]
    ds = d[ds_cols].assign(split=split).rename(columns=UNITS | {"behaviour": "Load class", "circuit": "Circuit",
                                                                "sample_id": "Sample ID", "split": "Data split"})
    feat_out = feats.assign(split=split).rename(columns=UNITS | {
        "behaviour": "Load class", "circuit": "Circuit", "sample_id": "Sample ID", "split": "Data split",
        "dV_sweep_pct": "dV over sweep (%)", "dI_sweep_pct": "dI over sweep (%)", "dP_sweep_pct": "dP over sweep (%)",
        "nI_current_exponent": "Current exponent nI (-)", "nP_power_exponent": "Power exponent nP (-)",
        "freq_dev_Hz": "Frequency deviation (Hz)"})
    cm_tbl = pd.DataFrame(cm, columns=[f"Predicted {c}" for c in CLASSES]); cm_tbl.insert(0, "Actual", CLASSES)
    cm_pct = pd.DataFrame((cm / cm.sum(1, keepdims=True) * 100).round(2),
                          columns=[f"Predicted {c} (%)" for c in CLASSES]); cm_pct.insert(0, "Actual", CLASSES)
    summary = pd.DataFrame({"Item": ["Selected model", "Selection rule", "Train / validation / test loads",
                                     "Input shape", "Classes", "Target"] + list(s),
                            "Value": [best, "highest validation accuracy (test set not used)",
                                      f"{len(idx_tr)} / {len(idx_va)} / {len(idx_te)}", "4 time steps x 6 features",
                                      "CP, CC, CZ", ">= 95 % accuracy, < 5 % error"] + [round(v, 3) for v in s.values()]})
    pc = per_class.reset_index().rename(columns={"index": "Class"})
    hist_out = hist.assign(train_acc=100 * hist.train_acc, val_acc=100 * hist.val_acc).rename(columns={
        "model": "Model", "epoch": "Epoch", "train_loss": "Train loss", "val_loss": "Validation loss",
        "train_acc": "Train accuracy (%)", "val_acc": "Validation accuracy (%)"})
    frames = {
        "Complete Dataset": (ds, 0),
        "RL Dataset": (ds[ds.Circuit == "RL"], 0),
        "RC Dataset": (ds[ds.Circuit == "RC"], 0),
        "RLC Dataset": (ds[ds.Circuit == "RLC"], 0),
        "Feature Extraction": (feat_out, 0),
        "Actual vs Predicted": (res.rename(columns={"sample_id": "Sample ID", "circuit": "Circuit",
                                                    "condition": "Condition", "actual": "Actual class",
                                                    "predicted": "Predicted class", "confidence": "Confidence"}), 0),
        "RNN Performance Metrics": ([("Summary (selected model, unseen test set)", summary),
                                     ("Model comparison (test set)", model_tbl.round(3)),
                                     ("Per-class metrics (selected model)", pc.round(3)),
                                     ("Accuracy by circuit and condition (selected model)", by_group.round(3)),
                                     ("Training history (per epoch)", hist_out.round(5))], None),
        "Confusion Matrix": ([("Confusion matrix - counts (rows = actual)", cm_tbl),
                              ("Confusion matrix - row % (recall)", cm_pct)], None),
    }
    xlsx = OUT / "RNN_Load_Classification_Results.xlsx"
    write_excel(xlsx, frames, {"cm": cm_png, "all": figs})
    print(f"\nWrote {xlsx}\nFigures in {FIG}")


if __name__ == "__main__":
    main()
