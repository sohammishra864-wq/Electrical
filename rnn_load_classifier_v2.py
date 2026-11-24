"""RNN electrical load classification (CP / CC / CZ) on RL, RC, RLC (+R, L, C, LC) circuit data.  (version 2)

Self-contained: the RNN, its training (backpropagation through time + gradient descent), the data split,
scaling, metrics and circuit simulation are all written out here. Libraries used only for:
  numpy      - arrays and matrix arithmetic
  matplotlib - drawing graphs
  openpyxl   - reading / writing the .xlsx files

Run:    python rnn_load_classifier_v2.py
Input:  load_dataset_baseline_v3.xlsx  (read-only)
Output: outputs2/  -> figures (PNG), trained weights (.npz), RNN_Load_Classification_Results.xlsx
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import FormulaRule
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------- settings (all in one place)
SEED = 42
HIDDEN = 32                     # RNN hidden units
EPOCHS = 60                     # passes over the training set
BATCH = 64                      # loads per gradient step
LEARNING_RATES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
SEEDS = [1, 2, 3]               # each learning rate is trained 3 times with different random start weights

HERE = Path(__file__).parent
DATA = HERE / "load_dataset_baseline_v3.xlsx"
OUT = HERE / "outputs2"
FIG = OUT / "figures"
CLASSES = ["CP", "CC", "CZ"]    # dataset's third class is CZ (constant impedance) - see report
CLASS_NAME = {"CP": "Constant Power", "CC": "Constant Current", "CZ": "Constant Impedance"}
LEVELS = [0.90, 0.95, 1.00, 1.05]   # voltage-sweep test levels (L0..L3)
COL = {"CP": "#2a78d6", "CC": "#eb6834", "CZ": "#1baf7a"}   # validated palette slots 1-3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "savefig.dpi": 130, "savefig.bbox": "tight", "axes.axisbelow": True})

UNITS = {"sample_id": "Sample ID", "behaviour": "Load class", "circuit": "Circuit", "condition": "Condition",
         "Vrms": "Vrms (V)", "Irms": "Irms (A)", "P": "Active Power P (W)", "Q1": "Reactive Power Q (var)",
         "S": "Apparent Power S (VA)", "PF": "Power Factor (-)", "phi_deg": "Phase Angle (deg)",
         "Zmag": "Impedance |Z| (ohm)", "frequency_Hz": "Frequency (Hz)", "THD_V": "THD_V (pu)",
         "THD_I": "THD_I (pu)", "split": "Data split",
         "dV_sweep_pct": "dV over sweep (%)", "dI_sweep_pct": "dI over sweep (%)",
         "dP_sweep_pct": "dP over sweep (%)", "nI": "Current exponent nI (-)", "nP": "Power exponent nP (-)",
         "freq_dev_Hz": "Frequency deviation (Hz)"}
for k, lv in enumerate(LEVELS):
    UNITS |= {f"V_L{k}": f"V @{lv:.0%} (V)", f"I_L{k}": f"I @{lv:.0%} (A)", f"P_L{k}": f"P @{lv:.0%} (W)"}


# ---------------------------------------------------------------- 1. data
def read_sheet(wb, name):
    rows = wb[name].iter_rows(values_only=True)
    head = list(next(rows))
    return head, [r for r in rows if r[0] is not None]


def load_data():
    """Returns a dict: column name -> numpy array (one entry per load)."""
    wb = load_workbook(DATA, read_only=True)
    head, rows = read_sheet(wb, "Dataset")
    src_head, src_rows = read_sheet(wb, "Source link")
    wb.close()
    src = {r[0]: r for r in src_rows}
    fi, ci = src_head.index("frequency_used_Hz"), src_head.index("condition")
    cols = {h: [r[i] for r in rows] for i, h in enumerate(head)}
    cols["frequency_Hz"] = [src[s][fi] for s in cols["sample_id"]]
    cols["condition"] = [src[s][ci] for s in cols["sample_id"]]
    assert len(rows) == 16800 and all(v is not None for c in cols.values() for v in c), "missing values"
    d = {k: np.array(v) if isinstance(v[0], str) else np.array(v, dtype=float) for k, v in cols.items()}
    assert set(d["behaviour"]) == set(CLASSES)
    return d


def first(d, **match):
    """Index of the first load matching e.g. circuit='RLC', behaviour='CP'."""
    mask = np.ones(len(d["sample_id"]), bool)
    for k, v in match.items():
        mask &= d[k] == v
    return np.flatnonzero(mask)[0]


def row(d, i):
    return {k: v[i] for k, v in d.items()}


def stratified_split(groups, fracs=(0.70, 0.15, 0.15), seed=SEED):
    """Shuffle each group (class x circuit x condition) separately and cut it 70/15/15,
    so train, validation and test all contain the same mix."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for g in np.unique(groups):
        idx = rng.permutation(np.flatnonzero(groups == g))
        a, b = round(fracs[0] * len(idx)), round((fracs[0] + fracs[1]) * len(idx))
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    return np.sort(tr), np.sort(va), np.sort(te)


# ---------------------------------------------------------------- 2. circuit models
def series_circuit(R, L, C, f):
    """Series R-L-C branch, steady state: Z = R + j(wL - 1/wC). C=None -> no capacitor."""
    w = 2 * np.pi * f
    X = w * L - (0 if C is None else 1 / (w * C))
    Z = complex(R, X)
    return abs(Z), np.degrees(np.angle(Z))


def rlc_from_sample(r, kind):
    """Back out R, L, C from a sample's |Z| and phase angle.
    ponytail: RLC split uses X_L = 2|X|, X_C = |X| (inductive) or the mirror (capacitive) - any split
    with the same net X gives the same 50 Hz steady state; only the transient shape depends on it."""
    w = 2 * np.pi * r["frequency_Hz"]
    R = r["Zmag"] * np.cos(np.radians(r["phi_deg"]))
    X = r["Zmag"] * np.sin(np.radians(r["phi_deg"]))
    if kind == "RL":
        return R, abs(X) / w, None
    if kind == "RC":
        return R, 0.0, 1 / (w * abs(X))
    XL, XC = (2 * abs(X), abs(X)) if X >= 0 else (abs(X), 2 * abs(X))
    return R, XL / w, 1 / (w * XC)


def switch_on_transient(R, L, C, Vrms, f, t_end=0.1, dt=1e-5):
    """Switch-on current from KVL  v(t) = R i + L di/dt + q/C,  i = dq/dt,  stepped with Euler's method:
    new value = old value + (rate of change) x dt.  dt = 10 us is far smaller than every L/R and RC here."""
    n = int(t_end / dt)
    t = np.arange(n) * dt
    v = np.sqrt(2) * Vrms * np.sin(2 * np.pi * f * t)
    i = np.zeros(n)
    q = 0.0                                              # capacitor charge
    for k in range(n - 1):
        if L == 0:                                       # RC: no inductor, current set by R directly
            i[k] = (v[k] - q / C) / R
        else:                                            # RL or RLC
            vc = q / C if C else 0.0
            i[k + 1] = i[k] + (v[k] - R * i[k] - vc) / L * dt
        q += i[k] * dt
    if L == 0:
        i[-1] = (v[-1] - q / C) / R
    return t, v, i


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
    I ~ V^nI, P ~ V^nP  ->  CP: nI=-1, nP=0 | CC: nI=0, nP=1 | CZ: nI=+1, nP=2.
    Each exponent is the least-squares slope of ln(I) or ln(P) against ln(V) over the 4 sweep points."""
    lnV = np.log(np.stack([d[f"V_L{k}"] for k in range(4)], 1))
    dv = lnV - lnV.mean(1, keepdims=True)

    def slope(name):
        y = np.log(np.stack([d[f"{name}_L{k}"] for k in range(4)], 1))
        return (dv * (y - y.mean(1, keepdims=True))).sum(1) / (dv ** 2).sum(1)

    keep = ["sample_id", "circuit", "condition", "behaviour", "Vrms", "Irms", "P", "Q1", "S", "PF",
            "phi_deg", "Zmag", "frequency_Hz", "THD_V", "THD_I"]
    f = {k: d[k] for k in keep}
    f["dV_sweep_pct"] = 100 * (d["V_L3"] - d["V_L0"]) / d["V_L2"]
    f["dI_sweep_pct"] = 100 * (d["I_L3"] - d["I_L0"]) / d["I_L2"]
    f["dP_sweep_pct"] = 100 * (d["P_L3"] - d["P_L0"]) / d["P_L2"]
    f["nI"], f["nP"] = slope("I"), slope("P")
    f["freq_dev_Hz"] = d["frequency_Hz"] - 50.0
    return f


def build_sequences(d):
    """RNN input: one time step per sweep level (90 -> 95 -> 100 -> 105 % voltage).
    Each step = [V, I, P, V/V_100, I/I_100, P/P_100]  ->  shape (N, 4, 6)."""
    steps = [np.stack([d[f"V_L{k}"], d[f"I_L{k}"], d[f"P_L{k}"],
                       d[f"V_L{k}"] / d["V_L2"], d[f"I_L{k}"] / d["I_L2"], d[f"P_L{k}"] / d["P_L2"]], 1)
             for k in range(4)]
    return np.stack(steps, 1)


# ---------------------------------------------------------------- 4. the RNN (written from scratch)
class SimpleRNN:
    """One recurrent layer + one output layer.

        h_0 = 0
        h_t = tanh(x_t Wx + h_(t-1) Wh + bh)        t = 1..4   (one step per sweep level)
        s   = h_4 Wy + by                           3 scores, one per class
        p   = softmax(s) = exp(s) / sum(exp(s))     class probabilities
        loss = -ln p[true class]                    cross-entropy
    """

    def __init__(self, n_in, n_hidden, n_out, seed):
        rng = np.random.default_rng(seed)        # small random start weights, scaled by 1/sqrt(inputs)
        self.Wx = rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden))
        self.Wh = rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_hidden))
        self.bh = np.zeros(n_hidden)
        self.Wy = rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_out))
        self.by = np.zeros(n_out)

    def params(self):
        return [self.Wx, self.Wh, self.bh, self.Wy, self.by]

    def forward(self, X):
        H = [np.zeros((len(X), self.Wh.shape[0]))]
        for t in range(X.shape[1]):
            H.append(np.tanh(X[:, t] @ self.Wx + H[-1] @ self.Wh + self.bh))
        s = H[-1] @ self.Wy + self.by
        e = np.exp(s - s.max(1, keepdims=True))          # subtract max: same result, no overflow
        return e / e.sum(1, keepdims=True), H

    def loss_and_grads(self, X, y):
        """Backpropagation through time: apply the chain rule backwards from the loss to every weight."""
        N = len(X)
        P, H = self.forward(X)
        loss = -np.log(P[np.arange(N), y] + 1e-12).mean()
        ds = P.copy(); ds[np.arange(N), y] -= 1; ds /= N   # d loss / d scores = p - one_hot(y)
        gWy, gby = H[-1].T @ ds, ds.sum(0)
        dh = ds @ self.Wy.T                                # d loss / d h_4
        gWx, gWh, gbh = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.bh)
        for t in range(X.shape[1], 0, -1):                 # step 4 -> 1
            da = dh * (1 - H[t] ** 2)                      # through tanh:  d tanh(a)/da = 1 - tanh(a)^2
            gWx += X[:, t - 1].T @ da
            gWh += H[t - 1].T @ da
            gbh += da.sum(0)
            dh = da @ self.Wh.T                            # pass the error to the previous time step
        return loss, [gWx, gWh, gbh, gWy, gby]

    def predict_proba(self, X):
        return self.forward(X)[0]


def gradient_check():
    """Self-test: the hand-derived gradients must match numerical slopes (loss(w+e) - loss(w-e)) / 2e."""
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(5, 4, 6)), rng.integers(0, 3, 5)
    m = SimpleRNN(6, 8, 3, seed=0)
    _, grads = m.loss_and_grads(X, y)
    for p, g in zip(m.params(), grads):
        for _ in range(5):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p[i]
            p[i] = old + 1e-5; lp = m.loss_and_grads(X, y)[0]
            p[i] = old - 1e-5; lm = m.loss_and_grads(X, y)[0]
            p[i] = old
            num = (lp - lm) / 2e-5
            assert abs(num - g[i]) < 1e-7 + 1e-4 * abs(num), f"gradient mismatch {num} vs {g[i]}"


def evaluate(m, X, y):
    P = m.predict_proba(X)
    return -np.log(P[np.arange(len(y)), y] + 1e-12).mean(), (P.argmax(1) == y).mean()


def train(Xtr, ytr, Xva, yva, lr, seed):
    """Mini-batch gradient descent: for each batch, weight <- weight - lr x gradient.
    After every epoch the validation loss is checked; the weights from the best epoch are kept."""
    m = SimpleRNN(Xtr.shape[2], HIDDEN, len(CLASSES), seed)
    rng = np.random.default_rng(seed)
    hist, best_loss, best_w = [], np.inf, None
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(len(Xtr))
        for b in range(0, len(order), BATCH):
            j = order[b:b + BATCH]
            _, grads = m.loss_and_grads(Xtr[j], ytr[j])
            for p, g in zip(m.params(), grads):
                p -= lr * g
        tl, ta = evaluate(m, Xtr, ytr)
        vl, va = evaluate(m, Xva, yva)
        hist.append({"lr": lr, "seed": seed, "epoch": ep, "train_loss": tl, "val_loss": vl,
                     "train_acc": ta, "val_acc": va})
        if not np.isfinite(vl):                       # training blew up at this learning rate
            break
        if vl < best_loss:
            best_loss, best_w = vl, [p.copy() for p in m.params()]
    if best_w is not None:
        for p, w in zip(m.params(), best_w):
            p[...] = w
    return m, hist


# ---------------------------------------------------------------- 5. metrics (written from scratch)
def confusion(y, p, k=3):
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, p), 1)                         # rows = actual, columns = predicted
    return cm


def metrics(cm):
    """Accuracy = correct / all.   For each class:  precision = TP / predicted-as-class,
    recall = TP / actually-class,  F1 = 2PR / (P + R).  Macro = plain average over the 3 classes."""
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(len(tp)), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(len(tp)), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0)
    acc = tp.sum() / cm.sum()
    return {"Accuracy (%)": 100 * acc, "Precision (%)": 100 * prec.mean(), "Recall (%)": 100 * rec.mean(),
            "F1-score (%)": 100 * f1.mean(), "Classification Error (%)": 100 * (1 - acc)}, prec, rec, f1


# ---------------------------------------------------------------- 6. figures
def save(fig, name):
    p = FIG / name
    fig.savefig(p); plt.close(fig)
    return p


def fig_architecture(lr):
    fig, ax = plt.subplots(figsize=(13, 4.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2)
    stages = [("Circuit\nSimulation", "RL / RC / RLC\n(+R, L, C, LC)"),
              ("Dataset\nGeneration", "16,800 loads\nvoltage sweep"),
              ("Data\nPreprocessing", "split 70/15/15\nscale on train"),
              ("Feature\nExtraction", "RMS, P, Q, S\nPF, f, nI, nP"),
              ("RNN\nClassifier", "4 steps x 6\n32 tanh units"),
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
            "Recurrent layer h_t = tanh(x_t Wx + h_(t-1) Wh + bh), 32 units  ->  Output layer (3) + Softmax",
            ha="center", fontsize=9, color=INK2)
    ax.text(6.5, 0.35, f"Loss: cross-entropy   Training: backpropagation through time + mini-batch gradient descent "
            f"(learning rate {lr})   Best-validation epoch kept", ha="center", fontsize=9, color=INK2)
    return save(fig, "01_architecture.png")


def sweep_signals(r, cyc=3, n=128):
    """v(t), i(t), p(t) during the 90/95/100/105 % sweep, cyc cycles per level."""
    f = r["frequency_Hz"]
    ts, vs, is_ = [], [], []
    for k in range(4):
        t = np.arange(cyc * n) / (n * f)
        v, i = waveform(r[f"V_L{k}"], r[f"I_L{k}"], r["phi_deg"], f, r["THD_V"], r["THD_I"], t)
        ts.append(t + k * cyc / f); vs.append(v); is_.append(i)
    t, v, i = map(np.concatenate, (ts, vs, is_))
    return t * 1000, v, i, v * i


def fig_sweeps(d):
    picks = {c: row(d, first(d, circuit="RLC", behaviour=c, condition="normal")) for c in CLASSES}
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
            ax.plot(t, (v, i, p)[qi], color=COL[c], lw=1.2)
            if qi == 2:   # average power per level
                for k in range(4):
                    ax.hlines(r[f"P_L{k}"], k * 60, (k + 1) * 60, color=INK, lw=1.5, ls="--")
            for k in range(1, 4):
                ax.axvline(k * 60, color=INK2, lw=0.6)
            ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
            ax.set_title(f"{c} - {CLASS_NAME[c]}  (RLC load {r['sample_id']})", loc="left", fontsize=10,
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
        r = row(d, first(d, circuit=circ, condition="normal"))
        t = np.linspace(0, 2 / r["frequency_Hz"], 600)
        v, i = waveform(r["Vrms"], r["Irms"], r["phi_deg"], r["frequency_Hz"], r["THD_V"], r["THD_I"], t)
        ax.plot(t * 1000, v / v.max(), color=COL["CP"], label=f"Voltage (peak {v.max():.0f} V)")
        ax.plot(t * 1000, i / i.max(), color=COL["CC"], label=f"Current (peak {i.max():.1f} A)")
        ax.set_title(f"{circ} circuit  -  phase angle {r['phi_deg']:+.1f} deg\n"
                     f"({'current lags' if r['phi_deg'] > 0 else 'current leads'} voltage)", fontsize=10)
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Normalised amplitude (pu of peak)")
        ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.suptitle("Voltage and current waveforms (2 cycles, 50 Hz)", weight="bold"); fig.tight_layout()
    return save(fig, "05_voltage_current_waveforms.png")


def fig_transients(d):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, circ in zip(axs, ["RL", "RC", "RLC"]):
        r = row(d, first(d, circuit=circ, condition="normal"))
        R, L, C = rlc_from_sample(r, circ)
        Zc, _ = series_circuit(R, L, C, r["frequency_Hz"])
        assert abs(Zc - r["Zmag"]) / r["Zmag"] < 1e-6, "circuit model does not reproduce sample |Z|"
        t, v, i = switch_on_transient(R, L, C, r["Vrms"], r["frequency_Hz"])
        ax.plot(t * 1000, i, color=COL["CC"], lw=1.4)
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Current (A)")
        lab = f"R={R:.1f} ohm" + (f", L={L*1000:.1f} mH" if L else "") + (f", C={C*1e6:.0f} uF" if C else "")
        ax.set_title(f"{circ} switch-on transient\n{lab}, |Z|={Zc:.1f} ohm", fontsize=9.5)
    fig.suptitle("Circuit simulation models: series RL / RC / RLC from KVL (Euler method, dt = 10 us)",
                 weight="bold")
    fig.tight_layout()
    return save(fig, "06_circuit_transients.png")


def fig_features(f):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axs[0]
    for c in CLASSES:
        s = f["behaviour"] == c
        ax.scatter(f["nI"][s], f["nP"][s], s=8, alpha=0.35, color=COL[c], edgecolors="none",
                   label=f"{c} - {CLASS_NAME[c]}")
    for c, (x, y) in {"CP": (-1, 0), "CC": (0, 1), "CZ": (1, 2)}.items():
        ax.scatter([x], [y], marker="X", s=120, color=COL[c], edgecolors=SURF, linewidths=2)
        ax.annotate(f"ideal {c}", (x, y), (8, -12), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Current exponent nI  (I ~ V^nI)"); ax.set_ylabel("Power exponent nP  (P ~ V^nP)")
    ax.set_title("How current and power respond to the voltage sweep", fontsize=10)
    ax.legend(fontsize=8, frameon=False, markerscale=2)
    ax = axs[1]
    cols, names = ["PF", "THD_I", "dI_sweep_pct", "nI"], ["Power factor", "THD_I (pu)", "dI over sweep (%)", "nI exponent"]
    pos = np.arange(len(cols))
    for k, c in enumerate(CLASSES):
        s = f["behaviour"] == c
        data = [np.clip((f[col][s] - f[col].mean()) / f[col].std(), -4, 4) for col in cols]
        bp = ax.boxplot(data, positions=pos + (k - 1) * 0.25, widths=0.2, patch_artist=True, showfliers=False)
        for b in bp["boxes"]: b.set(facecolor=COL[c], edgecolor=COL[c], alpha=0.8)
        for el in ("medians", "whiskers", "caps"):
            for ln in bp[el]: ln.set(color=INK2, lw=1)
        ax.plot([], [], color=COL[c], lw=6, label=c)
    ax.set_xticks(pos, names); ax.set_ylabel("Standardised value (z-score)")
    ax.set_title("Feature distributions by class (only the sweep features separate them)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Feature analysis", weight="bold"); fig.tight_layout()
    return save(fig, "07_feature_analysis.png")


def fig_training(h, lr):
    ep = [r["epoch"] for r in h]
    paths = []
    for metric, ylabel, fname, title in [("acc", "Accuracy (%)", "08_rnn_accuracy.png", "RNN model accuracy"),
                                         ("loss", "Cross-entropy loss", "09_rnn_loss.png",
                                          "RNN training and validation loss")]:
        k = 100 if metric == "acc" else 1
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ep, [k * r[f"train_{metric}"] for r in h], color=COL["CP"], label="Training")
        ax.plot(ep, [k * r[f"val_{metric}"] for r in h], color=COL["CC"], label="Validation")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend(frameon=False)
        if metric == "acc": ax.set_ylim(80, 100.5)
        ax.set_title(f"{title}  (selected model, learning rate {lr})", weight="bold")
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


def fig_actual_vs_pred(yt, pred, n=120):
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ya, yp = yt[:n], pred[:n]
    ax.scatter(range(n), ya, s=70, facecolors="none", edgecolors=INK2, lw=1.2, label="Actual")
    ax.scatter(range(n), yp, s=18, color=COL["CP"], label="Predicted")
    wrong = np.flatnonzero(ya != yp)
    ax.scatter(wrong, yp[wrong], s=60, marker="x", color="#e34948", label=f"Misclassified ({len(wrong)})")
    ax.set_yticks(range(3), CLASSES); ax.set_xlabel(f"Test sample index (first {n} of {len(yt)})")
    ax.set_title("Actual vs predicted classification on unseen test loads", weight="bold", loc="left")
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0)); ax.set_ylim(-0.5, 2.5)
    return save(fig, "11_actual_vs_predicted.png")


def fig_lr_sweep(lr_tbl, hist_all, best_lr):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axs[0]
    lrs = [r["Learning rate"] for r in lr_tbl]
    for key, col, lab in [("Val acc mean (%)", COL["CC"], "Validation"), ("Test acc mean (%)", COL["CP"], "Test")]:
        m = np.array([r[key] for r in lr_tbl]); s = np.array([r[key.replace("mean", "std")] for r in lr_tbl])
        ax.errorbar(lrs, m, yerr=s, color=col, marker="o", ms=7, capsize=4, label=f"{lab} (mean +/- std, 3 runs)")
    ax.axvline(best_lr, color=INK2, ls=":", lw=1.2)
    ax.text(best_lr, ax.get_ylim()[0], f" selected {best_lr}", fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel("Learning rate"); ax.set_ylabel("Accuracy (%)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Accuracy vs learning rate", fontsize=10)
    ax = axs[1]
    blues = plt.cm.Blues(np.linspace(0.35, 1.0, len(lrs)))
    for c, lr in zip(blues, lrs):
        h = [r for r in hist_all if r["lr"] == lr and r["seed"] == SEEDS[0]]
        ax.plot([r["epoch"] for r in h], [r["val_loss"] for r in h], color=c, lw=1.5, label=f"lr {lr}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss (log scale)"); ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title(f"Validation loss per learning rate (seed {SEEDS[0]})", fontsize=10)
    fig.suptitle("Learning-rate experiment", weight="bold"); fig.tight_layout()
    return save(fig, "13_learning_rate_sweep.png")


def fig_comparison(lr_tbl, prec, rec, f1, groups, best_lr):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    x = np.arange(len(lr_tbl))
    vals = np.array([r["Test acc mean (%)"] for r in lr_tbl]); err = [r["Test acc std (%)"] for r in lr_tbl]
    ax.bar(x, vals, yerr=err, capsize=3, width=0.6,
           color=[COL["CC"] if r["Learning rate"] == best_lr else COL["CP"] for r in lr_tbl])
    for xi, v in zip(x, vals): ax.text(xi, v + 0.25, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(95, color="#e34948", ls="--", lw=1); ax.text(-0.4, 95.15, "95 % target", fontsize=8, color=INK2)
    ax.set_xticks(x, [str(r["Learning rate"]) for r in lr_tbl]); ax.set_ylim(90, 100.8)
    ax.set_xlabel("Learning rate (orange = selected)"); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Test accuracy (%)"); ax.set_title("Learning-rate comparison (test, mean of 3 runs)", fontsize=10)
    ax = axs[1]
    for k, c in enumerate(CLASSES):
        ax.bar(np.arange(3) + (k - 1) * 0.27, [100 * prec[k], 100 * rec[k], 100 * f1[k]], width=0.25,
               color=COL[c], label=c)
    ax.set_xticks(range(3), ["Precision", "Recall", "F1-score"]); ax.set_ylim(90, 100.8)
    ax.grid(axis="x", visible=False)
    ax.set_title("Per-class metrics (selected model)", fontsize=10); ax.legend(frameon=False, ncol=3, fontsize=8)
    ax = axs[2]
    acc = [r["Accuracy (%)"] for r in groups]
    ax.barh(range(len(groups)), acc, color=COL["CZ"], height=0.6)
    for yi, v in enumerate(acc): ax.text(v + 0.05, yi, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(groups)), [r["Group"] for r in groups], fontsize=8.5); ax.set_xlim(90, 101)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Test accuracy (%)"); ax.set_title("Accuracy by circuit and by condition", fontsize=10)
    fig.suptitle("Classification performance comparison", weight="bold"); fig.tight_layout()
    return save(fig, "12_performance_comparison.png")


# ---------------------------------------------------------------- 7. Excel
HDR_FILL, HDR_FONT = PatternFill("solid", fgColor="1F4E78"), Font(bold=True, color="FFFFFF")
THIN = Border(*[Side(style="thin", color="BFBFBF")] * 4)
CAPTION = Font(bold=True, size=12, color="1F4E78")


def py(v):
    """numpy scalar -> plain Python value for openpyxl."""
    return v.item() if hasattr(v, "item") else v


def put_table(ws, r, header, rows, caption=None, borders=True):
    """Write a captioned, formatted table starting at row r; returns the next free row."""
    if caption:
        ws.cell(r, 1, caption).font = CAPTION; r += 1
    for j, h in enumerate(header, 1):
        c = ws.cell(r, j, h)
        c.fill, c.font, c.border = HDR_FILL, HDR_FONT, THIN
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[r].height = 32
    n = 0
    for i, values in enumerate(rows, r + 1):
        for j, v in enumerate(values, 1):
            c = ws.cell(i, j, py(v))
            if isinstance(c.value, float): c.number_format = "0.0000"
            if borders: c.border = THIN
        n += 1
    return r + n + 3


def data_sheet(wb, name, cols, table, idx):
    ws = wb.create_sheet(name)
    put_table(ws, 1, [UNITS.get(c, c) for c in cols], zip(*[table[c][idx].tolist() for c in cols]), borders=False)
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(idx) + 1}"
    for j in range(1, len(cols) + 1):
        ws.column_dimensions[get_column_letter(j)].width = 14
    return ws


def dict_rows(rows, keys):
    return [[r[k] for k in keys] for r in rows]


# ---------------------------------------------------------------- main
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    gradient_check()
    print("Gradient check passed (hand-written backpropagation matches numerical slopes).")
    d = load_data()
    n = len(d["sample_id"])
    print(f"Loaded {n} loads:", {c: int((d['behaviour'] == c).sum()) for c in CLASSES})

    # --- split FIRST -> nothing from validation/test influences scaling or training
    groups = np.char.add(np.char.add(d["behaviour"], d["circuit"]), d["condition"])
    tr, va, te = stratified_split(groups)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    split = np.full(n, "train", dtype=object); split[va] = "validation"; split[te] = "test"
    print(f"Split: train={len(tr)} validation={len(va)} test={len(te)}")

    y = np.array([CLASSES.index(b) for b in d["behaviour"]])
    seq = build_sequences(d)
    mean, std = seq[tr].mean(0), seq[tr].std(0)              # z-score using TRAINING loads only
    std[std == 0] = 1.0                                      # ratios at the 100 % step are always 1 -> constant
    X = (seq - mean) / std
    feats = extract_features(d)

    # --- learning-rate experiment: every lr x 3 seeds; choose by VALIDATION accuracy (test untouched)
    runs, hist_all = [], []
    for lr in LEARNING_RATES:
        for seed in SEEDS:
            m, h = train(X[tr], y[tr], X[va], y[va], lr, seed)
            _, vacc = evaluate(m, X[va], y[va])
            _, tacc = evaluate(m, X[te], y[te])
            runs.append({"lr": lr, "seed": seed, "model": m, "hist": h, "val_acc": vacc, "test_acc": tacc,
                         "best_epoch": int(np.nanargmin([r["val_loss"] for r in h])) + 1})
            hist_all += h
            print(f"lr={lr:.2f} seed={seed}  best epoch={runs[-1]['best_epoch']:2d}  "
                  f"val={100*vacc:.2f}%  test={100*tacc:.2f}%", flush=True)
    lr_tbl = []
    for lr in LEARNING_RATES:
        rs = [r for r in runs if r["lr"] == lr]
        v, t = 100 * np.array([r["val_acc"] for r in rs]), 100 * np.array([r["test_acc"] for r in rs])
        lr_tbl.append({"Learning rate": lr, "Val acc mean (%)": v.mean(), "Val acc std (%)": v.std(),
                       "Test acc mean (%)": t.mean(), "Test acc std (%)": t.std(),
                       "Test error mean (%)": 100 - t.mean(),
                       "Best epoch (mean)": np.mean([r["best_epoch"] for r in rs])})
    best_lr = max(lr_tbl, key=lambda r: r["Val acc mean (%)"])["Learning rate"]
    best = max((r for r in runs if r["lr"] == best_lr), key=lambda r: r["val_acc"])
    m = best["model"]
    np.savez(OUT / "rnn_model_weights.npz", Wx=m.Wx, Wh=m.Wh, bh=m.bh, Wy=m.Wy, by=m.by,
             scale_mean=mean, scale_std=std, learning_rate=best_lr, classes=CLASSES)

    # --- final evaluation on the unseen TEST set
    prob = m.predict_proba(X[te]); pred = prob.argmax(1); yt = y[te]
    cm = confusion(yt, pred)
    s, prec, rec, f1 = metrics(cm)
    grp = []
    for col, vals in [("circuit", ["R", "L", "C", "RL", "RC", "LC", "RLC"]), ("condition", ["normal", "tough"])]:
        for v in vals:
            k = d[col][te] == v
            grp.append({"Group": f"{col}: {v}", "Test loads": int(k.sum()),
                        "Accuracy (%)": 100 * (pred[k] == yt[k]).mean(), "Errors": int((pred[k] != yt[k]).sum())})
    print(f"\nSelected: learning rate {best_lr}, seed {best['seed']}   TEST  " +
          "  ".join(f"{k}={v:.2f}" for k, v in s.items()))
    print("Confusion matrix (rows=actual CP,CC,CZ):\n", cm)

    # --- figures
    figs = [("Project architecture / block diagram", fig_architecture(best_lr))]
    sw = fig_sweeps(d)
    figs += [("Voltage vs time (voltage-sweep test)", sw[0]), ("Current vs time", sw[1]), ("Power vs time", sw[2]),
             ("Voltage and current waveforms - RL, RC, RLC", fig_waveforms(d)),
             ("Circuit simulation models - switch-on transients", fig_transients(d)),
             ("Feature analysis", fig_features(feats))]
    tr_figs = fig_training(best["hist"], best_lr)
    figs += [("RNN model accuracy", tr_figs[0]), ("RNN training and validation loss", tr_figs[1])]
    cm_png = fig_confusion(cm, f"Confusion matrix - RNN (test set, n={len(yt)})", "10_confusion_matrix.png")
    figs += [("Confusion matrix", cm_png), ("Actual vs predicted classification", fig_actual_vs_pred(yt, pred)),
             ("Classification performance comparison", fig_comparison(lr_tbl, prec, rec, f1, grp, best_lr)),
             ("Learning-rate experiment", fig_lr_sweep(lr_tbl, hist_all, best_lr))]

    # --- Excel
    wb = Workbook(); wb.remove(wb.active)
    d["split"] = split; feats["split"] = split
    ds_cols = ["sample_id", "behaviour", "circuit", "condition", "Vrms", "Irms", "P", "Q1", "S", "PF", "phi_deg",
               "Zmag", "frequency_Hz", "THD_V", "THD_I"] + [f"{q}_L{k}" for k in range(4) for q in "VIP"] + ["split"]
    data_sheet(wb, "Complete Dataset", ds_cols, d, np.arange(n))
    for circ in ["RL", "RC", "RLC"]:
        data_sheet(wb, f"{circ} Dataset", ds_cols, d, np.flatnonzero(d["circuit"] == circ))
    data_sheet(wb, "Feature Extraction", list(feats), feats, np.arange(n))

    ws = wb.create_sheet("Actual vs Predicted")
    rows = [[d["sample_id"][i], d["circuit"][i], d["condition"][i], CLASSES[a], CLASSES[p_],
             *np.round(pr, 4), round(float(pr.max()), 4), "YES" if a == p_ else "NO"]
            for i, a, p_, pr in zip(te, yt, pred, prob)]
    put_table(ws, 1, ["Sample ID", "Circuit", "Condition", "Actual class", "Predicted class",
                      "P(CP)", "P(CC)", "P(CZ)", "Confidence", "Correct"], rows, borders=False)
    ws.freeze_panes = "A2"; ws.auto_filter.ref = f"A1:J{len(rows) + 1}"
    ws.conditional_formatting.add(f"A2:J{len(rows) + 1}", FormulaRule(
        formula=['$J2="NO"'], fill=PatternFill("solid", fgColor="F8CBAD")))
    for j in range(1, 11): ws.column_dimensions[get_column_letter(j)].width = 15

    ws = wb.create_sheet("RNN Performance Metrics")
    summary = [["Model", f"Simple RNN, 1 recurrent layer, {HIDDEN} tanh units (written from scratch, numpy)"],
               ["Training", f"backpropagation through time, mini-batch gradient descent, batch {BATCH}, {EPOCHS} epochs"],
               ["Selected learning rate", best_lr],
               ["Selection rule", "highest mean validation accuracy over 3 runs (test set not used)"],
               ["Train / validation / test loads", f"{len(tr)} / {len(va)} / {len(te)}"],
               ["Input shape", "4 time steps x 6 features"], ["Classes", "CP, CC, CZ"],
               ["Target", ">= 95 % accuracy, < 5 % error"]] + [[k, round(v, 3)] for k, v in s.items()]
    r = put_table(ws, 1, ["Item", "Value"], summary, "Summary (selected model, unseen test set)")
    keys = list(lr_tbl[0])
    r = put_table(ws, r, keys, dict_rows(lr_tbl, keys), "Learning-rate experiment (3 runs per learning rate)")
    r = put_table(ws, r, ["Learning rate", "Seed", "Best epoch", "Validation accuracy (%)", "Test accuracy (%)"],
                  [[x["lr"], x["seed"], x["best_epoch"], 100 * x["val_acc"], 100 * x["test_acc"]] for x in runs],
                  "All training runs")
    r = put_table(ws, r, ["Class", "Precision (%)", "Recall (%)", "F1-score (%)", "Support (loads)"],
                  [[c, 100 * prec[k], 100 * rec[k], 100 * f1[k], int(cm[k].sum())] for k, c in enumerate(CLASSES)],
                  "Per-class metrics (selected model)")
    r = put_table(ws, r, list(grp[0]), dict_rows(grp, list(grp[0])), "Accuracy by circuit and condition")
    put_table(ws, r, ["Epoch", "Train loss", "Validation loss", "Train accuracy (%)", "Validation accuracy (%)"],
              [[h["epoch"], h["train_loss"], h["val_loss"], 100 * h["train_acc"], 100 * h["val_acc"]]
               for h in best["hist"]], f"Training history (selected model, learning rate {best_lr})")
    for j in range(1, 8): ws.column_dimensions[get_column_letter(j)].width = 22
    ws.column_dimensions["B"].width = 40

    ws = wb.create_sheet("Confusion Matrix")
    pct = cm / cm.sum(1, keepdims=True) * 100
    r = put_table(ws, 1, ["Actual"] + [f"Predicted {c}" for c in CLASSES],
                  [[c, *cm[k]] for k, c in enumerate(CLASSES)], "Confusion matrix - counts (rows = actual)")
    put_table(ws, r, ["Actual"] + [f"Predicted {c} (%)" for c in CLASSES],
              [[c, *np.round(pct[k], 2)] for k, c in enumerate(CLASSES)], "Confusion matrix - row % (recall)")
    for j in range(1, 5): ws.column_dimensions[get_column_letter(j)].width = 18
    ws.add_image(XLImage(str(cm_png)), "F2")

    ws = wb.create_sheet("Graphical Analysis")
    ws["A1"] = "Graphical Analysis - all figures (also saved as PNG in outputs2/figures)"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    r = 3
    for caption, p in figs:
        ws.cell(r, 1, caption).font = Font(bold=True, size=11)
        img = XLImage(str(p))
        k = min(1.0, 1000 / img.width)
        img.width, img.height = img.width * k, img.height * k
        ws.add_image(img, f"A{r + 1}")
        r += int(img.height / 20) + 4
    xlsx = OUT / "RNN_Load_Classification_Results.xlsx"
    wb.save(xlsx)
    print(f"\nWrote {xlsx}\nFigures in {FIG}")


if __name__ == "__main__":
    main()
