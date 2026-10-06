"""Smoke test logika deteksi tanpa Streamlit dan tanpa model sungguhan.

Memverifikasi konsistensi internal `FraudDetector` dan `SamplePool` memakai
stub, sehingga bisa dijalankan tanpa artefak apa pun.

Jalankan: python tests/test_detector.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import detector as D  # noqa: E402


class FakeBooster:
    """Booster yang mengembalikan probabilitas dan margin tetap."""

    def __init__(self, margin=2.0):
        self._margin = margin
        self.calls: list[dict] = []

    def predict(self, d, output_margin=False, pred_contribs=False, **kw):
        n = d.num_row()
        self.calls.append(dict(kw))
        if pred_contribs:
            out = np.zeros((n, len(D.FEATURES) + 1))
            out[:, 0] = self._margin * 0.4
            out[:, 5] = self._margin * 0.3
            out[:, -1] = self._margin * 0.3
            return out
        if output_margin:
            return np.full(n, self._margin)
        return np.full(n, 1.0 / (1.0 + np.exp(-self._margin)))

    def get_score(self, importance_type=None):
        return {"V1": 3.0, "V14": 5.0, "Amount_log": 1.0}


class FakeScaler:
    n_features_in_ = 31
    feature_names_in_ = np.array(D.FEATURES, dtype=object)

    def transform(self, X):
        return np.asarray(X, dtype=float)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float)


def make_detector(margin=2.0, optimal=0.83, n_trees=100):
    return D.FraudDetector(
        booster=FakeBooster(margin),
        scaler=FakeScaler(),
        optimal_threshold=optimal,
        n_trees=n_trees,
        optimal_f1=0.85,
    )


def make_frame(n_normal=40, n_fraud=6, seed=1):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.normal(size=n_normal + n_fraud)
                          for c in D.RAW_COLS})
    frame[D.LABEL] = [D.CLASS_NORMAL] * n_normal + [D.CLASS_FRAUD] * n_fraud
    return frame


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}{(' - ' + detail) if detail else ''}")
    return bool(condition)


def main():
    results = []
    det = make_detector()

    print("Klasifikasi 3 level")
    pot = det.potential_threshold
    results.append(check("ambang potential = 0,9 x ambang fraud",
                         np.isclose(pot, 0.83 * 0.9), f"{pot:.4f}"))
    results.append(check("proba >= ambang fraud -> FRAUD",
                         det.classify(0.95) == D.LEVEL_FRAUD))
    results.append(check("tepat di ambang fraud -> FRAUD",
                         det.classify(0.83) == D.LEVEL_FRAUD,
                         f"ambang {det.optimal_threshold}"))
    results.append(check("tepat di ambang potential -> POTENTIAL FRAUD",
                         det.classify(pot) == D.LEVEL_POTENTIAL, f"{pot:.4f}"))
    results.append(check("sedikit di bawah ambang potential -> NORMAL",
                         det.classify(np.nextafter(pot, 0.0)) == D.LEVEL_NORMAL))
    results.append(check("di antara kedua ambang -> POTENTIAL FRAUD",
                         det.classify(0.80) == D.LEVEL_POTENTIAL))
    results.append(check("proba rendah -> NORMAL",
                         det.classify(0.10) == D.LEVEL_NORMAL))
    results.append(check("ambang kustom dipakai",
                         det.classify(0.80, threshold=0.9) == D.LEVEL_NORMAL))
    results.append(check("3 level", len(D.LEVEL_ORDER) == 3, str(D.LEVEL_ORDER)))
    results.append(check("POTENTIAL di urutan tengah",
                         D.LEVEL_ORDER[1] == D.LEVEL_POTENTIAL))

    print("\nFeature engineering")
    fe = D.add_engineered_features(
        pd.DataFrame([{"Time": 3 * 3600, "Amount": 250.0,
                       **{c: 0.5 for c in D.V_COLS}}]))
    results.append(check("Amount_log = log1p(250)",
                         np.isclose(fe["Amount_log"].iloc[0], np.log1p(250.0))))
    results.append(check("Time_hour = 3", fe["Time_hour"].iloc[0] == 3))
    results.append(check("Time_day = 0", fe["Time_day"].iloc[0] == 0))
    results.append(check("31 fitur, urut benar",
                         list(fe[D.FEATURES].columns) == D.FEATURES))

    print("\nPipeline predict_row")
    row = pd.Series({"Time": 3 * 3600, "Amount": 250.0,
                     **{c: 0.5 for c in D.V_COLS}})
    res = det.predict_row(row)
    expected = 1.0 / (1.0 + np.exp(-2.0))
    results.append(check("proba = sigmoid(margin stub)", np.isclose(res["proba"], expected),
                         f"{res['proba']:.4f}"))
    results.append(check("level terisi", res["level"] in D.LEVEL_ORDER, res["level"]))
    results.append(check("X_raw shape (1, 31)", res["X_raw"].shape == (1, 31)))
    results.append(check("contribs = 31 fitur + bias", len(res["contribs"]) == 32,
                         str(len(res["contribs"]))))
    results.append(check("kontribusi summing ke logit stub",
                         np.isclose(res["contribs"].sum(), 2.0, atol=1e-9)))

    print("\niteration_range diteruskan")
    det.booster.calls.clear()
    det.predict_row(row)
    results.append(check("iteration_range = (0, n_trees)",
                         det.booster.calls[0].get("iteration_range") == (0, 100),
                         str(det.booster.calls[0].get("iteration_range"))))

    det_tanpa = D.FraudDetector(FakeBooster(), FakeScaler(), 0.83, n_trees=None)
    det_tanpa.booster.calls.clear()
    det_tanpa.predict_row(row)
    results.append(check("n_trees=None tidak kirim iteration_range",
                         "iteration_range" not in det_tanpa.booster.calls[0]))

    print("\nKolom mentah kurang harus error")
    try:
        row_bad = row.copy()
        row_bad["Amount"] = np.nan
        det.predict_row(row_bad)
        results.append(check("Amount NaN harus tetap punya Amount_log", True))
    except Exception as exc:  # noqa: BLE001
        results.append(check("Amount NaN harus tetap punya Amount_log", False, str(exc)))

    print("\nSamplePool: konstruksi dan validasi")
    pool = D.SamplePool(make_frame())
    results.append(check("total baris", pool.n_total == 46, str(pool.n_total)))
    results.append(check("jumlah fraud", pool.count(D.CLASS_FRAUD) == 6))
    results.append(check("jumlah sah", pool.count(D.CLASS_NORMAL) == 40))

    try:
        D.SamplePool(make_frame().drop(columns=[D.LABEL]))
        results.append(check("tanpa Class harus error", False))
    except D.SamplePoolError:
        results.append(check("tanpa Class harus error", True))

    try:
        bad = make_frame()
        bad[D.LABEL] = 7
        D.SamplePool(bad)
        results.append(check("Class bukan 0/1 harus error", False))
    except D.SamplePoolError:
        results.append(check("Class bukan 0/1 harus error", True))

    only_fraud = make_frame()
    only_fraud = only_fraud[only_fraud[D.LABEL] == D.CLASS_FRAUD]
    try:
        D.SamplePool(only_fraud)
        results.append(check("salah satu kelas kosong harus error", False))
    except D.SamplePoolError:
        results.append(check("salah satu kelas kosong harus error", True))

    print("\nSamplePool: pengambilan acak")
    rng = random.Random(42)
    draws = [pool.draw(D.CLASS_FRAUD, rng) for _ in range(30)]
    results.append(check("semua draw dari kelas diminta",
                         all(int(d[D.LABEL]) == D.CLASS_FRAUD for d in draws)))
    results.append(check("tidak selalu baris sama",
                         len({d.name for d in draws}) > 1,
                         f"{len({d.name for d in draws})} baris berbeda"))
    results.append(check("draw menghasilkan Series", isinstance(draws[0], pd.Series)))

    try:
        pool.draw(99, rng)
        results.append(check("label tak dikenal harus error", False))
    except D.SamplePoolError:
        results.append(check("label tak dikenal harus error", True))

    print("\nSamplePool: draw_any (seluruh pool)")
    campur = [pool.draw_any(random.Random(i)) for i in range(60)]
    labels = {int(d[D.LABEL]) for d in campur}
    results.append(check("bisa menghasilkan kelas apa pun",
                         labels <= {D.CLASS_NORMAL, D.CLASS_FRAUD}, str(labels)))
    results.append(check("tidak selalu kelas yang sama",
                         len({d.name for d in campur}) > 1,
                         f"{len({d.name for d in campur})} baris berbeda"))
    results.append(check("nilai V valid",
                         all(D.V_COLS[0] in d.index for d in campur)))

    a = pool.draw_any(random.Random(0))
    b = pool.draw_any(random.Random(0))
    results.append(check("seed sama -> baris sama", a.equals(b)))
    c = pool.draw(D.CLASS_FRAUD, random.Random(0))
    d = pool.draw(D.CLASS_FRAUD, random.Random(0))
    results.append(check("draw kelas juga deterministik per seed", c.equals(d)))

    print("\ndescribe_time")
    t0 = D.describe_time(0)
    t1 = D.describe_time(3 * 3600)
    t2 = D.describe_time(172_800 + 2 * 3600)
    results.append(check("pukul 00:00", (t0["hour"], t0["text"]) == (0, "00:00"),
                         t0["text"]))
    results.append(check("pukul 03:00", (t1["hour"], t1["text"]) == (3, "03:00"),
                         t1["text"]))
    results.append(check("rollover hari tidak mengubah jam",
                         (t2["hour"], t2["text"]) == (2, "02:00"), t2["text"]))
    results.append(check("tidak ada lagi kolom 'day'", "day" not in t1))
    results.append(check("tandai jam sepi",
                         t1["jam_sepi"] is True and t0["jam_sepi"] is True))
    results.append(check("pukul 14 bukan jam sepi",
                         D.describe_time(14 * 3600)["jam_sepi"] is False))
    results.append(check("pukul 06 bukan jam sepi",
                         D.describe_time(6 * 3600)["jam_sepi"] is False))

    print("\nTEST_METRICS")
    results.append(check("7 metrik", len(D.TEST_METRICS) == 7,
                         str(len(D.TEST_METRICS))))
    results.append(check("nilai sesuai Notebook 5",
                         D.TEST_METRICS["Precision"] == 0.9310
                         and D.TEST_METRICS["ROC-AUC"] == 0.9715
                         and D.TEST_METRICS["False Alarm Rate (FPR)"] == 0.0001))
    results.append(check("semua di rentang 0..1",
                         all(0.0 <= v <= 1.0 for v in D.TEST_METRICS.values())))

    print("\nKonstanta")
    results.append(check("28 fitur V", len(D.V_COLS) == 28))
    results.append(check("31 fitur model", len(D.FEATURES) == 31))
    results.append(check("feur tanggungan-features = 3",
                         set(D.FEATURES) - set(D.V_COLS) ==
                         {"Amount_log", "Time_hour", "Time_day"}))
    results.append(check("RAW_COLS memuat Time+V+Amount",
                         D.RAW_COLS == ["Time"] + D.V_COLS + ["Amount"]))

    print("\ndefault_paths tidak melempar untuk root asing")
    try:
        p = D.default_paths(Path("/tmp/tidak-ada-sama-sekali"))
        results.append(check("tidak melempar", isinstance(p, D.ArtifactPaths)))
        results.append(check("scaler & samples di data_dir yang sama",
                             p.scaler.parent == p.samples.parent))
    except Exception as exc:  # noqa: BLE001
        results.append(check("tidak melempar", False, str(exc)))

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 46)
    print(f"  {passed}/{total} pemeriksaan lolos")
    print("=" * 46)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
