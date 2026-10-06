"""Test end-to-end aplikasi Streamlit.

Melatih model XGBoost kecil dari data sintetis, menulis artefak dan
`samples.csv` ke folder sementara, lalu menjalankan `app.py` lewat
`streamlit.testing.v1.AppTest` dan memeriksa output yang benar-benar dirender.

Test ini tidak membaca dataset asli, jadi bisa dijalankan di mana saja.

Jalankan: python tests/test_app.py
"""

from __future__ import annotations

import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import detector as D  # noqa: E402

RESULTS: list[bool] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    RESULTS.append(bool(condition))
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")


def build_dataset(n=2500, n_fraud_target=120, seed=3):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.normal(size=n) for c in D.V_COLS})
    frame["Amount"] = rng.uniform(0.0, 3000.0, n)
    frame["Time"] = rng.uniform(0.0, 172_800.0, n)

    score = frame["V14"] * 1.4 + frame["V10"] * 0.9 - 0.4
    fraud_idx = np.argsort(score.to_numpy())[-n_fraud_target:]
    label = np.zeros(n, dtype=int)
    label[fraud_idx] = 1
    frame[D.LABEL] = label
    return frame


def build_artifacts(target: Path, samples: pd.DataFrame):
    """Latih model kecil dan tulis artefak + samples.csv ke `target`."""
    import xgboost as xgb
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    models_dir = target / "data" / "models"
    data_dir = target / "data"
    models_dir.mkdir(parents=True, exist_ok=True)

    fe = D.add_engineered_features(samples)
    X = fe[D.FEATURES]
    y = samples[D.LABEL].to_numpy()

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    scaler = StandardScaler().fit(X_train)
    Xtr = pd.DataFrame(scaler.transform(X_train), columns=D.FEATURES)
    Xva = pd.DataFrame(scaler.transform(X_val), columns=D.FEATURES)

    dtrain = xgb.DMatrix(Xtr, label=y_train)
    dval = xgb.DMatrix(Xva, label=y_val)
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "max_depth": 5, "eta": 0.15, "subsample": 0.9,
        "colsample_bytree": 0.9, "min_child_weight": 2,
        "tree_method": "hist", "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=60,
                        evals=[(dval, "val")], early_stopping_rounds=15,
                        verbose_eval=False)
    booster.save_model(str(models_dir / "xgb_model.json"))
    n_trees = int(booster.best_iteration) + 1

    val_proba = booster.predict(dval, iteration_range=(0, n_trees))
    precision, recall, thr = precision_recall_curve(y_val, val_proba)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    optimal_threshold = float(thr[int(np.argmax(f1))])

    with open(models_dir / "threshold.pkl", "wb") as fh:
        pickle.dump({"optimal_threshold": optimal_threshold,
                     "optimal_f1_val": float(np.max(f1))}, fh)
    with open(data_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    fe.to_csv(data_dir / "samples.csv", index=False)

    print(f"  trees={n_trees}  ambang={optimal_threshold:.4f}  F1={np.max(f1):.4f}")
    print(f"  samples.csv: {len(samples)} baris "
          f"(fraud {int(samples[D.LABEL].sum())})")
    return models_dir, data_dir


def history_frame(at):
    for d in at.dataframe:
        cols = set(d.value.columns)
        if {"Label asli", "Prediksi"} <= cols:
            return d.value
    return None


def metrics_frame(at):
    """Tabel Performance Metrics di sidebar."""
    for d in at.sidebar.dataframe:
        if "Metric" in d.value.columns and "Value" in d.value.columns:
            return d.value
    return None


def tombol(at, kata: str):
    """Cari tombol lewat label. `at.button` memuat tombol sidebar juga."""
    for b in at.button:
        if kata in b.label:
            return b
    return None


def tombol_acak(at):
    return tombol(at, "Ambil acak")


def tombol_deteksi(at):
    return tombol(at, "Deteksi") or tombol(at, "deteksi")


def subheader_ada(at, teks: str) -> bool:
    return any(teks in s.value for s in at.subheader)


def _hapus_aktif_berikutnya(at) -> bool:
    """Status `disabled` tombol sidebar dihitung dari state di awal run, jadi
    selalu satu run tertinggal. Ceknya dilakukan setelah satu interaksi lagi."""
    at.run()
    return at.sidebar.button[0].disabled is False


def main() -> int:
    from streamlit.testing.v1 import AppTest

    tmp = Path(tempfile.mkdtemp(prefix="fraud_app_"))
    print("Menyiapkan artefak sintetis")
    samples = build_dataset()
    models_dir, data_dir = build_artifacts(tmp, samples)

    os.environ["FRAUD_MODELS_DIR"] = str(models_dir)
    os.environ["FRAUD_DATA_DIR"] = str(data_dir)

    det = D.load_detector(D.ArtifactPaths(models_dir, data_dir))
    pool = D.load_sample_pool(D.ArtifactPaths(models_dir, data_dir))
    print(f"  ambang={det.optimal_threshold:.4f}  "
          f"potential={det.potential_threshold:.4f}  "
          f"pool={pool.n_total} baris")

    print("\nLangkah 1-2: render awal dan ambil acak")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    check("tanpa exception", not at.exception, str(at.exception)[:300])
    check("judul tampil", any("Deteksi Fraud" in t.value for t in at.title))
    check("1 radio (pilih jenis transaksi)", len(at.radio) == 1,
          f"{len(at.radio)} radio")
    check("3 opsi: Fraud, Normal, Random",
          len(at.radio[0].options) == 3, str(at.radio[0].options))
    check(" Radio berisi opsi random",
          any("Random" in str(o) for o in at.radio[0].options))
    check("tombol Ambil acak ada", tombol_acak(at) is not None)
    check("tombol Deteksi belum ada sebelum ambil sampel",
          tombol_deteksi(at) is None)
    check("langkah 1 tampil", subheader_ada(at, "Pilih jenis transaksi"))
    check("belum ada rincian", not subheader_ada(at, "Rincian transaksi"))
    check("belum ada hasil", not subheader_ada(at, "Hasil deteksi"))
    check("tanpa number_input (tidak ada form manual)",
          len(at.number_input) == 0, f"{len(at.number_input)} input")
    check("tombol hapus riwayat nonaktif saat riwayat kosong",
          len(at.sidebar.button) == 1 and at.sidebar.button[0].disabled is True,
          str(at.sidebar.button))

    mtab = metrics_frame(at)
    check("tabel performance metrics di sidebar", mtab is not None)
    check("7 metrik", mtab is not None and len(mtab) == 7,
          str(len(mtab)) if mtab is not None else "tidak ada")
    if mtab is not None:
        nilai = dict(zip(mtab["Metric"], mtab["Value"]))
        check("Precision tampil", nilai.get("Precision") == "0.9310",
              str(nilai.get("Precision")))
        check("ROC-AUC tampil", nilai.get("ROC-AUC") == "0.9715",
              str(nilai.get("ROC-AUC")))
        check("Fraud Detection Rate tampil",
              nilai.get("Fraud Detection Rate (TPR)") == "0.7606",
              str(nilai.get("Fraud Detection Rate (TPR)")))
        check("False Alarm Rate tampil",
              nilai.get("False Alarm Rate (FPR)") == "0.0001",
              str(nilai.get("False Alarm Rate (FPR)")))
    sidebar_teks = " ".join(m.value for m in at.sidebar.markdown) + \
                   " ".join(m.value for m in at.sidebar.caption)
    check("deskripsi model dihapus dari sidebar",
          "Seberapa baik" not in sidebar_teks and "Arsitektur" not in sidebar_teks)
    check("tidak ada 2 metrik performa lagi", len(at.sidebar.metric) == 0,
          f"{len(at.sidebar.metric)} metrik di sidebar")

    print("\nLangkah 3: ambil acak")
    tombol_acak(at).click().run()
    check("tanpa exception setelah ambil", not at.exception,
          str(at.exception)[:300])
    check("langkah rincian tampil", subheader_ada(at, "Rincian transaksi"))
    check("tombol Deteksi muncul setelah ambil", tombol_deteksi(at) is not None)
    check("belum ada hasil sebelum klik Deteksi",
          not subheader_ada(at, "Hasil deteksi"))
    check("belum ada riwayat", history_frame(at) is None)
    teks = " ".join(m.value for m in at.markdown) + \
           " ".join(m.value for m in at.caption)
    check("tidak ada 'hari ke' di halaman", "hari ke" not in teks.lower())

    print("\nLangkah 4: deteksi")
    tombol_deteksi(at).click().run()
    check("tanpa exception setelah deteksi", not at.exception,
          str(at.exception)[:300])
    check("langkah hasil tampil", subheader_ada(at, "Hasil deteksi"))
    check("grafik kontribusi dirender", len(at.get("plotly_chart")) >= 1,
          f"{len(at.get('plotly_chart'))} grafik")

    hist = history_frame(at)
    check("tabel riwayat ada", hist is not None,
          str([list(d.value.columns) for d in at.dataframe]))
    check("1 baris riwayat setelah 1 deteksi", hist is not None and len(hist) == 1,
          f"{len(hist) if hist is not None else 0} baris")
    check("riwayat mencatat label Fraud",
          hist is not None and hist.iloc[0]["Label asli"] == D.format_class_label(D.CLASS_FRAUD),
          str(hist.iloc[0]["Label asli"]) if hist is not None else "")

    print("\nPilihan Normal: ambil lalu deteksi")
    at.radio[0].set_value("Normal")
    at.run()
    check("tanpa exception setelah ganti kelas", not at.exception,
          str(at.exception)[:300])
    tombol_acak(at).click().run()
    check("rincian transaksi sudah ada, hasil belum",
          subheader_ada(at, "Rincian transaksi")
          and not subheader_ada(at, "Hasil deteksi"))
    tombol_deteksi(at).click().run()
    check("tanpa exception setelah deteksi normal", not at.exception,
          str(at.exception)[:300])
    hist = history_frame(at)
    check("2 baris riwayat", hist is not None and len(hist) == 2,
          f"{len(hist) if hist is not None else 0} baris")
    check("riwayat terbaru bukan fraud",
          hist is not None and hist.iloc[0]["Label asli"] == D.format_class_label(D.CLASS_NORMAL),
          str(hist.iloc[0]["Label asli"]) if hist is not None else "")
    check("kolom Jam terisi", hist is not None and hist["Jam"].notna().all())

    print("\nPilihan Random: kelas boleh apa saja")
    at.radio[0].set_value("Random")
    at.run()
    check("tanpa exception pada pilihan Random", not at.exception,
          str(at.exception)[:300])
    for _ in range(4):
        tombol_acak(at).click().run()
        tombol_deteksi(at).click().run()
    check("tanpa exception setelah random", not at.exception,
          str(at.exception)[:300])
    hist = history_frame(at)
    check("6 baris riwayat setelah random", hist is not None and len(hist) == 6,
          f"{len(hist) if hist is not None else 0} baris")
    check("label asli random tetap 0/1",
          hist is not None
          and all(str(v) in (D.format_class_label(0), D.format_class_label(1))
                  for v in hist["Label asli"]),
          str(sorted(set(hist["Label asli"]))) if hist is not None else "")

    print("\nKonsistensi prediksi terhadap detector")
    if hist is not None:
        levels_valid = hist["Prediksi"].isin(D.LEVEL_ORDER).all()
        check("semua prediksi adalah level yang valid", bool(levels_valid),
              str(sorted(set(hist["Prediksi"]))))
        check("probabilitas dalam rentang 0..1",
              bool(hist["Probabilitas"].between(0.0, 1.0).all()))
        consistent = all(
            (r["Prediksi"] == D.LEVEL_FRAUD)
            == (r["Label asli"] == D.format_class_label(D.CLASS_FRAUD))
            for _, r in hist.iterrows()
        )
        check("kolom Benar konsisten dengan prediksi & label",
              bool(consistent), str(hist[["Label asli", "Prediksi",
                                          "Benar"]].to_dict("records")))
        check("level sesuai ambang detector",
              bool(((hist["Probabilitas"] >= det.optimal_threshold)
                    == (hist["Prediksi"] == D.LEVEL_FRAUD)).all()),
              f"ambang {det.optimal_threshold:.4f}")
        check("zona potential sesuai 0,9 x ambang",
              bool(((hist["Prediksi"] == D.LEVEL_POTENTIAL)
                    == ((hist["Probabilitas"] < det.optimal_threshold)
                        & (hist["Probabilitas"] >= det.potential_threshold))).all()),
              f"potential {det.potential_threshold:.4f}")

    print("\nLima pasangan ambil-deteksi")
    for _ in range(5):
        tombol_acak(at).click().run()
        tombol_deteksi(at).click().run()
    check("tanpa exception setelah 5 siklus", not at.exception,
          str(at.exception)[:300])
    hist = history_frame(at)
    check("11 baris riwayat", hist is not None and len(hist) == 11,
          f"{len(hist) if hist is not None else 0} baris")
    check("semua baris punya nominal & jam",
          hist is not None and hist["Nominal"].notna().all()
          and hist["Jam"].notna().all())
    check("3 level tampil sebagai metrik (termasuk yang kosong)",
          len(at.metric) >= 3, f"{len(at.metric)} metrik")

    print("\nHapus riwayat")
    check("tombol hapus riwayat di sidebar", len(at.sidebar.button) == 1,
          f"{len(at.sidebar.button)} tombol di sidebar")
    at.sidebar.button[0].click().run()
    check("tanpa exception saat hapus", not at.exception, str(at.exception)[:300])
    check("state riwayat jadi kosong",
          list(at.session_state.filtered_state.get("riwayat", [])) == [],
          str(at.session_state.filtered_state.get("riwayat")))
    check("riwayat tidak lagi dirender", history_frame(at) is None
          or len(history_frame(at)) == 0)
    check("hasil hilang", not subheader_ada(at, "Hasil deteksi"))
    check("rincian hilang", not subheader_ada(at, "Rincian transaksi"))

    print("\nBisa lanjut setelah hapus")
    tombol_acak(at).click().run()
    check("tanpa exception setelah hapus lalu ambil lagi", not at.exception,
          str(at.exception)[:300])
    tombol_deteksi(at).click().run()
    hist = history_frame(at)
    check("riwayat mulai dari 1 lagi", hist is not None and len(hist) == 1,
          f"{len(hist) if hist is not None else 0} baris")
    check("tombol hapus aktif lagi setelah ada riwayat",
          _hapus_aktif_berikutnya(at))

    print("\nBersihkan")
    shutil.rmtree(tmp, ignore_errors=True)
    check("folder temp dihapus", not tmp.exists())

    passed, total = sum(RESULTS), len(RESULTS)
    print("\n" + "=" * 46)
    print(f"  {passed}/{total} pemeriksaan lolos")
    print("=" * 46)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
