"""Integration test dengan model XGBoost sungguhan.

`test_detector.py` memakai stub, jadi tidak membuktikan kompatibilitas API
XGBoost. Test ini melatih model kecil sungguhan, menulis artefak dalam format
yang sama seperti Notebook 3-4, lalu memanggil `load_detector()` dan
`load_sample_pool()` untuk memastikan seluruh pipeline jalan.

Semua artefak dibuat di folder sementara — test tidak menyentuh data asli.

Jalankan: python tests/test_integration.py
"""

from __future__ import annotations

import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import detector as D  # noqa: E402

RESULTS: list[bool] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    RESULTS.append(bool(condition))
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}{(' - ' + detail) if detail else ''}")


def make_dataset(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.normal(size=n) for c in D.V_COLS})
    frame["Amount"] = rng.uniform(0.0, 3000.0, n)
    frame["Time"] = rng.uniform(0.0, 172_800.0, n)
    label = (frame["V14"] + frame["V10"] > 2.2).astype(int)
    return frame, label


def write_artifacts(models_dir: Path, data_dir: Path, scaler, threshold=0.5,
                    f1=0.8) -> D.ArtifactPaths:
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "threshold.pkl", "wb") as fh:
        pickle.dump({"optimal_threshold": threshold, "optimal_f1_val": f1}, fh)
    with open(data_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    return D.ArtifactPaths(models_dir=models_dir, data_dir=data_dir)


def write_samples(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    D.add_engineered_features(frame).to_csv(path, index=False)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fraud_it_"))
    models_dir = tmp / "data" / "models"
    data_dir = tmp / "data"

    print("Menyiapkan data sintetis")
    raw, y = make_dataset()
    print(f"  {len(raw)} baris, {int(y.sum())} fraud, {len(D.FEATURES)} fitur")

    fe = D.add_engineered_features(raw)
    scaler = StandardScaler().fit(fe[D.FEATURES])
    Xs = pd.DataFrame(scaler.transform(fe[D.FEATURES]), columns=D.FEATURES)

    print("\nMelatih model XGBoost")
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "seed": 42,
    }
    dtrain = xgb.DMatrix(Xs, label=y)
    booster = xgb.train(params, dtrain, num_boost_round=30, evals=[(dtrain, "train")],
                        early_stopping_rounds=10, verbose_eval=False)
    models_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(models_dir / "xgb_model.json"))
    n_trees = int(booster.best_iteration) + 1
    check("best_iteration tersedia", booster.best_iteration is not None,
          f"{n_trees} trees")

    paths = write_artifacts(models_dir, data_dir, scaler)

    samples = raw.copy()
    samples[D.LABEL] = y
    write_samples(data_dir / "samples.csv", samples)

    print("\nMemuat detector")
    det = D.load_detector(paths)
    check("load_detector berhasil", isinstance(det, D.FraudDetector))
    check("n_trees terbaca dari model", det.n_trees == n_trees, str(det.n_trees))
    check("scaler konsisten 31 fitur", len(D.FEATURES) == 31)
    check("ambang terbaca", det.optimal_threshold == 0.5)
    check("F1 tersimpan", np.isclose(det.optimal_f1, 0.8))
    check("hanya 3 artefak dibutuhkan",
          not any((data_dir / f).exists()
                  for f in ["X_train_scaled.pkl", "y_val.pkl", "y_test.pkl"]),
          "tidak ada X_train/X_val/y_test")

    print("\nMemuat pool sampel")
    pool = D.load_sample_pool(paths)
    check("jumlah baris pool", pool.n_total == len(raw), f"{pool.n_total}")
    check("jumlah fraud", pool.count(D.CLASS_FRAUD) == int(y.sum()))
    check("jumlah sah", pool.count(D.CLASS_NORMAL) == int((y == 0).sum()))

    print("\nPrediksi: kasus fraud harus terdeteksi")
    fraud_rows = samples[samples[D.LABEL] == D.CLASS_FRAUD].head(1)
    res = det.predict_row(fraud_rows.iloc[0])
    check("level FRAUD", res["level"] == D.LEVEL_FRAUD, res["level"])
    check("proba di atas ambang", res["proba"] > det.optimal_threshold,
          f"{res['proba']:.4f}")

    print("\nPrediksi: kasus sah harus lolos")
    ok_rows = samples[samples[D.LABEL] == D.CLASS_NORMAL].head(1)
    res_ok = det.predict_row(ok_rows.iloc[0])
    check("level NORMAL", res_ok["level"] == D.LEVEL_NORMAL, res_ok["level"])
    check("proba di bawah ambang", res_ok["proba"] < det.optimal_threshold,
          f"{res_ok['proba']:.6f}")

    print("\nKontribusi fitur (API XGBoost aktual)")
    contribs = res["contribs"]
    check("31 fitur + bias", len(contribs) == 32, str(len(contribs)))
    check("X_raw shape (1, 31)", res["X_raw"].shape == (1, 31))

    row_fe = D.add_engineered_features(
        fraud_rows.iloc[0].to_frame().T)[D.FEATURES].astype(float)
    d_row = xgb.DMatrix(pd.DataFrame(
        np.asarray(det.scaler.transform(row_fe), dtype=float), columns=D.FEATURES))
    margin = float(np.asarray(
        booster.predict(d_row, output_margin=True,
                        iteration_range=(0, n_trees))).ravel()[0])
    check("logit kontribusi == margin prediksi",
          np.isclose(float(contribs.sum()), margin, atol=1e-4),
          f"{contribs.sum():.6f} vs {margin:.6f}")

    print("\niteration_range membatasi jumlah pohon")
    pos_fraud = int(samples.index[samples[D.LABEL] == D.CLASS_FRAUD][0])
    d = xgb.DMatrix(Xs.iloc[[pos_fraud]])
    semua = booster.predict(d)
    dipangkas = booster.predict(d, iteration_range=(0, n_trees))
    check("prediksi terpotong benar-benar lebih pendek",
          not np.allclose(semua, dipangkas) or n_trees == booster.num_boosted_rounds(),
          f"{n_trees} dari {booster.num_boosted_rounds()}")
    check("predict_row memakai jumlah pohon yang sama",
          np.isclose(res["proba"], float(np.asarray(dipangkas).ravel()[0]), atol=1e-12),
          f"{res['proba']:.12f} vs {float(np.asarray(dipangkas).ravel()[0]):.12f}")

    print("\nError handling scaler tidak cocok")
    bad = StandardScaler().fit(fe[D.FEATURES].iloc[:, :20])
    with open(data_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(bad, fh)
    try:
        D.load_detector(paths)
        check("scaler salah dimensi harus error", False)
    except D.ArtifactError:
        check("scaler salah dimensi harus error", True)
    with open(data_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    print("\nError handling threshold.pkl tidak punya kunci")
    with open(models_dir / "threshold.pkl", "wb") as fh:
        pickle.dump({"ambang": 0.5}, fh)
    try:
        D.load_detector(paths)
        check("kunci hilang harus error", False)
    except D.ArtifactError:
        check("kunci hilang harus error", True)
    write_artifacts(models_dir, data_dir, scaler)

    print("\nError handling artefak model hilang")
    kosong = tmp / "kosong"
    (kosong / "data" / "models").mkdir(parents=True)
    (kosong / "data").mkdir(parents=True, exist_ok=True)
    try:
        D.load_detector(D.ArtifactPaths(kosong / "data" / "models", kosong / "data"))
        check("artefak hilang harus error", False)
    except D.ArtifactError:
        check("artefak hilang harus error", True)

    print("\nError handling samples.csv tidak ada")
    tanpa = tmp / "tanpa_sampel"
    p2 = write_artifacts(tanpa / "data" / "models", tanpa / "data", scaler)
    shutil.copy(models_dir / "xgb_model.json", tanpa / "data" / "models")
    try:
        D.load_sample_pool(p2)
        check("pool hilang harus error", False)
    except D.SamplePoolError:
        check("pool hilang harus error", True)

    print("\nError handling samples.csv tidak punya Class")
    cacat = tmp / "cacat"
    p3 = write_artifacts(cacat / "data" / "models", cacat / "data", scaler)
    shutil.copy(models_dir / "xgb_model.json", cacat / "data" / "models")
    samples.drop(columns=[D.LABEL]).to_csv(cacat / "data" / "samples.csv",
                                           index=False)
    try:
        D.load_sample_pool(p3)
        check("pool tanpa Class harus error", False)
    except D.SamplePoolError:
        check("pool tanpa Class harus error", True)

    print("\nRegression: model tanpa early stopping")
    plain = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "seed": 1},
                      dtrain, num_boost_round=7, verbose_eval=False)
    try:
        plain.best_iteration
        early = "ada"
    except AttributeError:
        early = "tidak ada"
    check("best_iteration melempar tanpa early stopping", early == "tidak ada", early)

    plain_dir = tmp / "plain"
    (plain_dir / "data" / "models").mkdir(parents=True)
    plain.save_model(str(plain_dir / "data" / "models" / "xgb_model.json"))
    p4 = write_artifacts(plain_dir / "data" / "models", plain_dir / "data", scaler)
    try:
        plain_det = D.load_detector(p4)
        check("load_detector menangani model tanpa early stopping", True,
              f"n_trees={plain_det.n_trees}")
        check("n_trees = jumlah pohon", plain_det.n_trees == 7,
              str(plain_det.n_trees))
        r = plain_det.predict_row(samples.iloc[0])
        check("prediksi tetap jalan", 0.0 <= r["proba"] <= 1.0, f"{r['proba']:.4f}")
    except AttributeError as exc:
        check("load_detector menangani model tanpa early stopping", False, str(exc))

    print("\nBersihkan")
    shutil.rmtree(tmp, ignore_errors=True)
    check("folder temp dihapus", not tmp.exists())

    passed = sum(RESULTS)
    total = len(RESULTS)
    print("\n" + "=" * 46)
    print(f"  {passed}/{total} pemeriksaan lolos")
    print("=" * 46)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
