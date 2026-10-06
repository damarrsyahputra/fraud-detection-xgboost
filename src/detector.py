import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time"] + V_COLS + ["Amount"]
FEATURES = V_COLS + ["Amount_log", "Time_hour", "Time_day"]
LABEL = "Class"

CLASS_NORMAL = 0
CLASS_FRAUD = 1
CLASS_LABELS = {CLASS_NORMAL: "Normal", CLASS_FRAUD: "Fraud"}

LEVEL_NORMAL = "NORMAL"
LEVEL_POTENTIAL = "POTENTIAL FRAUD"
LEVEL_FRAUD = "FRAUD"
LEVEL_ORDER = [LEVEL_NORMAL, LEVEL_POTENTIAL, LEVEL_FRAUD]
POTENTIAL_FACTOR = 0.9

LEVEL_META = {
    LEVEL_NORMAL: {
        "color": "#27ae60",
        "desc": "Probabilitas fraud jauh di bawah ambang.",
        "action": "Proses transaksi secara normal.",
    },
    LEVEL_POTENTIAL: {
        "color": "#f39c12",
        "desc": "Probabilitas fraud mendekati ambang. Perlu diperiksa.",
        "action": "Tahan sebentar, verifikasi manual.",
    },
    LEVEL_FRAUD: {
        "color": "#c0392b",
        "desc": "Probabilitas fraud di atas ambang optimal F1.",
        "action": "Tolak transaksi, arahkan ke investigasi.",
    },
}

TEST_METRICS = {
    "Precision": 0.9310,
    "Recall": 0.7606,
    "F1-Score": 0.8372,
    "ROC-AUC": 0.9715,
    "PR-AUC": 0.7878,
    "Fraud Detection Rate (TPR)": 0.7606,
    "False Alarm Rate (FPR)": 0.0001,
}


class ArtifactError(RuntimeError):
    pass


class SamplePoolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactPaths:
    models_dir: Path
    data_dir: Path

    @property
    def model(self) -> Path:
        return self.models_dir / "xgb_model.json"

    @property
    def threshold(self) -> Path:
        return self.models_dir / "threshold.pkl"

    @property
    def scaler(self) -> Path:
        return self.data_dir / "scaler.pkl"

    @property
    def samples(self) -> Path:
        return self.data_dir / "samples.csv"


def default_paths(root: Path | None = None) -> ArtifactPaths:
    root = root or Path(__file__).resolve().parent.parent
    return ArtifactPaths(
        models_dir=Path(os.environ.get("FRAUD_MODELS_DIR", root / "data" / "models")),
        data_dir=Path(os.environ.get("FRAUD_DATA_DIR", root / "data")),
    )


def add_engineered_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Time_hour"] = (df["Time"] // 3600) % 24
    df["Time_day"] = df["Time"] // 86400
    return df


def describe_time(time_sec: float) -> dict:
    hour = int((time_sec % 86400) // 3600)
    return {"hour": hour, "text": f"{hour:02d}:00", "jam_sepi": bool(0 <= hour < 5)}


def format_class_label(label: int) -> str:
    label = int(label)
    return f"{CLASS_LABELS.get(label, 'Tidak Diketahui')} ({label})"


class FraudDetector:
    def __init__(self, booster, scaler, optimal_threshold,
                 n_trees=None, optimal_f1=None):
        self.booster = booster
        self.scaler = scaler
        self.optimal_threshold = float(optimal_threshold)
        self.optimal_f1 = None if optimal_f1 is None else float(optimal_f1)
        self.n_trees = n_trees

    @property
    def potential_threshold(self) -> float:
        return self.optimal_threshold * POTENTIAL_FACTOR

    def classify(self, proba: float, threshold: float | None = None) -> str:
        th = self.optimal_threshold if threshold is None else float(threshold)
        if proba >= th:
            return LEVEL_FRAUD
        if proba >= th * POTENTIAL_FACTOR:
            return LEVEL_POTENTIAL
        return LEVEL_NORMAL

    def predict_row(self, row: pd.Series) -> dict:
        X_raw = add_engineered_features(row.to_frame().T)[FEATURES].astype(float)
        X_scaled = np.asarray(self.scaler.transform(X_raw), dtype=float)
        d = xgb.DMatrix(pd.DataFrame(X_scaled, columns=FEATURES))
        kw = {"iteration_range": (0, self.n_trees)} if self.n_trees else {}

        # Probabilitas diambil dari XGBoost, bukan sigmoid dari margin float64.
        # Ada baris di samples.csv yang probabilitasnya persis sama dengan
        # optimal_threshold, dan sigmoid float64 menggesernya ke bawah sehingga
        # verdict-nya terbalik jadi NORMAL.
        proba = float(np.asarray(self.booster.predict(d, **kw)).ravel()[0])
        contribs = np.asarray(
            self.booster.predict(d, pred_contribs=True, **kw), dtype=float
        ).ravel()

        return {
            "proba": proba,
            "level": self.classify(proba),
            "contribs": contribs,
            "X_raw": X_raw.values,
        }


class SamplePool:
    def __init__(self, frame: pd.DataFrame, source: str = "<memory>"):
        missing = [c for c in RAW_COLS + [LABEL] if c not in frame.columns]
        if missing:
            raise SamplePoolError(
                f"{source} tidak bisa dipakai sebagai pool demo. "
                f"Kolom kurang: {missing[:5]}...\n"
                "Jalankan notebooks/prepare_app_samples.ipynb untuk membuatnya."
            )

        bad = sorted(set(frame[LABEL].dropna().unique()) - {CLASS_NORMAL, CLASS_FRAUD})
        if bad:
            raise SamplePoolError(f"Kolom {LABEL} harus hanya berisi 0 dan 1, ada: {bad}")

        self.source = source
        self._frame = frame.reset_index(drop=True)
        self._by_class = {
            CLASS_NORMAL: self._frame[self._frame[LABEL] == CLASS_NORMAL].reset_index(drop=True),
            CLASS_FRAUD: self._frame[self._frame[LABEL] == CLASS_FRAUD].reset_index(drop=True),
        }
        for cls, sub in self._by_class.items():
            if sub.empty:
                raise SamplePoolError(
                    f"{source} tidak punya baris untuk kelas "
                    f"{CLASS_LABELS[cls]} — kelas itu tidak bisa didemokan."
                )

    @classmethod
    def from_csv(cls, path: Path) -> "SamplePool":
        path = Path(path)
        if not path.exists():
            raise SamplePoolError(
                f"Pool sampel tidak ditemukan: {path}\n"
                "Jalankan notebooks/prepare_app_samples.ipynb lebih dulu."
            )

        header = pd.read_csv(path, nrows=0).columns
        missing = [c for c in RAW_COLS + [LABEL] if c not in header]
        if missing:
            raise SamplePoolError(
                f"{path.name} tidak bisa dipakai sebagai pool demo. "
                f"Kolom kurang: {missing[:5]}\n"
                "Jalankan notebooks/prepare_app_samples.ipynb untuk membuatnya."
            )

        return cls(pd.read_csv(path, usecols=RAW_COLS + [LABEL]), source=str(path))

    @property
    def n_total(self) -> int:
        return len(self._frame)

    def count(self, label: int) -> int:
        return len(self._class_frame(label))

    def draw(self, label: int, rng: random.Random | None = None) -> pd.Series:
        sub = self._class_frame(label)
        return sub.iloc[(rng or random).randrange(len(sub))]

    def draw_any(self, rng: random.Random | None = None) -> pd.Series:
        return self._frame.iloc[(rng or random).randrange(len(self._frame))]

    def _class_frame(self, label: int) -> pd.DataFrame:
        label = int(label)
        if label not in self._by_class:
            raise SamplePoolError(
                f"Label {label} tidak dikenal. Pilihan: {sorted(self._by_class)}"
            )
        return self._by_class[label]


def load_detector(paths: ArtifactPaths | None = None) -> FraudDetector:
    paths = paths or default_paths()

    missing = [str(p) for p in (paths.model, paths.threshold, paths.scaler)
               if not p.exists()]
    if missing:
        raise ArtifactError(
            "Artefak berikut tidak ditemukan:\n  - " + "\n  - ".join(missing) +
            "\n\nSalin folder data/models/ dan data/scaler.pkl ke project ini, "
            "atau set FRAUD_MODELS_DIR / FRAUD_DATA_DIR."
        )

    booster = xgb.Booster()
    booster.load_model(str(paths.model))

    try:
        n_trees = int(booster.best_iteration) + 1
    except AttributeError:
        n_trees = int(booster.num_boosted_rounds())

    with open(paths.scaler, "rb") as fh:
        scaler = pickle.load(fh)
    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise ArtifactError(
            f"Scaler di-fit pada {getattr(scaler, 'n_features_in_', '?')} fitur, "
            f"tidak cocok dengan FEATURES ({len(FEATURES)})"
        )

    with open(paths.threshold, "rb") as fh:
        info = pickle.load(fh)
    if "optimal_threshold" not in info:
        raise ArtifactError(
            f"{paths.threshold} tidak punya kunci 'optimal_threshold'. "
            f"Kunci yang ada: {sorted(info)}"
        )

    return FraudDetector(
        booster=booster,
        scaler=scaler,
        optimal_threshold=float(info["optimal_threshold"]),
        n_trees=n_trees,
        optimal_f1=info.get("optimal_f1_val"),
    )


def load_sample_pool(paths: ArtifactPaths | None = None) -> SamplePool:
    paths = paths or default_paths()
    return SamplePool.from_csv(paths.samples)