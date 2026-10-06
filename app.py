from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.detector import (  # noqa: E402
    CLASS_FRAUD,
    CLASS_LABELS,
    CLASS_NORMAL,
    FEATURES,
    LABEL,
    LEVEL_FRAUD,
    LEVEL_META,
    LEVEL_NORMAL,
    LEVEL_ORDER,
    TEST_METRICS,
    ArtifactError,
    SamplePoolError,
    add_engineered_features,
    describe_time,
    format_class_label,
    load_detector,
    load_sample_pool,
)

st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="💳",
    layout="wide",
)

SUMBER_RANDOM = "Random"

OPSI_SUMBER = {
    CLASS_LABELS[CLASS_FRAUD]: CLASS_FRAUD,
    CLASS_LABELS[CLASS_NORMAL]: CLASS_NORMAL,
    SUMBER_RANDOM: SUMBER_RANDOM,
}

@st.cache_resource(show_spinner="Memuat model...")
def get_detector():
    return load_detector()

@st.cache_resource(show_spinner="Memuat pool sampel...")
def get_pool():
    return load_sample_pool()


def format_proba(p: float) -> str:
    if p < 0.001:
        return "< 0,1%"
    if p > 0.999:
        return "> 99,9%"
    return f"{p:.4f}".replace(".", ",")


try:
    detector = get_detector()
    pool = get_pool()
except ArtifactError as exc:
    st.error("Artefak model belum lengkap")
    st.code(str(exc), language="text")
    st.stop()
except SamplePoolError as exc:
    st.error("Pool sampel belum siap")
    st.code(str(exc), language="text")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Gagal memuat artefak: {exc}")
    st.stop()


st.sidebar.title("Performance Metrics Model")

st.sidebar.dataframe(
    pd.DataFrame(
        [{"Metric": k, "Value": f"{v:.4f}"} for k, v in TEST_METRICS.items()]
    ),
    width="stretch",
    hide_index=True,
)

st.sidebar.divider()
st.sidebar.caption(f"Fraud Threshold: `{detector.optimal_threshold:.4f}`")

st.sidebar.divider()
st.sidebar.subheader("Source dataset: Kaggle")
st.sidebar.markdown(
    "[Credit Card Fraud Detection]"
    "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)"
)

st.session_state.setdefault("transaksi", None)
st.session_state.setdefault("hasil", None)
st.session_state.setdefault("riwayat", [])

st.sidebar.divider()
st.sidebar.subheader("Session history")
jumlah_riwayat = len(st.session_state.riwayat)
st.sidebar.caption(f"{jumlah_riwayat} transaksi dideteksi.")

if st.sidebar.button("🗑️ Hapus riwayat", disabled=jumlah_riwayat == 0,
                     width="stretch"):
    st.session_state.riwayat = []
    st.session_state.hasil = None
    st.session_state.transaksi = None

st.title("Deteksi Kartu Kredit Fraud Menggunakan XGBoost")
st.caption(
    "Pilih jenis transaksi dan ambil satu sample secara acak dari sample, lalu jalankan model untuk melihat hasilnya."
)


st.subheader("1. Pilih jenis transaksi")

pilihan = st.radio(
    "Transaksi yang ingin diambil",
    options=list(OPSI_SUMBER),
    horizontal=True,
)
sumber = OPSI_SUMBER[pilihan]

jumlah_tersedia = pool.n_total if sumber == SUMBER_RANDOM else pool.count(sumber)
st.caption(f"{jumlah_tersedia:,} transaksi tersedia untuk pilihan ini.")

if st.button("🎲 Ambil acak", type="primary", width="stretch"):
    rng = random.Random()
    baris = pool.draw_any(rng) if sumber == SUMBER_RANDOM else pool.draw(sumber, rng)
    st.session_state.transaksi = {
        "baris": baris,
        "sumber": sumber,
    }
    st.session_state.hasil = None


transaksi = st.session_state.transaksi

if not transaksi:
    st.info("Pilih jenis transaksi lalu klik **Ambil acak**.", icon="ℹ️")
    st.stop()

baris = transaksi["baris"]
waktu = describe_time(float(baris["Time"]))
amount_log = float(np.log1p(baris["Amount"]))
fe_row = add_engineered_features(baris.to_frame().T)
label_asli_baris = int(baris[LABEL])

st.subheader("2. Rincian transaksi")

sumber = transaksi["sumber"]
nama_sumber = (format_class_label(sumber)
               if sumber != SUMBER_RANDOM else "Random")

k1, k2, k3 = st.columns(3)
k1.metric("Nominal", f"{float(baris['Amount']):,.2f}".replace(",", " ") + " EUR")
k2.metric("Label", format_class_label(label_asli_baris))

st.dataframe(
    pd.DataFrame({
        "Informasi": ["Nominal (EUR)", "Amount_log", "Jam", "Label asli"],
        "Nilai": [
            f"{float(baris['Amount']):,.2f}".replace(",", " "),
            f"{amount_log:.4f}".replace(".", ","),
            waktu["text"],
            format_class_label(label_asli_baris),
        ],
    }),
    width="stretch",
    hide_index=True,
)

if waktu["jam_sepi"]:
    st.caption("Jam sepi (00:00–05:00), kemungkinan fraud lebih tinggi.")

with st.expander("Lihat 31 fitur model"):
    st.dataframe(
        pd.DataFrame({
            "Fitur": FEATURES,
            "Nilai": fe_row[FEATURES].to_numpy(dtype=float).ravel(),
        }),
        width="stretch",
        hide_index=True,
    )


if st.button("🔍 Deteksi", type="primary", width="stretch"):
    prediksi = detector.predict_row(baris)
    label_asli = int(baris[LABEL])
    level = prediksi["level"]
    benar = (level == LEVEL_FRAUD) == (label_asli == CLASS_FRAUD)

    st.session_state.hasil = {
        "proba": prediksi["proba"],
        "level": level,
        "contribs": prediksi["contribs"],
        "label_asli": label_asli,
        "benar": benar,
    }
    st.session_state.riwayat.append({
        "Label asli": format_class_label(label_asli),
        "Prediksi": level,
        "Probabilitas": prediksi["proba"],
        "Benar": "ya" if benar else "tidak",
        "Nominal": float(baris["Amount"]),
        "Jam": waktu["text"],
    })


hasil = st.session_state.hasil

if not hasil:
    st.info("Klik **Deteksi** untuk menjalankan model pada transaksi ini.",
            icon="ℹ️")
    st.stop()

proba = hasil["proba"]
level = hasil["level"]
label_asli = hasil["label_asli"]
benar = hasil["benar"]
meta = LEVEL_META[level]

st.subheader("3. Hasil deteksi")

warna = meta["color"] if benar else "#7f8c8d"

st.markdown(
    f"""
    <div style="border-left:8px solid {warna};background:{warna}14;
                padding:18px 22px;border-radius:8px">
      <div style="font-size:13px;color:#666;text-transform:uppercase;
                  letter-spacing:1px">Prediksi model</div>
      <div style="font-size:28px;font-weight:800;color:{warna};
                  margin-top:2px">{level}</div>
      <div style="font-size:14px;color:#555;margin-top:6px">{meta['desc']}</div>
      <div style="font-size:15px;margin-top:10px">
        <b>Tindakan:</b> {meta['action']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Hasil", "Fraud" if benar else "Normal")
c2.metric("Probabilitas fraud", format_proba(proba))
c3.metric("Label asli", format_class_label(label_asli))

if benar:
    st.success(
        f"Model menebak dengan **benar**. Label asli `{format_class_label(label_asli)}`, "
        f"model menyebut `{level}`."
    )
else:
    st.error(
        f"Model menebak dengan **salah**. Label asli `{format_class_label(label_asli)}`, "
        f"tapi model menyebut `{level}` dengan probabilitas {format_proba(proba)}."
    )

with st.expander("Mengapa model memutuskan ini?"):
    contrib_df = pd.DataFrame({
        "Fitur": FEATURES,
        "Kontribusi": hasil["contribs"][:-1],
    })
    contrib_df["Abs"] = contrib_df["Kontribusi"].abs()
    top = contrib_df.sort_values("Abs", ascending=False).head(10).iloc[::-1]

    fig = go.Figure(go.Bar(
        x=top["Kontribusi"], y=top["Fitur"], orientation="h",
        marker=dict(color=[
            LEVEL_META[LEVEL_FRAUD]["color"] if v > 0
            else LEVEL_META[LEVEL_NORMAL]["color"]
            for v in top["Kontribusi"]
        ]),
    ))
    fig.add_vline(x=0, line=dict(color="#2c3e50", width=1))
    fig.update_layout(
        height=320, xaxis_title="Kontribusi (logit)", yaxis_title="",
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        contrib_df.sort_values("Abs", ascending=False).drop(columns="Abs").round(4),
        width="stretch", hide_index=True,
    )


st.subheader("4. Riwayat sesi")

if st.session_state.riwayat:
    hist = pd.DataFrame(st.session_state.riwayat)
    st.dataframe(hist.iloc[::-1], width="stretch", hide_index=True)

    tepat = int((hist["Benar"] == "ya").sum())
    salah = int((hist["Benar"] == "tidak").sum())
    total = tepat + salah
    st.caption(
        f"{tepat} tepat dari {total} deteksi ({100 * tepat / total:.1f}%). "
        "Hapus dari sidebar."
    )

    counts = hist["Prediksi"].value_counts()
    kolom = st.columns(len(LEVEL_ORDER))
    for col, lvl in zip(kolom, LEVEL_ORDER):
        col.metric(lvl, int(counts.get(lvl, 0)))
else:
    st.caption("Belum ada transaksi dideteksi.")
