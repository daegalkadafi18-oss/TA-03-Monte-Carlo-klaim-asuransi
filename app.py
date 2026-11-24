"""Aplikasi web (Streamlit) - TA-03 Monte Carlo: Estimasi Risiko Klaim Asuransi."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Monte Carlo Klaim Asuransi", layout="wide")
st.title("Simulasi Monte Carlo: Risiko Total Klaim Melebihi Anggaran")
st.caption("Dataset: Medical Cost Personal Datasets (Kaggle), fitur 'charges'. "
           "Nama: Daegal Fauzan Kadafi | NIM: 301240005")

up = st.sidebar.file_uploader("Upload CSV (opsional)", type="csv")
if up is not None:
    df = pd.read_csv(up)
elif os.path.exists("data.csv"):
    df = pd.read_csv("data.csv")
else:
    st.info("Unggah file CSV dataset di sidebar untuk memulai.")
    st.stop()

if "charges" not in df.columns:
    st.error("Kolom 'charges' tidak ditemukan. Kolom tersedia: " + ", ".join(df.columns))
    st.stop()

df = df.dropna(subset=["charges"]).drop_duplicates()
q1, q3 = df["charges"].quantile([0.25, 0.75])
upper = q3 + 3 * (q3 - q1)
remove_out = st.sidebar.checkbox("Hapus outlier ekstrem (3 x IQR)", value=True)
if remove_out:
    df = df[df["charges"] <= upper]

charges = df["charges"].values
mu, sigma = charges.mean(), charges.std(ddof=1)
log_c = np.log(charges)
mu_log, sd_log = log_c.mean(), log_c.std(ddof=1)

st.subheader("Parameter dari data historis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah data", f"{len(charges):,}")
c2.metric("Mean (μ)", f"{mu:,.0f}")
c3.metric("Std Dev (σ)", f"{sigma:,.0f}")
c4.metric("Min / Max", f"{charges.min():,.0f} / {charges.max():,.0f}")

with st.expander("Histogram data asli"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    ax[0].hist(charges, bins=40, color="steelblue", edgecolor="white")
    ax[0].axvline(mu, color="red", linestyle="--"); ax[0].set_title("Charges")
    ax[1].hist(log_c, bins=40, color="orange", edgecolor="white"); ax[1].set_title("log(Charges)")
    st.pyplot(fig)

st.sidebar.header("Pengaturan simulasi")
n_sim = st.sidebar.slider("Jumlah skenario", 5000, 50000, 10000, 1000)
n_pat = st.sidebar.slider("Jumlah pasien baru", 10, 500, 100, 10)
budget = st.sidebar.number_input("Anggaran perusahaan", value=float(round(1.10 * n_pat * mu, -4)), step=10000.0)
dist = st.sidebar.selectbox("Distribusi klaim", ["lognormal", "normal"])
seed = st.sidebar.number_input("Seed (NIM)", value=301240005, step=1)

if st.sidebar.button("Jalankan simulasi", type="primary"):
    np.random.seed(int(seed))
    if dist == "lognormal":
        claims = np.random.lognormal(mu_log, sd_log, (n_sim, n_pat))
    else:
        claims = np.clip(np.random.normal(mu, sigma, (n_sim, n_pat)), charges.min(), None)
    total = claims.sum(axis=1)
    p_exceed = float(np.mean(total > budget))

    st.subheader("Hasil")
    a, b, c = st.columns(3)
    a.metric("P(total klaim > anggaran)", f"{p_exceed:.2%}")
    b.metric("Rata-rata total klaim", f"{total.mean():,.0f}")
    c.metric("Persentil 95%", f"{np.percentile(total, 95):,.0f}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _, bins, patches = ax.hist(total, bins=50, color="steelblue", edgecolor="white")
    for bb, p in zip(bins[:-1], patches):
        if bb >= budget:
            p.set_facecolor("tomato")
    ax.axvline(budget, color="black", linestyle="--", label=f"Anggaran {budget:,.0f}")
    ax.axvline(total.mean(), color="green", linestyle=":", label="Mean")
    ax.set_xlabel("Total klaim"); ax.set_ylabel("Frekuensi"); ax.legend()
    st.pyplot(fig)
else:
    st.write("Atur parameter di sidebar lalu klik **Jalankan simulasi**.")
