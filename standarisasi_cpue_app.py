import base64
import io
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import bs
from PIL import Image
from scipy.interpolate import make_interp_spline
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# =========================================================
# 1. KONFIGURASI HALAMAN & STYLING STREAMLIT
# =========================================================
try:
  logo = Image.open("_ MDPI Primary Logo.png")
except Exception:
  logo = "🐟"

st.set_page_config(
    page_title="Standarisasi CPUE YFT - MDPI", page_icon=logo, layout="wide"
)

# Custom CSS termasuk penyesuaian ukuran font pada st.metric
st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="metric-container"] {
        background-color: #f7f9fc;
        border: 1px solid #e1e4e8;
        padding: 12px 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #0E4C92;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 20px !important;
        font-weight: bold;
        color: #0E4C92;
        word-break: break-word;
    }
    div[data-testid="stMetricLabel"] > label {
        font-size: 13px !important;
        color: #555555;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_base64_image(image_path):
  try:
    with open(image_path, "rb") as img_file:
      return base64.b64encode(img_file.read()).decode()
  except Exception:
    return None


logo_base64 = get_base64_image("_ MDPI Primary Logo.png")


def render_footer():
  st.markdown("---")
  if logo_base64:
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 15px; margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" width="170" style="margin-bottom: 10px;">
            <p style="font-size: 13px; color: #666666; margin: 0;">
                &copy; 2026 Yayasan MDPI. Hak Cipta Dilindungi. <i>Happy People Many Fish</i>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 15px; margin-bottom: 20px; color: #666666;">
            <p style="font-size: 13px; margin: 0;">
                &copy; 2026 Yayasan MDPI. Hak Cipta Dilindungi. <i>Happy People Many Fish</i>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Header Utama Dashboard
st.markdown(
    """
    <h1 style='color:#0E4C92; margin-bottom:0px;'>Aplikasi Standarisasi Catch Per Unit Effort (CPUE)</h1>
    <h3 style='color:#444444; margin-top:5px;'>Pemodelan GLM & GAM untuk Standarisasi Catch Per Unit Effort</h3>
    <p style='color:#666666;'>**Tahap Uji coba</p>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================================================
# 2. PETUNJUK STRUKTUR & UPLOAD FILE DATA EXCEL
# =========================================================
uploaded_file = st.file_uploader(
    "Upload file Excel Data Catch & Effort (.xlsx)", type=["xlsx"]
)

if uploaded_file is None:
  st.info(
      "**Silakan unggah file Excel (.xlsx)** yang berisi data operasional"
      " penangkapan untuk memulai proses analisis."
  )

  st.markdown(
      """
      <h3 style="font-size:18px; color:#0E4C92; margin-top:20px;">
      Petunjuk Struktur Kolom File Excel
      </h3>
      """,
      unsafe_allow_html=True,
  )

  petunjuk = pd.DataFrame({
      "Nama Kolom": [
          "berat_kg",
          "id_trip",
          "tahun",
          "bulan",
          "musim",
          "quarter",
          "jumlah_hari_memancing",
          "abk",
          "kapasitas_mesin",
          "panjang_kapal",
          "gross_tonnage",
          "teknik_penangkapan",
          "jenis_alat_tangkap",
          "daerah_spasial",
          "sst",
          "chl_a",
      ],
      "Tipe Data": [
          "Numerik (kg)",
          "Teks / Numerik",
          "Numerik (YYYY)",
          "Numerik (1–12)",
          "Teks (Barat / Timur / Peralihan)",
          "Numerik (1–4)",
          "Numerik (Hari)",
          "Numerik (Orang)",
          "Numerik (PK / HP)",
          "Numerik (Meter)",
          "Numerik (GT)",
          "Teks (Rumpon / Non Rumpon / Campuran)",
          "Teks (Handline / Longline / dll)",
          "Teks (Grid / WPP / Nama Perairan)",
          "Numerik (°C)",
          "Numerik (mg/m³)",
      ],
      "Status": ["Wajib (Target)"] + ["Opsional"] * 15,
      "Keterangan": [
          "Total berat hasil tangkapan (Variabel Target)",
          "Identitas unik perjalanan/trip penangkapan (Identifier)",
          "Tahun operasional penangkapan",
          "Bulan operasional penangkapan",
          "Musim penangkapan",
          "Kuartal tahunan (Q1–Q4)",
          "Jumlah hari memancing dalam 1 trip (Effort/Offset)",
          "Jumlah Anak Buah Kapal",
          "Kapasitas daya mesin kapal",
          "Panjang dimensi kapal",
          "Ukuran tonase kotor kapal (GT)",
          "Metode/teknik penangkapan",
          "Spesifikasi alat tangkap yang digunakan",
          "Lokasi/Grid/WPP/Wilayah spasial penangkapan",
          "Suhu Permukaan Laut (Sea Surface Temperature)",
          "Konsentrasi Klorofil-a",
      ],
  })

  st.dataframe(
      petunjuk, use_container_width=True, hide_index=True, height=520
  )

  render_footer()
  st.stop()

# Membaca Data
df = pd.read_excel(uploaded_file)

# Validasi Variabel Target Wajib
if "berat_kg" not in df.columns:
  st.error(
      "❌ Kolom target **'berat_kg'** tidak ditemukan dalam file Excel. Mohon"
      " pastikan nama kolom target sesuai."
  )
  render_footer()
  st.stop()

# =========================================================
# 3. PRE-PROCESSING & DETEKSI VARIABEL DINAMIS (SAFE FILTER)
# =========================================================
id_cols = ["id_trip", "id", "trip_id", "kode_trip"]

cat_candidates = [
    "tahun",
    "bulan",
    "musim",
    "quarter",
    "teknik_penangkapan",
    "jenis_alat_tangkap",
    "daerah_spasial",
    "daerah",
]
num_candidates = [
    "abk",
    "panjang_kapal",
    "kapasitas_mesin",
    "gross_tonnage",
    "gt",
    "sst",
    "chl_a",
]
effort_candidates = ["jumlah_hari_memancing", "days_at_sea", "das", "effort"]

avail_cats = [c for c in cat_candidates if c in df.columns]
avail_nums = [c for c in num_candidates if c in df.columns]
effort_col = next((c for c in effort_candidates if c in df.columns), None)

df["berat_kg"] = pd.to_numeric(df["berat_kg"], errors="coerce")
for c in avail_nums:
  df[c] = pd.to_numeric(df[c], errors="coerce")

if effort_col:
  df[effort_col] = pd.to_numeric(df[effort_col], errors="coerce")
  df = df[df[effort_col] > 0].copy()
  df["log_effort"] = np.log(df[effort_col])
else:
  df["log_effort"] = 0.0

used_cols = ["berat_kg"] + avail_cats + avail_nums
if effort_col:
  used_cols.append(effort_col)

df_model = df.dropna(subset=used_cols).copy()

valid_cats = []
for c in avail_cats:
  n_unq = df_model[c].nunique()
  if 1 < n_unq < (len(df_model) * 0.5):
    df_model[c] = df_model[c].astype(str)
    valid_cats.append(c)

valid_nums = []
for c in avail_nums:
  if df_model[c].nunique() > 1:
    valid_nums.append(c)

if len(df_model) < 10:
  st.error("❌ Jumlah data valid terlalu sedikit untuk melakukan pemodelan.")
  st.stop()

# =========================================================
# 4. PEMBENTUKAN FORMULA & PEMODELAN GLM / GAM DENGAN TRY-EXCEPT
# =========================================================
glm_terms = [f"C({c})" for c in valid_cats] + valid_nums
if not glm_terms:
  st.error(
      "❌ Tidak ada variabel prediktor yang valid untuk dimodelkan (semua"
      " kolom berkategori unik tunggal atau bernilai ID)."
  )
  st.stop()

formula_glm = "berat_kg ~ " + " + ".join(glm_terms)

gam_terms = [f"C({c})" for c in valid_cats]
for c in valid_nums:
  if df_model[c].nunique() > 4:
    gam_terms.append(f"bs({c}, df=4)")
  else:
    gam_terms.append(c)

formula_gam = "berat_kg ~ " + " + ".join(gam_terms)

models = {}
with st.spinner("Sedang melatih model GLM & GAM..."):
  try:
    models["GLM Poisson"] = smf.glm(
        formula=formula_glm,
        data=df_model,
        offset=df_model["log_effort"],
        family=sm.families.Poisson(link=sm.families.links.Log()),
    ).fit()
  except Exception:
    pass

  try:
    models["Tweedie"] = smf.glm(
        formula=formula_glm,
        data=df_model,
        offset=df_model["log_effort"],
        family=sm.families.Tweedie(
            var_power=1.5, link=sm.families.links.Log()
        ),
    ).fit()
  except Exception:
    pass

  try:
    models["GLM Negative Binomial"] = smf.glm(
        formula=formula_glm,
        data=df_model,
        offset=df_model["log_effort"],
        family=sm.families.NegativeBinomial(alpha=1.0),
    ).fit()
  except Exception:
    pass

  try:
    models["GAM / Spline Negative Binomial"] = smf.glm(
        formula=formula_gam,
        data=df_model,
        offset=df_model["log_effort"],
        family=sm.families.NegativeBinomial(alpha=1.0),
    ).fit()
  except Exception:
    pass

if not models:
  st.error(
      "❌ Seluruh model gagal konvergen. Periksa kembali korelasi antar variabel"
      " atau pastikan nilai variabel target tidak bernilai negatif/ekstrem."
  )
  st.stop()

metrics = []
for name, mod in models.items():
  pseudo_r2 = 1 - (mod.deviance / mod.null_deviance)
  metrics.append({
      "Model": name,
      "AIC": mod.aic,
      "Deviance": mod.deviance,
      "Null_Deviance": mod.null_deviance,
      "Pseudo_R2": pseudo_r2,
      "N": int(mod.nobs),
  })

metrics_df = pd.DataFrame(metrics).sort_values(by="AIC").reset_index(drop=True)
metrics_df["Delta_AIC"] = metrics_df["AIC"] - metrics_df["AIC"].min()

best_model_name = metrics_df.loc[metrics_df["AIC"].idxmin(), "Model"]
model_fix = models[best_model_name]

# =========================================================
# 5. DASHBOARD & HASIL ANALISIS (TAB INTEGRASI)
# =========================================================
st.caption(
    f"**Status Analisis:** Berhasil memproses **{len(df_model):,}** observasi"
    f" trip. Prediktor aktif: **{len(valid_cats)}** Kategorikal,"
    f" **{len(valid_nums)}** Numerik."
)

tab1, tab2, tab3 = st.tabs(
    ["📊 Evaluasi Model", "📈 Efek Parsial Parameter", "🐟 CPUE Terstandarisasi"]
)

# --- TAB 1: EVALUASI MODEL ---
with tab1:
  # 1. TABEL RINGKASAN STATISTIK DESKRIPTIF VARIABEL
  st.subheader("Ringkasan Statistik Deskriptif Variabel")

  stat_rows = []
  num_list = list(
      dict.fromkeys(
          ["berat_kg"] + ([effort_col] if effort_col else []) + valid_nums
      )
  )

  for col in num_list:
    if col in df_model.columns:
      s = df_model[col]
      stat_rows.append({
          "Nama Variabel": col,
          "Tipe Data": "Numerik",
          "Jumlah (N)": f"{len(s):,}",
          "Mean ± Std": f"{s.mean():.2f} ± {s.std():.2f}",
          "Min": f"{s.min():.2f}",
          "Median": f"{s.median():.2f}",
          "Max": f"{s.max():.2f}",
          "Keterangan / Modus": "-",
      })

  for col in valid_cats:
    if col in df_model.columns:
      s = df_model[col]
      mode_val = s.mode()[0] if not s.mode().empty else "-"
      stat_rows.append({
          "Nama Variabel": col,
          "Tipe Data": "Kategorikal",
          "Jumlah (N)": f"{len(s):,}",
          "Mean ± Std": "-",
          "Min": "-",
          "Median": "-",
          "Max": "-",
          "Keterangan / Modus": f"{s.nunique()} Kat. (Modus: {mode_val})",
      })

  df_stat_summary = pd.DataFrame(stat_rows)

  col_stat, _ = st.columns([4, 1])
  with col_stat:
    st.dataframe(df_stat_summary, use_container_width=True, hide_index=True)

  st.markdown("---")

  # 2. RINGKASAN PERBANDINGAN MODEL
  st.subheader("Ringkasan Perbandingan Model")

  col_m1, col_m2, col_m3 = st.columns(3)
  col_m1.metric("Model Terbaik Terpilih", best_model_name)
  col_m2.metric("AIC Terendah", f"{metrics_df['AIC'].min():,.2f}")
  col_m3.metric("Total Sampel Valid", f"{len(df_model):,} Data")

  st.markdown("**Tabel Perbandingan Kinerja Model (Sorted by AIC)**")
  metrics_display = metrics_df.copy()
  metrics_display["AIC"] = metrics_display["AIC"].round(2)
  metrics_display["Deviance"] = metrics_display["Deviance"].round(2)
  metrics_display["Null_Deviance"] = metrics_display["Null_Deviance"].round(2)
  metrics_display["Pseudo_R2"] = metrics_display["Pseudo_R2"].round(4)
  metrics_display["Delta_AIC"] = metrics_display["Delta_AIC"].round(2)

  col_tbl, _ = st.columns([4, 1])
  with col_tbl:
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

  if "GLM Poisson" in models:
    disp_pois = (
        models["GLM Poisson"].pearson_chi2 / models["GLM Poisson"].df_resid
    )
    st.caption(
        f"ℹ️ **Overdispersion Ratio (GLM Poisson):** {disp_pois:.4f} "
        + (
            "(Indikasi overdispersion tinggi, model NB/Tweedie lebih"
            " direkomendasikan)."
            if disp_pois > 1.5
            else ""
        )
    )

  st.markdown("---")
  st.subheader("Residual Plot Model")

  fig_res, axes = (
      plt.subplots(2, 2, figsize=(12, 8))
      if len(models) > 1
      else plt.subplots(1, 1, figsize=(6, 4))
  )
  axes_list = axes.flatten() if len(models) > 1 else [axes]

  for idx, (name, mod) in enumerate(models.items()):
    axes_list[idx].scatter(
        mod.fittedvalues, mod.resid_response, alpha=0.3, s=12, color="#0E4C92"
    )
    axes_list[idx].axhline(y=0, linestyle="--", linewidth=1, color="red")
    axes_list[idx].set_title(
        f"Residuals: {name}", fontsize=10, fontweight="bold"
    )
    axes_list[idx].set_xlabel("Fitted Values", fontsize=8)
    axes_list[idx].set_ylabel("Response Residuals", fontsize=8)

  for i in range(len(models), len(axes_list)):
    fig_res.delaxes(axes_list[i])

  plt.tight_layout()
  st.pyplot(fig_res)
  plt.close(fig_res)

# --- TAB 2: EFEK PARSIAL DINAMIS ---
with tab2:
  st.subheader(f"Plot Efek Parsial (Model Terbaik: {best_model_name})")

  defaults = {"log_effort": 0.0}
  for c in valid_cats:
    defaults[c] = df_model[c].mode()[0]
  for c in valid_nums:
    defaults[c] = df_model[c].mean()

  def make_dummy(override_col, override_vals):
    d = {k: v for k, v in defaults.items()}
    d[override_col] = override_vals
    return pd.DataFrame(d)

  total_plots = len(valid_nums) + len(valid_cats)
  cols_per_row = 3
  rows = int(np.ceil(total_plots / cols_per_row))

  fig_grid, axes = plt.subplots(
      rows, cols_per_row, figsize=(15, max(4 * rows, 5))
  )
  axes_flat = axes.flatten() if total_plots > 1 else [axes]

  plot_idx = 0

  # Plot Numerik
  for col_name in valid_nums:
    ax = axes_flat[plot_idx]
    grid = np.linspace(df_model[col_name].min(), df_model[col_name].max(), 150)
    pred = model_fix.get_prediction(make_dummy(col_name, grid))
    fit = pred.predicted_mean - pred.predicted_mean.mean()
    se = pred.se_mean

    ax.plot(grid, fit, "k-", lw=1.2)
    ax.plot(grid, fit + 1.96 * se, "k--", lw=0.8)
    ax.plot(grid, fit - 1.96 * se, "k--", lw=0.8)
    ax.plot(
        df_model[col_name],
        np.full_like(df_model[col_name], ax.get_ylim()[0]),
        "|k",
        ms=5,
        alpha=0.5,
    )
    ax.set_title(f"Effect: {col_name}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Partial effect", fontsize=8)
    plot_idx += 1

  # Plot Kategorikal
  for cat_col in valid_cats:
    ax = axes_flat[plot_idx]
    uniques = sorted(df_model[cat_col].dropna().unique())

    if len(uniques) > 12:
      uniques = df_model[cat_col].value_counts().index[:10].tolist()

    pred = model_fix.get_prediction(make_dummy(cat_col, uniques))
    fit = pred.predicted_mean - pred.predicted_mean.mean()
    se = pred.se_mean
    x_pos = np.arange(len(uniques))

    ax.bar(
        x_pos,
        fit,
        yerr=1.96 * se,
        color="grey",
        edgecolor="black",
        capsize=4,
        alpha=0.7,
    )
    ax.axhline(0, color="red", ls="--", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [str(u) for u in uniques], rotation=30, ha="right", fontsize=8
    )
    ax.set_title(f"Effect: {cat_col}", fontsize=10, fontweight="bold")
    plot_idx += 1

  for i in range(plot_idx, len(axes_flat)):
    fig_grid.delaxes(axes_flat[i])

  plt.tight_layout()
  st.pyplot(fig_grid)
  plt.close(fig_grid)

# --- TAB 3: STANDARISASI CPUE ---
with tab3:
  st.subheader("Hasil Standarisasi CPUE")

  # Pemetaan Nama Bulan
  month_map = {
      "1": "Jan",
      "2": "Feb",
      "3": "Mar",
      "4": "Apr",
      "5": "Mei",
      "6": "Jun",
      "7": "Jul",
      "8": "Agu",
      "9": "Sep",
      "10": "Okt",
      "11": "Nov",
      "12": "Des",
  }

  # 1. Standarisasi Tahunan (Jika Kolom 'tahun' Ada)
  if "tahun" in valid_cats:
    years = sorted(
        df_model["tahun"].unique(),
        key=lambda x: int(x) if str(x).isdigit() else x,
    )
    grid_yr_dict = {c: [defaults[c]] * len(years) for c in defaults}
    grid_yr_dict["tahun"] = years
    if effort_col:
      grid_yr_dict[effort_col] = 1.0
      grid_yr_dict["log_effort"] = 0.0

    grid_yr = pd.DataFrame(grid_yr_dict)
    pred_yr = model_fix.get_prediction(grid_yr).summary_frame()
    grid_yr["CPUE_std (kg/hari)"] = pred_yr["mean"]
    grid_yr["Lower CI"] = pred_yr["mean_ci_lower"]
    grid_yr["Upper CI"] = pred_yr["mean_ci_upper"]

    col_t1, col_t2 = st.columns([1, 1.5])
    with col_t1:
      st.markdown("**CPUE Standar Tahunan**")
      st.dataframe(
          grid_yr[
              ["tahun", "CPUE_std (kg/hari)", "Lower CI", "Upper CI"]
          ].round(2),
          use_container_width=True,
          hide_index=True,
      )

    with col_t2:
      fig_yr, ax_yr = plt.subplots(figsize=(7, 3.5))
      x_raw = np.arange(len(years))

      if len(years) > 2:
        x_smooth = np.linspace(x_raw.min(), x_raw.max(), 300)
        k_deg = min(3, len(years) - 1)

        spl_m = make_interp_spline(
            x_raw, grid_yr["CPUE_std (kg/hari)"], k=k_deg
        )
        spl_l = make_interp_spline(x_raw, grid_yr["Lower CI"], k=k_deg)
        spl_u = make_interp_spline(x_raw, grid_yr["Upper CI"], k=k_deg)

        ax_yr.fill_between(
            x_smooth,
            spl_l(x_smooth),
            spl_u(x_smooth),
            color="#E67E22",
            alpha=0.18,
            edgecolor="none",
        )
        ax_yr.plot(x_smooth, spl_m(x_smooth), color="#E67E22", linewidth=1.5)
        ax_yr.scatter(
            x_raw,
            grid_yr["CPUE_std (kg/hari)"],
            color="#D35400",
            s=20,
            zorder=5,
        )
      else:
        ax_yr.fill_between(
            x_raw,
            grid_yr["Lower CI"],
            grid_yr["Upper CI"],
            color="#E67E22",
            alpha=0.18,
            edgecolor="none",
        )
        ax_yr.plot(
            x_raw,
            grid_yr["CPUE_std (kg/hari)"],
            color="#E67E22",
            marker="o",
            markersize=4,
            linewidth=1.5,
        )

      ax_yr.set_xticks(x_raw)
      ax_yr.set_xticklabels(years)
      ax_yr.set_xlabel("Tahun")
      ax_yr.set_ylabel("CPUE Standar (kg/hari)")
      ax_yr.set_title("Tren CPUE Standar Tahunan", fontweight="bold")
      st.pyplot(fig_yr)
      plt.close(fig_yr)

    st.markdown("---")

  # 2. Standarisasi Musiman / Bulanan
  time_cat = next(
      (c for c in ["bulan", "musim", "quarter"] if c in valid_cats), None
  )
  if time_cat:
    # Mengurutkan waktu secara numerik agar 1..12 tidak terurut sebagai string
    time_units = sorted(
        df_model[time_cat].unique(),
        key=lambda x: int(x) if str(x).isdigit() else x,
    )

    grid_tm_dict = {c: [defaults[c]] * len(time_units) for c in defaults}
    grid_tm_dict[time_cat] = time_units
    if effort_col:
      grid_tm_dict[effort_col] = 1.0
      grid_tm_dict["log_effort"] = 0.0

    grid_tm = pd.DataFrame(grid_tm_dict)
    pred_tm = model_fix.get_prediction(grid_tm).summary_frame()
    grid_tm["CPUE_std (kg/hari)"] = pred_tm["mean"]
    grid_tm["Lower CI"] = pred_tm["mean_ci_lower"]
    grid_tm["Upper CI"] = pred_tm["mean_ci_upper"]

    # Format label khusus untuk bulan
    if time_cat == "bulan":
      grid_tm_display = grid_tm.copy()
      grid_tm_display["bulan"] = grid_tm_display["bulan"].apply(
          lambda x: month_map.get(str(x), str(x))
      )
      x_labels = [month_map.get(str(t), str(t)) for t in time_units]
    else:
      grid_tm_display = grid_tm.copy()
      x_labels = [str(t) for t in time_units]

    col_b1, col_b2 = st.columns([1, 1.5])
    with col_b1:
      st.markdown(f"**CPUE Standar Berdasarkan ({time_cat.title()})**")
      st.dataframe(
          grid_tm_display[
              [time_cat, "CPUE_std (kg/hari)", "Lower CI", "Upper CI"]
          ].round(2),
          use_container_width=True,
          hide_index=True,
      )

    with col_b2:
      fig_mo, ax_mo = plt.subplots(figsize=(7, 3.5))
      x_raw = np.arange(len(time_units))

      if len(time_units) > 2:
        x_smooth = np.linspace(x_raw.min(), x_raw.max(), 300)
        k_deg = min(3, len(time_units) - 1)

        spl_m = make_interp_spline(
            x_raw, grid_tm["CPUE_std (kg/hari)"], k=k_deg
        )
        spl_l = make_interp_spline(x_raw, grid_tm["Lower CI"], k=k_deg)
        spl_u = make_interp_spline(x_raw, grid_tm["Upper CI"], k=k_deg)

        ax_mo.fill_between(
            x_smooth,
            spl_l(x_smooth),
            spl_u(x_smooth),
            color="#E67E22",
            alpha=0.18,
            edgecolor="none",
        )
        ax_mo.plot(x_smooth, spl_m(x_smooth), color="#E67E22", linewidth=1.5)
        ax_mo.scatter(
            x_raw,
            grid_tm["CPUE_std (kg/hari)"],
            color="#D35400",
            s=20,
            zorder=5,
        )
      else:
        ax_mo.fill_between(
            x_raw,
            grid_tm["Lower CI"],
            grid_tm["Upper CI"],
            color="#E67E22",
            alpha=0.18,
            edgecolor="none",
        )
        ax_mo.plot(
            x_raw,
            grid_tm["CPUE_std (kg/hari)"],
            color="#E67E22",
            marker="s",
            markersize=4,
            linewidth=1.5,
        )

      ax_mo.set_xticks(x_raw)
      ax_mo.set_xticklabels(x_labels)
      ax_mo.set_xlabel(time_cat.title())
      ax_mo.set_ylabel("CPUE Standar (kg/hari)")
      ax_mo.set_title(
          f"Pola Standar CPUE Berdasarkan {time_cat.title()}",
          fontweight="bold",
      )
      st.pyplot(fig_mo)
      plt.close(fig_mo)

  # Export Multi-sheet
  output = io.BytesIO()
  with pd.ExcelWriter(output, engine="openpyxl") as writer:
    if "tahun" in valid_cats:
      grid_yr[
          ["tahun", "CPUE_std (kg/hari)", "Lower CI", "Upper CI"]
      ].to_excel(writer, sheet_name="CPUE_Tahunan", index=False)
    if time_cat:
      grid_tm_display[
          [time_cat, "CPUE_std (kg/hari)", "Lower CI", "Upper CI"]
      ].to_excel(writer, sheet_name=f"CPUE_{time_cat}", index=False)

  st.download_button(
      label="📥 Download Hasil Standarisasi CPUE (Excel)",
      data=output.getvalue(),
      file_name="Hasil_Standarisasi_CPUE_YFT.xlsx",
      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  )

# =========================================================
# 6. FOOTER APLIKASI
# =========================================================
render_footer()