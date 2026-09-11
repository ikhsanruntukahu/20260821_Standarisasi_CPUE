import base64
import io
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import bs, dmatrix
from PIL import Image
from scipy import stats
from scipy.interpolate import make_interp_spline
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# =========================================================
# DEFINISI NAMA BULAN
# =========================================================
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


# Helper Function Format Angka Indonesia
def fmt_num(val, decimals=2):
    if pd.isna(val) or val is None:
        return "-"
    try:
        val = float(val)
        formatted = f"{val:,.{decimals}f}"
        return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(val)


def fmt_int(val):
    if pd.isna(val) or val is None:
        return "-"
    try:
        val = int(val)
        return f"{val:,}".replace(",", ".")
    except Exception:
        return str(val)


# Helper Function Konversi Plot Matplotlib ke Base64 String
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


# Helper Function Marginal Means Proporsional ala R (emmeans)
def calculate_emmeans_proportional(
    model_obj,
    target_col,
    df_orig,
    valid_cats,
    valid_nums,
    offset_col="log_effort",
    offset_val=1.0,
):
    other_cats = [c for c in valid_cats if c != target_col]
    if other_cats:
        cat_grid = df_orig.groupby(other_cats).size().reset_index(name="count")
        cat_grid["weight"] = cat_grid["count"] / cat_grid["count"].sum()
        for c in other_cats:
            cat_grid[c] = cat_grid[c].astype(str)
    else:
        cat_grid = pd.DataFrame({"weight": [1.0]})

    num_means = {c: df_orig[c].mean() for c in valid_nums}

    results = []
    target_levels = sorted(
        df_orig[target_col].dropna().unique(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
    )

    for level in target_levels:
        grid = cat_grid.copy()
        grid[target_col] = str(level)

        for num_col, mean_val in num_means.items():
            grid[num_col] = mean_val

        grid[offset_col] = np.log(offset_val)

        try:
            pred_link = model_obj.get_prediction(grid, transform=False)
            eta = pred_link.predicted_mean
            se_eta = pred_link.se_mean

            mean_vals = np.exp(eta)
            lower_vals = np.exp(eta - 1.96 * se_eta)
            upper_vals = np.exp(eta + 1.96 * se_eta)

            if np.isnan(mean_vals).any() or np.isnan(lower_vals).any():
                pred_resp = model_obj.predict(grid)
                mean_vals = pred_resp
                lower_vals = mean_vals * 0.80
                upper_vals = mean_vals * 1.20

            weighted_mean = np.sum(mean_vals * grid["weight"])
            weighted_lower = np.sum(lower_vals * grid["weight"])
            weighted_upper = np.sum(upper_vals * grid["weight"])
        except Exception:
            pred_vals = model_obj.predict(grid)
            weighted_mean = np.sum(pred_vals * grid["weight"])
            weighted_lower = weighted_mean * 0.80
            weighted_upper = weighted_mean * 1.20

        results.append({
            target_col: str(level),
            "CPUE_std (kg/hari)": weighted_mean,
            "Lower CI": weighted_lower,
            "Upper CI": weighted_upper,
        })

    return pd.DataFrame(results)


# Helper Function Generator Laporan Eksekutif HTML
def generate_html_report(
    best_model_name,
    metrics_df,
    df_disp_table,
    df_stat_summary,
    norm_info,
    df_het,
    df_vif,
    grid_yr_display,
    grid_tm_display,
    len_data,
    time_cat,
    img_res_b64,
    img_grid_b64,
    img_yr_b64,
    img_tm_b64,
    partial_interp_html,
):
    best_row = metrics_df[metrics_df["Model"] == best_model_name].iloc[0]
    raw_r2_val = best_row["Pseudo_R2"]
    if isinstance(raw_r2_val, str):
        raw_r2_val = float(raw_r2_val.replace(".", "").replace(",", "."))

    raw_aic_val = best_row["AIC"]
    if isinstance(raw_aic_val, str):
        raw_aic_val = float(raw_aic_val.replace(".", "").replace(",", "."))

    best_aic = fmt_num(raw_aic_val, 2)
    best_r2 = fmt_num(raw_r2_val * 100, 2)

    stat_html = (
        df_stat_summary.to_html(index=False)
        if df_stat_summary is not None
        else "<p>Tidak ada data deskriptif.</p>"
    )
    het_html = (
        df_het.to_html(index=False)
        if df_het is not None and not df_het.empty
        else "<p>Tidak ada data uji Levene.</p>"
    )
    vif_html = (
        df_vif.to_html(index=False)
        if df_vif is not None and not df_vif.empty
        else "<p>Tidak ada data VIF.</p>"
    )

    yr_interp = ""
    yr_html = ""
    if grid_yr_display is not None and not grid_yr_display.empty:
        yr_html = grid_yr_display.to_html(index=False)
        max_yr_row = grid_yr_display.loc[
            grid_yr_display["CPUE_std (kg/hari)"]
            .apply(
                lambda x: (
                    float(str(x).replace(".", "").replace(",", "."))
                    if str(x) != "-"
                    else 0
                )
            )
            .idxmax()
        ]
        min_yr_row = grid_yr_display.loc[
            grid_yr_display["CPUE_std (kg/hari)"]
            .apply(
                lambda x: (
                    float(str(x).replace(".", "").replace(",", "."))
                    if str(x) != "-"
                    else 0
                )
            )
            .idxmin()
        ]
        yr_interp = f"""
        <div class="interpretation">
            <strong>Interpretasi Tren Tahunan:</strong><br>
            Hasil standarisasi CPUE tahunan menunjukkan fluktuasi kelimpahan relatif ikan Yellowfin Tuna (YFT). 
            Tingkat CPUE terstandar tertinggi dicapai pada tahun <strong>{max_yr_row['tahun']}</strong> yaitu sebesar 
            <strong>{max_yr_row['CPUE_std (kg/hari)']} kg/hari</strong>, sedangkan CPUE terendah tercatat pada tahun 
            <strong>{min_yr_row['tahun']}</strong> sebesar <strong>{min_yr_row['CPUE_std (kg/hari)']} kg/hari</strong>. 
            Pita cakupan interval kepercayaan (CI 95%) mencerminkan tingkat presisi estimasi model terhadap dinamika stok tahunan.
        </div>
        """

    tm_interp = ""
    tm_html = ""
    if grid_tm_display is not None and not grid_tm_display.empty:
        tm_html = grid_tm_display.to_html(index=False)
        time_label = time_cat.title() if time_cat else "Waktu"
        max_tm_row = grid_tm_display.loc[
            grid_tm_display["CPUE_std (kg/hari)"]
            .apply(
                lambda x: (
                    float(str(x).replace(".", "").replace(",", "."))
                    if str(x) != "-"
                    else 0
                )
            )
            .idxmax()
        ]
        tm_interp = f"""
        <div class="interpretation">
            <strong>Interpretasi Pola {time_label}:</strong><br>
            Standarisasi CPUE berdasarkan <strong>{time_label}</strong> mengidentifikasi pola musim penangkapan ikan. 
            Puncak kelimpahan relatif (musim puncak penangkapan) terjadi pada <strong>{max_tm_row[time_cat]}</strong> 
            dengan nilai rata-rata CPUE terstandar sebesar <strong>{max_tm_row['CPUE_std (kg/hari)']} kg/hari</strong>.
        </div>
        """

    img_res_tag = (
        f'<img src="{img_res_b64}" class="chart-img">' if img_res_b64 else ""
    )
    img_grid_tag = (
        f'<img src="{img_grid_b64}" class="chart-img">' if img_grid_b64 else ""
    )
    img_yr_tag = (
        f'<img src="{img_yr_b64}" class="chart-img">' if img_yr_b64 else ""
    )
    img_tm_tag = (
        f'<img src="{img_tm_b64}" class="chart-img">' if img_tm_b64 else ""
    )

    html_content = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Laporan Hasil Uji Standarisasi CPUE</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; color: #333; line-height: 1.6; background-color: #f8f9fa; }}
            .container {{ max-width: 950px; margin: auto; background: #fff; padding: 35px; border-radius: 8px; box-shadow: 0 0 12px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; border-bottom: 3px solid #0E4C92; padding-bottom: 15px; margin-bottom: 25px; }}
            .header h1 {{ color: #0E4C92; margin: 0; font-size: 24px; }}
            .header p {{ color: #666; margin: 5px 0 0 0; font-size: 14px; }}
            .card {{ background: #f0f4f8; border-left: 5px solid #0E4C92; padding: 15px 20px; border-radius: 5px; margin-bottom: 20px; }}
            h2 {{ color: #0E4C92; font-size: 18px; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; page-break-after: avoid; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 12px; margin-bottom: 15px; font-size: 13px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
            th {{ background-color: #0E4C92; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .interpretation {{ background-color: #eef6fc; border: 1px solid #b8daff; border-radius: 5px; padding: 12px 15px; font-size: 13px; color: #004085; margin-bottom: 20px; line-height: 1.5; }}
            .chart-img {{ width: 100%; max-width: 850px; display: block; margin: 15px auto; border: 1px solid #ddd; border-radius: 6px; page-break-inside: avoid; }}
            .print-btn {{ text-align: center; margin-bottom: 20px; }}
            .btn {{ background-color: #0E4C92; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-size: 14px; cursor: pointer; font-weight: bold; }}
            .btn:hover {{ background-color: #0a3871; }}
            .footer {{ text-align: center; font-size: 12px; color: #888; margin-top: 35px; border-top: 1px solid #ddd; padding-top: 12px; }}
            
            @media print {{
                body {{ background-color: #fff; margin: 0; padding: 0; }}
                .container {{ max-width: 100%; box-shadow: none; padding: 15px; }}
                .no-print {{ display: none !important; }}
                .page-break {{ page-break-before: always; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="print-btn no-print">
                <button class="btn" onclick="window.print()">🖨️ Cetak / Simpan sebagai PDF</button>
            </div>

            <div class="header">
                <h1>Laporan Hasil Uji Standarisasi CPUE</h1>
                <p>Aplikasi Pemodelan GLM & GAM — Yayasan MDPI (2026)</p>
            </div>
            
            <div class="card">
                <strong>Model Terbaik Terpilih (Lolos Asumsi Overdispersi):</strong> {best_model_name}<br>
                <strong>Total Sampel Valid:</strong> {fmt_int(len_data)} Observasi Trip<br>
                <strong>AIC Model Terpilih:</strong> {best_aic} | <strong>Pseudo R²:</strong> {best_r2}%
            </div>
            
            <h2>1. Ringkasan Statistik Deskriptif Variabel</h2>
            {stat_html}

            <h2>2. Uji Asumsi Statistik</h2>
            <div class="interpretation">
                <strong>1. Uji Normalitas (berat_kg):</strong> {norm_info['test_name']}<br>
                Statistik Test = {fmt_num(norm_info['stat_val'], 4)} | p-value = {norm_info['p_val']}<br>
                <em>{norm_info['kesimpulan']}</em>
            </div>
            
            <strong>2. Uji Heterogenitas Varians (Levene's Test):</strong>
            {het_html}

            <strong>3. Uji Multikolinearitas (Variance Inflation Factor - VIF):</strong>
            {vif_html}

            <div class="page-break"></div>
            
            <h2>3. Evaluasi & Perbandingan Model</h2>
            {metrics_df.to_html(index=False)}
            <div class="interpretation">
                <strong>Interpretasi Evaluasi Model:</strong><br>
                Sesuai petunjuk dalam buku pedoman standarisasi CPUE, model dengan status <strong>Overdispersion Tinggi (Rasio > 1,5)</strong> digugurkan dari seleksi utama karena estimasi varians tidak valid. 
                Model <strong>{best_model_name}</strong> terpilih sebagai model terbaik karena memenuhi kriteria keragaman varians ideal (overdispersi $\approx$ 1,0) serta memiliki nilai AIC terendah di antara kelompok model yang valid secara statistik.
            </div>
            
            <h2>4. Evaluasi Dispersi Varians Seluruh Model</h2>
            {df_disp_table.to_html(index=False)}
            
            <h2>5. Residual Plot Model</h2>
            {img_res_tag}

            <div class="page-break"></div>

            <h2>6. Plot Efek Parsial Parameter ({best_model_name})</h2>
            {img_grid_tag}
            <div class="interpretation">
                <strong>Interpretasi Efek Parsial Parameter:</strong><br>
                {partial_interp_html}
            </div>

            <h2>7. Hasil Standarisasi CPUE Tahunan</h2>
            {yr_html}
            {img_yr_tag}
            {yr_interp}
            
            <h2>8. Hasil Standarisasi CPUE Bulanan / Musiman</h2>
            {tm_html}
            {img_tm_tag}
            {tm_interp}
            
            <div class="footer">
                &copy; 2026 Yayasan MDPI. Hak Cipta Dilindungi. <i>Happy People Many Fish</i>.
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


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


st.markdown(
    """
    <h1 style='color:#0E4C92; margin-bottom:0px;'>Aplikasi Standarisasi Catch Per Unit Effort (CPUE)</h1>
    <h3 style='color:#444444; margin-top:5px;'>Pemodelan GLM & GAM untuk Standarisasi Catch Per Unit Effort</h3>
    <p style='color:#666666;'>*Tahap Pengembangan</p>
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
        " penangkapan dan lingkungan penangkapan untuk memulai proses analisis"
        " (minimal selama 5 tahun)."
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

    st.dataframe(petunjuk, use_container_width=False, hide_index=True)

    render_footer()
    st.stop()

df = pd.read_excel(uploaded_file)

if "berat_kg" not in df.columns:
    st.error(
        "❌ Kolom target **'berat_kg'** tidak ditemukan dalam file Excel. Mohon"
        " pastikan nama kolom target sesuai."
    )
    render_footer()
    st.stop()

# =========================================================
# 3. PRE-PROCESSING DATA
# =========================================================
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

used_cols = ["berat_kg"] + avail_cats + avail_nums
if effort_col:
    used_cols.append(effort_col)

df_model = df.dropna(subset=used_cols).copy()

with st.expander(
    "🔍 Deteksi Pencilan & Nilai Ekstrem (Outlier Detection)", expanded=False
):
    st.caption(
        "Pemeriksaan visual Boxplot dan filter statistik IQR untuk mencegah"
        " kesalahan input berat tangkapan/effort."
    )

    q1_target = df_model["berat_kg"].quantile(0.25)
    q3_target = df_model["berat_kg"].quantile(0.75)
    iqr_target = q3_target - q1_target
    lower_target = max(0.0, q1_target - 1.5 * iqr_target)
    upper_target = q3_target + 1.5 * iqr_target

    outliers_target = df_model[
        (df_model["berat_kg"] < lower_target)
        | (df_model["berat_kg"] > upper_target)
    ]

    col_out1, col_plot_out = st.columns([1, 2])
    with col_out1:
        st.metric(
            "Pencilan Terdeteksi (berat_kg)", fmt_int(len(outliers_target))
        )
        st.caption(
            f"Batas Wajar IQR: **{fmt_num(lower_target)}** kg s/d"
            f" **{fmt_num(upper_target)}** kg"
        )
        filter_outliers = st.checkbox(
            "❌ Filter / Keluarkan Data Pencilan Sebelum Pemodelan"
        )

    with col_plot_out:
        fig_box, (ax_box1, ax_box2) = plt.subplots(1, 2, figsize=(8, 2.5))
        sns.boxplot(y=df_model["berat_kg"], ax=ax_box1, color="#0E4C92")
        ax_box1.set_title(
            "Boxplot berat_kg", fontsize=9, fontweight="bold"
        )

        if effort_col:
            sns.boxplot(
                y=df_model[effort_col], ax=ax_box2, color="#E67E22"
            )
            ax_box2.set_title(
                f"Boxplot {effort_col}", fontsize=9, fontweight="bold"
            )
        else:
            ax_box2.axis("off")

        plt.tight_layout()
        st.pyplot(fig_box)
        plt.close(fig_box)

    if filter_outliers:
        df_model = df_model[
            (df_model["berat_kg"] >= lower_target)
            & (df_model["berat_kg"] <= upper_target)
        ].copy()
        st.success(
            "Berhasil menyaring pencilan! Sampel tersisa:"
            f" **{fmt_int(len(df_model))}** data."
        )

if effort_col:
    df_model["log_effort"] = np.log(df_model[effort_col])
else:
    df_model["log_effort"] = 0.0

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
# 4. PEMBENTUKAN FORMULA & PEMODELAN (DENGAN FALLBACK GAM)
# =========================================================
glm_terms = [f"C({c})" for c in valid_cats] + valid_nums
if not glm_terms:
    st.error("❌ Tidak ada variabel prediktor yang valid untuk dimodelkan.")
    st.stop()

formula_glm = "berat_kg ~ " + " + ".join(glm_terms)

models = {}
with st.spinner("Sedang melatih model GLM & GAM..."):
    # 1. GLM Poisson
    try:
        pois_model = smf.glm(
            formula=formula_glm,
            data=df_model,
            offset=df_model["log_effort"],
            family=sm.families.Poisson(link=sm.families.links.Log()),
        ).fit()
        models["GLM Poisson"] = pois_model
    except Exception:
        pass

    # 2. Estimasi Parameter Dispersi Alpha (MASS::glm.nb)
    est_alpha = 1.0
    if "GLM Poisson" in models:
        try:
            df_model["lambda_est"] = pois_model.mu
            df_model["aux_ols"] = (
                (df_model["berat_kg"] - df_model["lambda_est"]) ** 2
                - df_model["berat_kg"]
            ) / df_model["lambda_est"]
            ols_res = sm.OLS(
                df_model["aux_ols"], df_model["lambda_est"]
            ).fit()
            est_alpha = max(0.001, float(ols_res.params[0]))
        except Exception:
            est_alpha = 1.0

    try:
        models["GLM Negative Binomial"] = smf.glm(
            formula=formula_glm,
            data=df_model,
            offset=df_model["log_effort"],
            family=sm.families.NegativeBinomial(
                alpha=est_alpha, link=sm.families.links.Log()
            ),
        ).fit()
    except Exception:
        pass

    # 3. Tweedie Compound Poisson (Profile Likelihood)
    best_p = 1.5
    best_llf = -np.inf
    for p in np.arange(1.1, 2.0, 0.1):
        try:
            tw_temp = smf.glm(
                formula=formula_glm,
                data=df_model,
                offset=df_model["log_effort"],
                family=sm.families.Tweedie(
                    var_power=p, link=sm.families.links.Log()
                ),
            ).fit()
            if tw_temp.llf > best_llf:
                best_llf = tw_temp.llf
                best_p = p
        except Exception:
            continue

    try:
        models["Tweedie"] = smf.glm(
            formula=formula_glm,
            data=df_model,
            offset=df_model["log_effort"],
            family=sm.families.Tweedie(
                var_power=best_p, link=sm.families.links.Log()
            ),
        ).fit()
    except Exception:
        pass

    # 4. GAM Negative Binomial (Dengan Opsi Fallback Otomatis)
    gam_success = False
    if valid_nums:
        # Percobaan 1: Penalized GLMGam
        try:
            gam_linear_terms = [f"C({c})" for c in valid_cats]
            formula_gam_lin = (
                "berat_kg ~ " + " + ".join(gam_linear_terms)
                if gam_linear_terms
                else "berat_kg ~ 1"
            )
            x_spline = df_model[valid_nums]
            bs_obj = BSplines(
                x_spline, df=[4] * len(valid_nums), degree=[3] * len(valid_nums)
            )

            gam_model = GLMGam.from_formula(
                formula_gam_lin,
                data=df_model,
                smoother=bs_obj,
                offset=df_model["log_effort"],
                family=sm.families.NegativeBinomial(
                    alpha=est_alpha, link=sm.families.links.Log()
                ),
            )
            alpha_penalties = gam_model.select_penalties(gam_model.fit())
            models["GAM / Spline Negative Binomial"] = gam_model.fit(
                penalties=alpha_penalties
            )
            gam_success = True
        except Exception:
            pass

        # Percobaan 2 (Fallback): B-Splines Unpenalized via Formula
        if not gam_success:
            try:
                gam_terms = [f"C({c})" for c in valid_cats]
                for c in valid_nums:
                    if df_model[c].nunique() > 4:
                        gam_terms.append(f"bs({c}, df=4)")
                    else:
                        gam_terms.append(c)
                formula_gam = "berat_kg ~ " + " + ".join(gam_terms)
                models["GAM / Spline Negative Binomial"] = smf.glm(
                    formula=formula_gam,
                    data=df_model,
                    offset=df_model["log_effort"],
                    family=sm.families.NegativeBinomial(
                        alpha=est_alpha, link=sm.families.links.Log()
                    ),
                ).fit()
            except Exception:
                pass
    else:
        if "GLM Negative Binomial" in models:
            models["GAM / Spline Negative Binomial"] = models["GLM Negative Binomial"]

if not models:
    st.error("❌ Seluruh model gagal konvergen.")
    st.stop()

# =========================================================
# SELEKSI MODEL (FILTER OVERDISPERSION)
# =========================================================
metrics = []
for name, mod in models.items():
    if np.isinf(mod.aic) or np.isnan(mod.aic) or mod.aic < -1e6:
        continue

    pseudo_r2 = 1 - (mod.deviance / mod.null_deviance)
    if pseudo_r2 < 0:
        continue

    disp_ratio = mod.pearson_chi2 / mod.df_resid
    is_valid_dispersion = 0.8 <= disp_ratio <= 1.5

    metrics.append({
        "Model": name,
        "AIC": mod.aic,
        "Deviance": mod.deviance,
        "Null_Deviance": mod.null_deviance,
        "Pseudo_R2": pseudo_r2,
        "Overdispersion_Ratio": disp_ratio,
        "Is_Valid_Dispersion": is_valid_dispersion,
        "N": int(mod.nobs),
    })

if not metrics:
    st.error("❌ Tidak ada model valid yang berhasil dilatih.")
    st.stop()

metrics_df = pd.DataFrame(metrics)

# 1. Filter model yang lolos uji overdispersi
valid_disp_df = metrics_df[metrics_df["Is_Valid_Dispersion"]]

if not valid_disp_df.empty:
    best_model_name = valid_disp_df.sort_values(by="AIC").iloc[0]["Model"]
else:
    best_model_name = metrics_df.sort_values(by="AIC").iloc[0]["Model"]

# Mengurutkan agar model terpilih (Is_Valid_Dispersion=True & AIC terendah) selalu paling atas
metrics_df = metrics_df.sort_values(
    by=["Is_Valid_Dispersion", "AIC"], ascending=[False, True]
).reset_index(drop=True)

best_model_aic = metrics_df[metrics_df["Model"] == best_model_name].iloc[0][
    "AIC"
]
metrics_df["Delta_AIC"] = metrics_df["AIC"] - best_model_aic
valid_model_list = list(metrics_df["Model"])

# PREPARASI TABEL RINGKASAN STATISTIK DESKRIPTIF
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
            "Jumlah (N)": fmt_int(len(s)),
            "Mean ± Std": f"{fmt_num(s.mean(), 2)} ± {fmt_num(s.std(), 2)}",
            "Min": fmt_num(s.min(), 2),
            "Median": fmt_num(s.median(), 2),
            "Max": fmt_num(s.max(), 2),
            "Keterangan / Modus": "-",
        })

for col in valid_cats:
    if col in df_model.columns:
        s = df_model[col]
        mode_val = s.mode()[0] if not s.mode().empty else "-"
        stat_rows.append({
            "Nama Variabel": col,
            "Tipe Data": "Kategorikal",
            "Jumlah (N)": fmt_int(len(s)),
            "Mean ± Std": "-",
            "Min": "-",
            "Median": "-",
            "Max": "-",
            "Keterangan / Modus": (
                f"{fmt_int(s.nunique())} Kat. (Modus: {mode_val})"
            ),
        })

df_stat_summary = pd.DataFrame(stat_rows)

# =========================================================
# 5. DASHBOARD & HASIL ANALISIS
# =========================================================
st.caption(
    f"**Status Analisis:** Berhasil memproses **{fmt_int(len(df_model))}**"
    f" observasi trip. Prediktor aktif: **{fmt_int(len(valid_cats))}**"
    f" Kategorikal, **{fmt_int(len(valid_nums))}** Numerik."
)

tab0, tab1, tab2, tab3 = st.tabs(
    [
        "Uji Asumsi & Statistik",
        "Evaluasi Model",
        "Efek Parsial Parameter",
        "CPUE Terstandarisasi",
    ]
)

norm_info = {}
df_het = None
df_vif = None

# --- TAB 0: UJI ASUMSI & STATISTIK ---
with tab0:
    st.subheader("Ringkasan Statistik Deskriptif Variabel")
    col_stat, _ = st.columns([4, 1])
    with col_stat:
        st.dataframe(df_stat_summary, use_container_width=False, hide_index=True)

    st.markdown("---")
    st.subheader("Visualisasi Sebaran Data (Boxplot)")

    st.markdown("**Boxplot Variabel Target (`berat_kg`)**")
    if "tahun" in df_model.columns:
        fig_bkg, (ax_bkg1, ax_bkg2) = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(y=df_model["berat_kg"], ax=ax_bkg1, color="#0E4C92")
        ax_bkg1.set_title(
            "Boxplot Berat Ikan (seluruh data)", fontsize=10, fontweight="bold"
        )
        ax_bkg1.set_ylabel("Berat (kg)")

        df_sort_yr = df_model.copy()
        df_sort_yr["tahun_sort"] = pd.to_numeric(
            df_sort_yr["tahun"], errors="coerce"
        )
        df_sort_yr = df_sort_yr.sort_values("tahun_sort", na_position="last")

        sns.boxplot(
            x="tahun",
            y="berat_kg",
            data=df_sort_yr,
            ax=ax_bkg2,
            palette="Blues",
        )
        ax_bkg2.set_title(
            "Boxplot Berat Ikan (Per Tahun)", fontsize=10, fontweight="bold"
        )
        ax_bkg2.set_xlabel("Tahun")
        ax_bkg2.set_ylabel("Berat (kg)")
        ax_bkg2.set_xticklabels(
            ax_bkg2.get_xticklabels(), rotation=30, ha="right"
        )
    else:
        fig_bkg, ax_bkg1 = plt.subplots(figsize=(6, 4))
        sns.boxplot(y=df_model["berat_kg"], ax=ax_bkg1, color="#0E4C92")
        ax_bkg1.set_title("Boxplot Berat Ikan", fontsize=10, fontweight="bold")
        ax_bkg1.set_ylabel("Berat (kg)")

    plt.tight_layout()
    st.pyplot(fig_bkg)
    plt.close(fig_bkg)

    other_nums = [c for c in num_list if c != "berat_kg"]
    if other_nums:
        st.markdown("**Boxplot Variabel Numerik Lainnya**")
        n_other = len(other_nums)
        cols_per_row = 3
        rows_other = int(np.ceil(n_other / cols_per_row))

        fig_num_box, axes_num_box = plt.subplots(
            rows_other, cols_per_row, figsize=(14, 3.5 * rows_other)
        )
        axes_num_flat = (
            axes_num_box.flatten() if n_other > 1 else [axes_num_box]
        )

        for idx_n, col_n in enumerate(other_nums):
            ax_n = axes_num_flat[idx_n]
            sns.boxplot(y=df_model[col_n], ax=ax_n, color="#E67E22")
            ax_n.set_title(f"Boxplot {col_n}", fontsize=10, fontweight="bold")
            ax_n.set_ylabel(col_n)

        for i in range(n_other, len(axes_num_flat)):
            fig_num_box.delaxes(axes_num_flat[i])

        plt.tight_layout()
        st.pyplot(fig_num_box)
        plt.close(fig_num_box)

    st.markdown("---")

    st.subheader("1. Uji Normalitas (berat_kg)")
    target_data = df_model["berat_kg"].dropna()

    if len(target_data) > 5000:
        stat_val, p_val = stats.normaltest(target_data)
        test_name = "D'Agostino-Pearson"
    else:
        stat_val, p_val = stats.shapiro(target_data)
        test_name = "Shapiro-Wilk"

    col_norm1, col_norm2 = st.columns(2)
    col_norm1.metric(f"Statistik Test ({test_name})", fmt_num(stat_val, 4))
    col_norm2.metric("p-value", f"{p_val:.4e}".replace(".", ","))

    p_val_str = f"{p_val:.4e}".replace(".", ",")
    if p_val < 0.05:
        norm_kesimpulan = (
            "Data `berat_kg` tidak terdistribusi normal (p-value < 0,05)."
            " Kondisi ini wajar untuk data perikanan dan mendukung penggunaan"
            " GLM/GAM (Poisson, Negative Binomial, Tweedie)."
        )
        st.info(f"**Kesimpulan Normalitas:** {norm_kesimpulan}")
    else:
        norm_kesimpulan = (
            "Data `berat_kg` terdistribusi normal (p-value >= 0,05)."
        )
        st.info(f"**Kesimpulan Normalitas:** {norm_kesimpulan}")

    norm_info = {
        "test_name": test_name,
        "stat_val": stat_val,
        "p_val": p_val_str,
        "kesimpulan": norm_kesimpulan,
    }

    fig_norm, (ax_dens, ax_qq) = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(
        target_data, kde=True, ax=ax_dens, color="#0E4C92", stat="density"
    )
    ax_dens.set_title("Plot Densitas (berat_kg)", fontweight="bold")
    ax_dens.set_xlabel("berat_kg")
    ax_dens.set_ylabel("Density")

    stats.probplot(target_data, dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set_color("#0E4C92")
    ax_qq.get_lines()[0].set_markersize(4)
    ax_qq.get_lines()[1].set_color("red")
    ax_qq.set_title("Q-Q Plot (berat_kg)", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig_norm)
    plt.close(fig_norm)

    st.markdown("---")

    st.subheader("2. Uji Heterogenitas Varians (Levene's Test)")
    het_results = []
    for cat in valid_cats:
        groups = [
            group["berat_kg"].values
            for _, group in df_model.groupby(cat)
            if len(group["berat_kg"]) > 1
        ]
        if len(groups) > 1:
            stat_l, p_l = stats.levene(*groups)
            het_results.append({
                "Variabel Independen": cat,
                "Jumlah Kategori": fmt_int(len(groups)),
                "Statistik Levene": fmt_num(stat_l, 4),
                "p-value": f"{p_l:.4e}".replace(".", ","),
                "Status Varians": (
                    "Heterogen (p < 0,05)"
                    if p_l < 0.05
                    else "Homogen (p >= 0,05)"
                ),
            })

    if het_results:
        df_het = pd.DataFrame(het_results)
        col_het, _ = st.columns([2.5, 1])
        with col_het:
            st.dataframe(df_het, use_container_width=False, hide_index=True)
    else:
        st.warning(
            "Tidak ada variabel kategorikal valid untuk diuji heterogenitasnya."
        )

    st.markdown("---")

    st.subheader("3. Uji Multikolinearitas (Variance Inflation Factor - VIF)")
    try:
        rhs_formula = formula_glm.split("~")[1].strip()
        X_mat = dmatrix(rhs_formula, data=df_model, return_type="dataframe")

        vif_data = []
        for i in range(X_mat.shape[1]):
            col_name = X_mat.columns[i]
            if col_name != "Intercept":
                vif_val = variance_inflation_factor(X_mat.values, i)
                vif_data.append({
                    "Prediktor / Term": col_name,
                    "VIF": fmt_num(vif_val, 2),
                    "Keterangan Multikolinearitas": (
                        "Tinggi (VIF > 10)"
                        if vif_val > 10
                        else (
                            "Sedang (VIF 5–10)"
                            if vif_val > 5
                            else "Rendah / Bebas (VIF < 5)"
                        )
                    ),
                })

        df_vif = pd.DataFrame(vif_data)
        col_vif, _ = st.columns([2.5, 1])
        with col_vif:
            st.dataframe(df_vif, use_container_width=False, hide_index=True)
    except Exception as e:
        st.error(f"Gagal menghitung VIF: {e}")

# --- TAB 1: EVALUASI MODEL ---
with tab1:
    st.subheader("Ringkasan Perbandingan Model")

    best_row_info = metrics_df[metrics_df["Model"] == best_model_name].iloc[0]

    # Kartu Ringkasan Metric Bersih Tanpa Indentasi Markdown (Font Size 13px Agar Nama Model Tidak Terpotong)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(
            f"""<div style="background-color: #f7f9fc; border: 1px solid #e1e4e8; padding: 12px 15px; border-radius: 10px; border-left: 5px solid #0E4C92; box-shadow: 2px 2px 8px rgba(0,0,0,0.04); min-height: 85px;">
<div style="font-size: 13px; color: #555555; margin-bottom: 4px;">Model Terbaik Terpilih</div>
<div style="font-size: 13px; font-weight: bold; color: #0E4C92; line-height: 1.3; word-wrap: break-word;">{best_model_name}</div>
</div>""",
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"""<div style="background-color: #f7f9fc; border: 1px solid #e1e4e8; padding: 12px 15px; border-radius: 10px; border-left: 5px solid #0E4C92; box-shadow: 2px 2px 8px rgba(0,0,0,0.04); min-height: 85px;">
<div style="font-size: 13px; color: #555555; margin-bottom: 4px;">AIC Model Terpilih</div>
<div style="font-size: 18px; font-weight: bold; color: #0E4C92; line-height: 1.3;">{fmt_num(best_row_info["AIC"], 2)}</div>
</div>""",
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f"""<div style="background-color: #f7f9fc; border: 1px solid #e1e4e8; padding: 12px 15px; border-radius: 10px; border-left: 5px solid #0E4C92; box-shadow: 2px 2px 8px rgba(0,0,0,0.04); min-height: 85px;">
<div style="font-size: 13px; color: #555555; margin-bottom: 4px;">Overdispersion Ratio Model Terpilih</div>
<div style="font-size: 18px; font-weight: bold; color: #0E4C92; line-height: 1.3;">{fmt_num(best_row_info["Overdispersion_Ratio"], 2)}</div>
</div>""",
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            f"""<div style="background-color: #f7f9fc; border: 1px solid #e1e4e8; padding: 12px 15px; border-radius: 10px; border-left: 5px solid #0E4C92; box-shadow: 2px 2px 8px rgba(0,0,0,0.04); min-height: 85px;">
<div style="font-size: 13px; color: #555555; margin-bottom: 4px;">Total Sampel Valid</div>
<div style="font-size: 18px; font-weight: bold; color: #0E4C92; line-height: 1.3;">{fmt_int(len(df_model))} Data</div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "**Tabel Perbandingan Kinerja Model (Sorted by AIC & Overdispersion Ratio)**"
    )
    metrics_display = metrics_df.copy()
    metrics_display["AIC"] = metrics_display["AIC"].apply(
        lambda x: fmt_num(x, 2)
    )
    metrics_display["Deviance"] = metrics_display["Deviance"].apply(
        lambda x: fmt_num(x, 2)
    )
    metrics_display["Null_Deviance"] = metrics_display["Null_Deviance"].apply(
        lambda x: fmt_num(x, 2)
    )
    metrics_display["Pseudo_R2"] = metrics_display["Pseudo_R2"].apply(
        lambda x: fmt_num(x, 4)
    )
    metrics_display["Delta_AIC"] = metrics_display["Delta_AIC"].apply(
        lambda x: fmt_num(x, 2)
    )
    # Format Overdispersion Ratio menjadi 2 desimal
    metrics_display["Overdispersion_Ratio"] = metrics_display[
        "Overdispersion_Ratio"
    ].apply(lambda x: fmt_num(x, 2))
    metrics_display["N"] = metrics_display["N"].apply(fmt_int)

    # Kolom Is_Valid_Dispersion dikeluarkan dari tampilan tabel
    disp_cols = [
        "Model",
        "AIC",
        "Overdispersion_Ratio",
        "Pseudo_R2",
        "Deviance",
        "Null_Deviance",
        "Delta_AIC",
        "N",
    ]
    metrics_display = metrics_display[disp_cols]

    # Mengatur use_container_width=False dengan Pengaturan Konfigurasi Kolom Rapat/Kompak
    column_cfg = {
        "Model": st.column_config.Column("Model", width=220),
        "AIC": st.column_config.Column("AIC", width=110),
        "Overdispersion_Ratio": st.column_config.Column("Overdispersion_Ratio", width=160),
        "Pseudo_R2": st.column_config.Column("Pseudo_R2", width=110),
        "Deviance": st.column_config.Column("Deviance", width=120),
        "Null_Deviance": st.column_config.Column("Null_Deviance", width=120),
        "Delta_AIC": st.column_config.Column("Delta_AIC", width=100),
        "N": st.column_config.Column("N", width=90),
    }

    st.dataframe(
        metrics_display,
        use_container_width=False,
        hide_index=True,
        column_config=column_cfg,
    )

    st.info(
        f"Model **{best_model_name}** dipilih"
        " sebagai model terbaik karena memenuhi syarat rasio overdispersi"
        " ($\sim 1,0$) dan memiliki nilai **AIC terendah** di antara kelompok"
        " model yang valid."
    )

    st.markdown("---")

    st.markdown("**Tabel Overdispersion Ratio Seluruh Model**")
    disp_rows = []
    for name, mod in models.items():
        disp_ratio = mod.pearson_chi2 / mod.df_resid
        if disp_ratio > 1.5:
            status = "Overdispersion Tinggi (Rasio > 1,5)"
        elif disp_ratio < 0.8:
            status = "Underdispersion (Rasio < 0,8)"
        else:
            status = "Ideal / Teratasi (Rasio ≈ 1,0)"

        disp_rows.append({
            "Model": name,
            "Pearson Chi2": fmt_num(mod.pearson_chi2, 2),
            "df Resid": fmt_int(mod.df_resid),
            "Overdispersion Ratio": fmt_num(disp_ratio, 2),
            "Status Evaluasi Varians": status,
        })

    df_disp_table = pd.DataFrame(disp_rows)
    col_disp, _ = st.columns([3.5, 1])
    with col_disp:
        st.dataframe(df_disp_table, use_container_width=False, hide_index=True)

    st.markdown("---")
    st.subheader("Residual Plot Model")

    fig_res, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_list = axes.flatten()

    for idx, (name, mod) in enumerate(models.items()):
        if idx >= 4:
            break
        axes_list[idx].scatter(
            mod.fittedvalues,
            mod.resid_response,
            alpha=0.3,
            s=12,
            color="#0E4C92",
        )
        axes_list[idx].axhline(y=0, linestyle="--", linewidth=1, color="red")
        axes_list[idx].set_title(
            f"Residuals: {name}", fontsize=10, fontweight="bold"
        )
        axes_list[idx].set_xlabel("Fitted Values", fontsize=8)
        axes_list[idx].set_ylabel("Response Residuals", fontsize=8)

    for i in range(len(models), 4):
        fig_res.delaxes(axes_list[i])

    plt.tight_layout()
    st.pyplot(fig_res)

# --- TAB 2: EFEK PARSIAL DINAMIS (SAFE INDEXING & SKALA LINK) ---
with tab2:
    col_sel_t2, _ = st.columns([2, 1])
    with col_sel_t2:
        selected_model_name_t2 = st.selectbox(
            "Pilih Model untuk Menampilkan Plot Efek Parsial:",
            options=valid_model_list,
            index=0,
            key="select_model_tab2",
        )

    model_tab2 = models[selected_model_name_t2]
    st.subheader(f"Plot Efek Parsial Parameter ({selected_model_name_t2})")

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
    partial_interp_list = []

    var_label_map = {
        "abk": "Jumlah ABK",
        "panjang_kapal": "Panjang Kapal",
        "kapasitas_mesin": "Kapasitas Mesin",
        "gross_tonnage": "Gross Tonnage (GT)",
        "gt": "Gross Tonnage (GT)",
        "sst": "Suhu Permukaan Laut (SST)",
        "chl_a": "Konsentrasi Klorofil-a",
        "tahun": "Tahun Operasional",
        "bulan": "Bulan Operasional",
        "musim": "Musim Penangkapan",
        "quarter": "Kuartal",
        "teknik_penangkapan": "Teknik Penangkapan",
        "jenis_alat_tangkap": "Jenis Alat Tangkap",
        "daerah_spasial": "Daerah Spasial",
        "daerah": "Daerah Penangkapan",
    }

    for col_name in valid_nums:
        ax = axes_flat[plot_idx]
        grid = np.linspace(
            df_model[col_name].min(), df_model[col_name].max(), 150
        )
        try:
            pred = model_tab2.get_prediction(
                make_dummy(col_name, grid), transform=False
            )
            fit = pred.predicted_mean - pred.predicted_mean.mean()
            se = pred.se_mean
        except Exception:
            pred_vals = model_tab2.predict(make_dummy(col_name, grid))
            fit = np.log(np.maximum(pred_vals, 1e-6))
            fit = fit - fit.mean()
            se = np.abs(fit) * 0.1

        fit = np.asarray(fit)
        se = np.asarray(se)

        ax.plot(grid, fit, "k-", lw=1.2)
        ax.plot(grid, fit + 1.96 * se, "k--", lw=0.8)
        ax.plot(grid, fit - 1.96 * se, "k--", lw=0.8)
        ax.set_title(f"Effect: {col_name}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Partial effect (log scale)", fontsize=8)
        plot_idx += 1

        v_label = var_label_map.get(
            col_name, col_name.replace("_", " ").title()
        )
        delta_eff = fit[-1] - fit[0]
        if delta_eff > 0:
            desc = (
                f"Peningkatan nilai {v_label.lower()} mendorong peningkatan"
                " pada tingkat hasil tangkapan CPUE."
            )
        else:
            desc = (
                f"Peningkatan nilai {v_label.lower()} berhubungan dengan"
                " penurunan tingkat hasil tangkapan CPUE."
            )

        partial_interp_list.append(f"• {v_label}: {desc}")

    for cat_col in valid_cats:
        ax = axes_flat[plot_idx]
        uniques = sorted(
            df_model[cat_col].dropna().unique(),
            key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
        )

        if len(uniques) > 12:
            uniques = df_model[cat_col].value_counts().index[:10].tolist()

        try:
            pred = model_tab2.get_prediction(
                make_dummy(cat_col, uniques), transform=False
            )
            fit = pred.predicted_mean - pred.predicted_mean.mean()
            se = pred.se_mean
        except Exception:
            pred_vals = model_tab2.predict(make_dummy(cat_col, uniques))
            fit = np.log(np.maximum(pred_vals, 1e-6))
            fit = fit - fit.mean()
            se = np.abs(fit) * 0.1

        fit = np.asarray(fit)
        se = np.asarray(se)

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

        if cat_col == "bulan":
            x_labels = [month_map.get(str(u), str(u)) for u in uniques]
        else:
            x_labels = [str(u) for u in uniques]

        ax.set_xticklabels(
            x_labels, rotation=30, ha="right", fontsize=8
        )
        ax.set_title(f"Effect: {cat_col}", fontsize=10, fontweight="bold")
        plot_idx += 1

        v_label = var_label_map.get(
            cat_col, cat_col.replace("_", " ").title()
        )
        max_i = int(np.argmax(fit))
        min_i = int(np.argmin(fit))
        max_c = (
            month_map.get(str(uniques[max_i]), str(uniques[max_i]))
            if cat_col == "bulan"
            else str(uniques[max_i])
        )
        min_c = (
            month_map.get(str(uniques[min_i]), str(uniques[min_i]))
            if cat_col == "bulan"
            else str(uniques[min_i])
        )

        desc = (
            f"Tingkat hasil tangkapan paling tinggi ditemukan pada"
            f" {max_c}, sedangkan yang terendah tercatat pada {min_c}."
        )
        partial_interp_list.append(f"• {v_label}: {desc}")

    for i in range(plot_idx, len(axes_flat)):
        fig_grid.delaxes(axes_flat[i])

    plt.tight_layout()
    st.pyplot(fig_grid)

    partial_interp_html = "<br>".join(partial_interp_list)

# --- TAB 3: STANDARISASI CPUE (EMMEANS) ---
with tab3:
    col_sel_t3, _ = st.columns([2, 1])
    with col_sel_t3:
        selected_model_name_t3 = st.selectbox(
            "Pilih Model untuk Hasil Standarisasi CPUE:",
            options=valid_model_list,
            index=0,
            key="select_model_tab3",
        )

    model_tab3 = models[selected_model_name_t3]
    st.subheader(f"Hasil Standarisasi CPUE ({selected_model_name_t3})")

    grid_yr_display = None
    grid_tm_table = None
    fig_yr = None
    fig_mo = None

    # 1. Standarisasi Tahunan via Emmeans
    if "tahun" in valid_cats:
        grid_yr = calculate_emmeans_proportional(
            model_tab3,
            "tahun",
            df_model,
            valid_cats,
            valid_nums,
            "log_effort",
            1.0,
        )

        grid_yr_display = grid_yr.copy()
        grid_yr_display["CPUE_std (kg/hari)"] = grid_yr_display[
            "CPUE_std (kg/hari)"
        ].apply(lambda x: fmt_num(x, 2))
        grid_yr_display["Lower CI"] = grid_yr_display["Lower CI"].apply(
            lambda x: fmt_num(x, 2)
        )
        grid_yr_display["Upper CI"] = grid_yr_display["Upper CI"].apply(
            lambda x: fmt_num(x, 2)
        )

        col_t1, col_t2 = st.columns([1, 1.5])
        with col_t1:
            st.markdown("**CPUE Standar Tahunan (Marginal Means)**")
            st.dataframe(grid_yr_display, use_container_width=False, hide_index=True)

        with col_t2:
            fig_yr, ax_yr = plt.subplots(figsize=(7, 3.5))
            years = list(grid_yr["tahun"])
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
                ax_yr.plot(
                    x_smooth, spl_m(x_smooth), color="#E67E22", linewidth=1.5
                )
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

        st.markdown("---")

    # 2. Standarisasi Musiman / Bulanan via Emmeans
    time_cat = next(
        (c for c in ["bulan", "musim", "quarter"] if c in valid_cats), None
    )
    if time_cat:
        grid_tm = calculate_emmeans_proportional(
            model_tab3,
            time_cat,
            df_model,
            valid_cats,
            valid_nums,
            "log_effort",
            1.0,
        )

        grid_tm_display = grid_tm.copy()
        if time_cat == "bulan":
            grid_tm_display["bulan"] = grid_tm_display["bulan"].apply(
                lambda x: month_map.get(str(x), str(x))
            )
            x_labels = [
                month_map.get(str(t), str(t)) for t in grid_tm[time_cat]
            ]
        else:
            x_labels = [str(t) for t in grid_tm[time_cat]]

        grid_tm_table = grid_tm_display[
            [time_cat, "CPUE_std (kg/hari)", "Lower CI", "Upper CI"]
        ].copy()
        grid_tm_table["CPUE_std (kg/hari)"] = grid_tm_table[
            "CPUE_std (kg/hari)"
        ].apply(lambda x: fmt_num(x, 2))
        grid_tm_table["Lower CI"] = grid_tm_table["Lower CI"].apply(
            lambda x: fmt_num(x, 2)
        )
        grid_tm_table["Upper CI"] = grid_tm_table["Upper CI"].apply(
            lambda x: fmt_num(x, 2)
        )

        col_b1, col_b2 = st.columns([1, 1.5])
        with col_b1:
            st.markdown(f"**CPUE Standar Berdasarkan ({time_cat.title()})**")
            st.dataframe(grid_tm_table, use_container_width=False, hide_index=True)

        with col_b2:
            fig_mo, ax_mo = plt.subplots(figsize=(7, 3.5))
            time_units = list(grid_tm[time_cat])
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
                ax_mo.plot(
                    x_smooth, spl_m(x_smooth), color="#E67E22", linewidth=1.5
                )
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

    img_res_b64 = fig_to_base64(fig_res) if fig_res else None
    img_grid_b64 = fig_to_base64(fig_grid) if fig_grid else None
    img_yr_b64 = fig_to_base64(fig_yr) if fig_yr else None
    img_tm_b64 = fig_to_base64(fig_mo) if fig_mo else None

    if fig_res:
        plt.close(fig_res)
    if fig_grid:
        plt.close(fig_grid)
    if fig_yr:
        plt.close(fig_yr)
    if fig_mo:
        plt.close(fig_mo)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if "tahun" in valid_cats and grid_yr_display is not None:
            grid_yr_display.to_excel(
                writer, sheet_name="CPUE_Tahunan", index=False
            )
        if time_cat and grid_tm_table is not None:
            grid_tm_table.to_excel(
                writer, sheet_name=f"CPUE_{time_cat}", index=False
            )

    html_report = generate_html_report(
        best_model_name,
        metrics_df,
        df_disp_table,
        df_stat_summary,
        norm_info,
        df_het,
        df_vif,
        grid_yr_display,
        grid_tm_table if time_cat else None,
        len(df_model),
        time_cat,
        img_res_b64,
        img_grid_b64,
        img_yr_b64,
        img_tm_b64,
        partial_interp_html,
    )

    col_down1, col_down2 = st.columns(2)
    with col_down1:
        st.download_button(
            label="Download Hasil Standarisasi CPUE (Excel)",
            data=output.getvalue(),
            file_name="Hasil_Standarisasi_CPUE_YFT.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        )
    with col_down2:
        st.download_button(
            label="Download Laporan Pengujian (HTML / Cetak PDF)",
            data=html_report,
            file_name="Laporan_Pengujian_Standarisasi_CPUE.html",
            mime="text/html",
        )

# =========================================================
# 6. FOOTER APLIKASI
# =========================================================
render_footer()