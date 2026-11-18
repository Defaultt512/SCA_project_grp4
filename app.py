import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import milp, LinearConstraint, Bounds
import streamlit as st

st.set_page_config(
    page_title="AgriFlow Control Tower | Cold-Chain Network Planner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# ENTERPRISE LIGHT-MODE CSS (CLEAN SAAS CONTROL TOWER AESTHETIC)
# -------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .topbar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .topbar-sub {
        font-size: 0.85rem;
        color: #64748B;
        margin: 0.15rem 0 0 0;
    }
    .corridor-pill {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        white-space: nowrap;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
    }
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .alert-banner {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #D97706;
        color: #92400E;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        font-size: 0.9rem;
    }
    .success-banner {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 4px solid #059669;
        color: #065F46;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        font-size: 0.9rem;
    }
    .sidebar-brand {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        padding-bottom: 0.25rem;
    }
    .sidebar-caption {
        font-size: 0.78rem;
        color: #64748B;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# DATA & NETWORK MASTER METADATA (WITH REAL LAT/LON FOR CORRIDOR MAP)
# -------------------------------------------------------------------------
@st.cache_data
def load_sample_data():
    df_ts = pd.read_csv("agri_coldchain_104w_timeseries.csv")
    df_rt = pd.read_csv("agri_coldchain_network_routes.csv")
    return df_ts, df_rt

df_sample_ts, df_sample_rt = load_sample_data()

if "active_ts" not in st.session_state:
    st.session_state["active_ts"] = df_sample_ts.copy()
if "data_source_label" not in st.session_state:
    st.session_state["data_source_label"] = "104-Week Corridor Dataset (Nov 2024 – Oct 2026)"

CLUSTERS_META = [
    ("FC01", "Nashik (Pimpalgaon)", "Maharashtra", "FC01_Nashik_MT", 20.16, 73.98),
    ("FC02", "Narayangaon (Junner)", "Maharashtra", "FC02_Narayangaon_MT", 19.12, 73.97),
    ("FC03", "Satara (Wai-Karad)", "Maharashtra", "FC03_Satara_MT", 17.69, 74.00),
    ("FC04", "Sangli (Miraj-Tasgaon)", "Maharashtra", "FC04_Sangli_MT", 16.85, 74.58),
    ("FC05", "Kolhapur (Shirol)", "Maharashtra", "FC05_Kolhapur_MT", 16.70, 74.24),
    ("FC06", "Belagavi (Gokak)", "Karnataka", "FC06_Belagavi_MT", 16.17, 74.82),
    ("FC07", "Dharwad (Hubballi)", "Karnataka", "FC07_Dharwad_MT", 15.46, 75.01),
    ("FC08", "Kolar (Mulbagal)", "Karnataka", "FC08_Kolar_MT", 13.13, 78.13),
]

PACKHOUSES_META = [
    {"id": "PH01", "name": "Ozarkhed Packhouse (Nashik)", "cap": 850.0, "capex_l": 68.0, "fixed_wk": 145000.0, "var_mt": 620.0, "loss_red": 0.68, "lat": 20.22, "lon": 73.88},
    {"id": "PH02", "name": "Chakan Agro Hub (Pune N)", "cap": 750.0, "capex_l": 64.0, "fixed_wk": 135000.0, "var_mt": 590.0, "loss_red": 0.66, "lat": 18.76, "lon": 73.86},
    {"id": "PH03", "name": "Karad Integrated Packhouse", "cap": 650.0, "capex_l": 54.0, "fixed_wk": 115000.0, "var_mt": 560.0, "loss_red": 0.65, "lat": 17.29, "lon": 74.18},
    {"id": "PH04", "name": "Kagal Cold Chain Park", "cap": 600.0, "capex_l": 50.0, "fixed_wk": 108000.0, "var_mt": 550.0, "loss_red": 0.64, "lat": 16.58, "lon": 74.31},
    {"id": "PH05", "name": "Belagavi-Kittur Hub", "cap": 700.0, "capex_l": 56.0, "fixed_wk": 118000.0, "var_mt": 540.0, "loss_red": 0.67, "lat": 15.60, "lon": 74.79},
    {"id": "PH06", "name": "Hoskote-Kolar Packhouse", "cap": 800.0, "capex_l": 66.0, "fixed_wk": 140000.0, "var_mt": 600.0, "loss_red": 0.68, "lat": 13.07, "lon": 77.79},
]

HUBS_META = [
    {"id": "DH01", "name": "Mumbai Metro DC (Vashi)", "col": "DH01_Mumbai_Demand_MT", "spot_mt": 31500.0, "lat": 19.07, "lon": 72.99},
    {"id": "DH02", "name": "Pune Metro DC (Marketyard)", "col": "DH02_Pune_Demand_MT", "spot_mt": 29500.0, "lat": 18.49, "lon": 73.87},
    {"id": "DH03", "name": "Goa Coastal Hub (Panaji)", "col": "DH03_Goa_Demand_MT", "spot_mt": 33800.0, "lat": 15.49, "lon": 73.82},
    {"id": "DH04", "name": "Bengaluru Metro DC", "col": "DH04_Bengaluru_Demand_MT", "spot_mt": 30500.0, "lat": 13.02, "lon": 77.55},
]

# -------------------------------------------------------------------------
# SIDEBAR NAVIGATION (CLEAN, NO CLO TAGS)
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-brand'>AgriFlow Control Tower</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-caption'>Fresh Produce Cold-Chain DSS</div>", unsafe_allow_html=True)
    page = st.radio(
        "Workspace",
        [
            "Executive Overview",
            "Demand & Supply Forecast",
            "Network & Cold-Chain Optimizer",
            "Data & Input Templates",
            "Supply Chain Strategy & Context",
            "User Guide & Methodology",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("ACTIVE DATASET")
    st.info(st.session_state["data_source_label"])
    st.caption("GIM PGDM-BDA · Supply Chain Analytics")

df_ts = st.session_state["active_ts"]

# Top Control-Tower Bar
st.markdown(f"""
<div class="topbar">
    <div>
        <div class="topbar-title">{page}</div>
        <div class="topbar-sub">Western India Horticulture Supply Network · Tomato & Perishables</div>
    </div>
    <div class="corridor-pill">MH · GA · KA Corridor Active</div>
</div>
""", unsafe_allow_html=True)

def style_fig(fig, height=370):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=45, b=20),
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        title_font=dict(size=14, color="#0F172A"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0")
    return fig

# -------------------------------------------------------------------------
# 1. EXECUTIVE OVERVIEW
# -------------------------------------------------------------------------
if page == "Executive Overview":
    col_flt, _ = st.columns([2, 3])
    seasons = col_flt.multiselect(
        "Filter Agricultural Season",
        list(df_ts["Agri_Season"].unique()),
        default=list(df_ts["Agri_Season"].unique()),
    )
    df_f = df_ts[df_ts["Agri_Season"].isin(seasons)] if seasons else df_ts

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Weekly Supply", f"{df_f['Total_Cluster_Supply_MT'].mean():,.1f} MT", f"Peak {df_f['Total_Cluster_Supply_MT'].max():,.0f} MT")
    c2.metric("Avg Weekly Demand", f"{df_f['Total_Hub_Demand_MT'].mean():,.1f} MT", f"Peak {df_f['Total_Hub_Demand_MT'].max():,.0f} MT")
    amb_avg = df_f["Ambient_Spoilage_Pct"].mean() * 100
    cold_avg = df_f["ColdChain_Spoilage_Pct"].mean() * 100
    c3.metric("Ambient Spoilage Rate", f"{amb_avg:.2f}%", f"+{amb_avg - cold_avg:.2f}% vs Cold Chain", delta_color="inverse")
    c4.metric("Avg Mandi Modal Price", f"₹{df_f['Mandi_Modal_Price_INR_Kg'].mean():.2f}/kg", f"₹{df_f['Mandi_Modal_Price_INR_Kg'].min():.1f} – ₹{df_f['Mandi_Modal_Price_INR_Kg'].max():.1f}")

    st.write("")
    col_left, col_right = st.columns(2)
    with col_left:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=df_f["Week_Start_Date"], y=df_f["Total_Cluster_Supply_MT"], name="Cluster Supply (MT)", line=dict(color="#0F766E", width=2.2)))
        fig_ts.add_trace(go.Scatter(x=df_f["Week_Start_Date"], y=df_f["Total_Hub_Demand_MT"], name="Urban Demand (MT)", line=dict(color="#D97706", width=2.0, dash="dot")))
        fig_ts.update_layout(title="Weekly Supply vs. Urban Demand (MT)", yaxis_title="Metric Tonnes")
        st.plotly_chart(style_fig(fig_ts), use_container_width=True)

    with col_right:
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(x=df_f["Week_Start_Date"], y=df_f["Ambient_Spoilage_Pct"] * 100, name="Ambient Transit Loss (%)", line=dict(color="#DC2626", width=2)))
        fig_sp.add_trace(go.Scatter(x=df_f["Week_Start_Date"], y=df_f["ColdChain_Spoilage_Pct"] * 100, name="Cold-Chain Transit Loss (%)", line=dict(color="#059669", width=2)))
        fig_sp.update_layout(title="Post-Harvest Spoilage: Ambient vs. Pre-Cooled Cold Chain", yaxis_title="Loss (%)")
        st.plotly_chart(style_fig(fig_sp), use_container_width=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        cl_means = [{"Cluster": name, "Mean Supply (MT)": round(df_f[col].mean(), 1)} for _, name, _, col, _, _ in CLUSTERS_META]
        fig_cl = px.bar(pd.DataFrame(cl_means), x="Cluster", y="Mean Supply (MT)", title="Mean Weekly Supply by Farming Cluster", color_discrete_sequence=["#0F766E"])
        st.plotly_chart(style_fig(fig_cl, 340), use_container_width=True)
    with col_b2:
        hb_means = [{"Hub": h["name"], "Mean Demand (MT)": round(df_f[h["col"]].mean(), 1)} for h in HUBS_META]
        fig_hb = px.bar(pd.DataFrame(hb_means), x="Hub", y="Mean Demand (MT)", title="Mean Weekly Demand by Urban Hub", color_discrete_sequence=["#0284C7"])
        st.plotly_chart(style_fig(fig_hb, 340), use_container_width=True)

# -------------------------------------------------------------------------
# 2. DEMAND & SUPPLY FORECAST
# -------------------------------------------------------------------------
elif page == "Demand & Supply Forecast":
    c_sel, c_hor, c_trn = st.columns([2, 1, 1])
    target_choice = c_sel.selectbox("Time Series to Forecast", ["Total_Hub_Demand_MT", "Total_Cluster_Supply_MT", "DH01_Mumbai_Demand_MT", "DH03_Goa_Demand_MT", "Mandi_Modal_Price_INR_Kg"])
    horizon = c_hor.slider("Horizon (Weeks)", 4, 16, 8)
    train_split = c_trn.slider("Train Split (Weeks)", 70, min(96, len(df_ts) - 6), min(92, len(df_ts) - 12))

    y = df_ts[target_choice].values
    n_tot = len(y)

    def run_seasonal_ar(y_arr, tr_end, h_ahead):
        t_tr = np.arange(1, tr_end + 1)
        y_tr = y_arr[:tr_end]
        X_tr = np.column_stack([
            np.ones(tr_end), t_tr,
            np.sin(2 * np.pi * t_tr / 52.0), np.cos(2 * np.pi * t_tr / 52.0),
            np.sin(4 * np.pi * t_tr / 52.0), np.cos(4 * np.pi * t_tr / 52.0),
        ])
        beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        resid = y_tr - X_tr @ beta
        phi = np.dot(resid[1:], resid[:-1]) / max(1e-9, np.dot(resid[:-1], resid[:-1]))

        t_te = np.arange(tr_end + 1, len(y_arr) + 1)
        X_te = np.column_stack([
            np.ones(len(t_te)), t_te,
            np.sin(2 * np.pi * t_te / 52.0), np.cos(2 * np.pi * t_te / 52.0),
            np.sin(4 * np.pi * t_te / 52.0), np.cos(4 * np.pi * t_te / 52.0),
        ])
        val_preds = np.array([(X_te[h] @ beta) + (phi ** (h + 1)) * resid[-1] for h in range(len(t_te))])
        val_actual = y_arr[tr_end:]
        mape = np.mean(np.abs((val_actual - val_preds) / np.maximum(1e-6, val_actual))) * 100.0
        rmse = np.sqrt(np.mean((val_actual - val_preds) ** 2))

        t_all = np.arange(1, len(y_arr) + 1)
        X_all = np.column_stack([
            np.ones(len(y_arr)), t_all,
            np.sin(2 * np.pi * t_all / 52.0), np.cos(2 * np.pi * t_all / 52.0),
            np.sin(4 * np.pi * t_all / 52.0), np.cos(4 * np.pi * t_all / 52.0),
        ])
        beta_f, *_ = np.linalg.lstsq(X_all, y_arr, rcond=None)
        res_f = y_arr - X_all @ beta_f
        phi_f = np.dot(res_f[1:], res_f[:-1]) / max(1e-9, np.dot(res_f[:-1], res_f[:-1]))
        sig_f = np.std(res_f[1:] - phi_f * res_f[:-1])

        t_fut = np.arange(len(y_arr) + 1, len(y_arr) + h_ahead + 1)
        X_fut = np.column_stack([
            np.ones(h_ahead), t_fut,
            np.sin(2 * np.pi * t_fut / 52.0), np.cos(2 * np.pi * t_fut / 52.0),
            np.sin(4 * np.pi * t_fut / 52.0), np.cos(4 * np.pi * t_fut / 52.0),
        ])
        fut_preds = np.array([(X_fut[h] @ beta_f) + (phi_f ** (h + 1)) * res_f[-1] for h in range(h_ahead)])
        fut_se = np.array([sig_f * np.sqrt(sum((phi_f ** (2 * j)) for j in range(h + 1))) for h in range(h_ahead)])
        return val_preds, mape, rmse, fut_preds, fut_preds - 1.96 * fut_se, fut_preds + 1.96 * fut_se

    val_preds, mape, rmse, fut_preds, ci_low, ci_high = run_seasonal_ar(y, train_split, horizon)

    k1, k2, k3 = st.columns(3)
    k1.metric("Out-of-Sample MAPE", f"{mape:.2f}%", "High Accuracy (<10%)" if mape < 10 else "Moderate")
    k2.metric("Out-of-Sample RMSE", f"{rmse:,.1f}")
    k3.metric(f"Next {horizon}-Wk Mean Forecast", f"{fut_preds.mean():,.1f}")

    st.write("")
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=list(range(1, n_tot + 1)), y=y, name="Historical Actual", line=dict(color="#334155", width=2)))
    fig_fc.add_trace(go.Scatter(x=list(range(train_split + 1, n_tot + 1)), y=val_preds, name="Holdout Validation Fit", line=dict(color="#2563EB", width=2, dash="dash")))
    fut_x = list(range(n_tot + 1, n_tot + horizon + 1))
    fig_fc.add_trace(go.Scatter(x=fut_x, y=fut_preds, name="Point Forecast", line=dict(color="#059669", width=2.5)))
    fig_fc.add_trace(go.Scatter(x=fut_x + fut_x[::-1], y=list(ci_high) + list(ci_low[::-1]), fill="toself", fillcolor="rgba(5,150,105,0.14)", line=dict(color="rgba(255,255,255,0)"), name="95% Confidence Band"))
    fig_fc.update_layout(title=f"Seasonal-AR Forecast & Holdout Validation: {target_choice}", xaxis_title="Week Index", yaxis_title=target_choice)
    st.plotly_chart(style_fig(fig_fc, 410), use_container_width=True)

    fc_df = pd.DataFrame({
        "Future_Week_Index": fut_x,
        "Point_Forecast": np.round(fut_preds, 2),
        "Lower_95_CI": np.round(ci_low, 2),
        "Upper_95_CI": np.round(ci_high, 2),
    })
    with st.expander("View Forecast Table & Export"):
        st.dataframe(fc_df, use_container_width=True)
        st.download_button("Download Forecast CSV", fc_df.to_csv(index=False).encode("utf-8"), "forecast_results.csv", "text/csv")

# -------------------------------------------------------------------------
# 3. NETWORK & COLD-CHAIN OPTIMIZER
# -------------------------------------------------------------------------
elif page == "Network & Cold-Chain Optimizer":
    col_cfg, col_res = st.columns([1, 2.2], gap="large")

    with col_cfg:
        st.markdown("<div class='section-card'><b>Goal Programming & Scenario Controls</b></div>", unsafe_allow_html=True)
        w_cost = st.slider("Cost Control Weight", 0.0, 1.0, 0.35, 0.05)
        w_spoil = st.slider("Spoilage Reduction Weight", 0.0, 1.0, 0.35, 0.05)
        w_short = st.slider("Fill-Rate SLA Weight", 0.0, 1.0, 0.30, 0.05)
        max_ph = st.slider("Max Packhouses to Open", 1, 6, 4)
        cost_target_lakhs = st.number_input("Weekly Cost Target (₹ Lakhs)", 300.0, 900.0, 560.0, 10.0)
        spoil_target_mt = st.number_input("Weekly Spoilage Target (MT)", 20.0, 300.0, 85.0, 5.0)
        spot_cap_pct = st.slider("Max Spot-Market Share (%)", 0, 50, 20, 5) / 100.0
        st.markdown("---")
        p_adverse = st.slider("Adverse Weather Prob (0.78x Yield)", 0.0, 0.8, 0.25, 0.05)
        p_normal = st.slider("Normal Monsoon Prob (1.00x Yield)", 0.0, 1.0, 0.50, 0.05)
        p_bumper = max(0.0, round(1.0 - p_adverse - p_normal, 2))
        st.caption(f"Bumper Glut Prob (auto-balanced): **{p_bumper:.2f}**")
        run_btn = st.button("Run Network Optimization", type="primary", use_container_width=True)

    with col_res:
        if abs(p_adverse + p_normal + p_bumper - 1.0) > 0.01:
            st.error("Scenario probabilities must sum to 1.00. Adjust Adverse and Normal sliders.")
        else:
            scenarios = [
                {"id": "S1_Adverse", "prob": p_adverse, "yield": 0.78, "amb_sp_m": 1.35, "ref_sp_m": 1.08, "spot_m": 1.42, "dem_m": 1.03},
                {"id": "S2_Normal",  "prob": p_normal,  "yield": 1.00, "amb_sp_m": 1.00, "ref_sp_m": 1.00, "spot_m": 1.00, "dem_m": 1.00},
                {"id": "S3_Bumper",  "prob": p_bumper,  "yield": 1.24, "amb_sp_m": 1.12, "ref_sp_m": 1.02, "spot_m": 0.76, "dem_m": 0.98},
            ]
            sup_base = {cid: df_ts[col].mean() for cid, _, _, col, _, _ in CLUSTERS_META}
            dem_base = {h["id"]: df_ts[h["col"]].mean() for h in HUBS_META}
            rt_map = {(r["Origin_ID"], r["Destination_ID"]): r for _, r in df_sample_rt.iterrows()}
            ph_ids = [p["id"] for p in PACKHOUSES_META]
            ph_dict = {p["id"]: p for p in PACKHOUSES_META}
            cl_ids = [c[0] for c in CLUSTERS_META]
            hb_ids = [h["id"] for h in HUBS_META]
            hb_dict = {h["id"]: h for h in HUBS_META}
            modes = ["AMB", "REF"]

            idx_y = {j: i for i, j in enumerate(ph_ids)}
            offset = len(ph_ids)
            idx_x, idx_z, idx_spot, idx_short = {}, {}, {}, {}
            for s in scenarios:
                for i in cl_ids:
                    for j in ph_ids:
                        for m in modes:
                            idx_x[(i, j, m, s["id"])] = offset; offset += 1
                for j in ph_ids:
                    for k in hb_ids:
                        for m in modes:
                            idx_z[(j, k, m, s["id"])] = offset; offset += 1
                for k in hb_ids:
                    idx_spot[(k, s["id"])] = offset; offset += 1
                    idx_short[(k, s["id"])] = offset; offset += 1

            idx_dcost, idx_dspoil, idx_dshort = offset, offset + 1, offset + 2
            n_vars = offset + 3

            c_obj = np.zeros(n_vars)
            c_obj[idx_dcost] = w_cost / (cost_target_lakhs * 100000.0)
            c_obj[idx_dspoil] = w_spoil / max(1.0, spoil_target_mt)
            c_obj[idx_dshort] = w_short / 120.0
            for j in ph_ids:
                c_obj[idx_y[j]] += 1e-9 * (ph_dict[j]["fixed_wk"] + ph_dict[j]["capex_l"] * 100000.0 / 52.0)

            rows_A, lb_A, ub_A = [], [], []
            def add_con(coeffs, lb, ub):
                row = np.zeros(n_vars)
                for idx, val in coeffs.items():
                    row[idx] += val
                rows_A.append(row); lb_A.append(lb); ub_A.append(ub)

            for s in scenarios:
                for i in cl_ids:
                    add_con({idx_x[(i, j, m, s["id"])]: 1.0 for j in ph_ids for m in modes}, 0.0, sup_base[i] * s["yield"])
                for j in ph_ids:
                    coeffs = {idx_x[(i, j, m, s["id"])]: 1.0 for i in cl_ids for m in modes}
                    coeffs[idx_y[j]] = -ph_dict[j]["cap"]
                    add_con(coeffs, -np.inf, 0.0)
                    fb = {}
                    for i in cl_ids:
                        r = rt_map[(i, j)]
                        fb[idx_x[(i, j, "AMB", s["id"])]] = 1.0 - min(0.25, r["Ambient_Transit_Loss_Pct"] * s["amb_sp_m"])
                        fb[idx_x[(i, j, "REF", s["id"])]] = 1.0 - min(0.10, r["Reefer_Transit_Loss_Pct"] * s["ref_sp_m"] * (1 - 0.3 * ph_dict[j]["loss_red"]))
                    for k in hb_ids:
                        for m in modes:
                            fb[idx_z[(j, k, m, s["id"])]] = -1.0
                    add_con(fb, 0.0, 0.0)
                for k in hb_ids:
                    dc = {}
                    for j in ph_ids:
                        r = rt_map[(j, k)]
                        dc[idx_z[(j, k, "AMB", s["id"])]] = 1.0 - min(0.25, r["Ambient_Transit_Loss_Pct"] * s["amb_sp_m"] * (1 - 0.5 * ph_dict[j]["loss_red"]))
                        dc[idx_z[(j, k, "REF", s["id"])]] = 1.0 - min(0.08, r["Reefer_Transit_Loss_Pct"] * s["ref_sp_m"] * (1 - ph_dict[j]["loss_red"]))
                    dc[idx_spot[(k, s["id"])]] = 1.0
                    dc[idx_short[(k, s["id"])]] = 1.0
                    dem_sk = dem_base[k] * s["dem_m"]
                    add_con(dc, dem_sk, dem_sk)
                    add_con({idx_spot[(k, s["id"])]: 1.0}, 0.0, spot_cap_pct * dem_sk)

            add_con({idx_y[j]: 1.0 for j in ph_ids}, 1.0, float(max_ph))

            cost_coeffs = {idx_y[j]: ph_dict[j]["fixed_wk"] + ph_dict[j]["capex_l"] * 100000.0 / 52.0 for j in ph_ids}
            spoil_coeffs = {}
            for s in scenarios:
                p = s["prob"]
                if p <= 0:
                    continue
                for i in cl_ids:
                    for j in ph_ids:
                        r = rt_map[(i, j)]
                        cost_coeffs[idx_x[(i, j, "AMB", s["id"])]] = p * (18000.0 + r["Ambient_Freight_INR_MT"] + ph_dict[j]["var_mt"])
                        cost_coeffs[idx_x[(i, j, "REF", s["id"])]] = p * (18000.0 + r["Reefer_Freight_INR_MT"] + ph_dict[j]["var_mt"])
                        spoil_coeffs[idx_x[(i, j, "AMB", s["id"])]] = p * min(0.25, r["Ambient_Transit_Loss_Pct"] * s["amb_sp_m"])
                        spoil_coeffs[idx_x[(i, j, "REF", s["id"])]] = p * min(0.10, r["Reefer_Transit_Loss_Pct"] * s["ref_sp_m"] * (1 - 0.3 * ph_dict[j]["loss_red"]))
                for j in ph_ids:
                    for k in hb_ids:
                        r = rt_map[(j, k)]
                        cost_coeffs[idx_z[(j, k, "AMB", s["id"])]] = p * r["Ambient_Freight_INR_MT"]
                        cost_coeffs[idx_z[(j, k, "REF", s["id"])]] = p * r["Reefer_Freight_INR_MT"]
                        spoil_coeffs[idx_z[(j, k, "AMB", s["id"])]] = p * min(0.25, r["Ambient_Transit_Loss_Pct"] * s["amb_sp_m"] * (1 - 0.5 * ph_dict[j]["loss_red"]))
                        spoil_coeffs[idx_z[(j, k, "REF", s["id"])]] = p * min(0.08, r["Reefer_Transit_Loss_Pct"] * s["ref_sp_m"] * (1 - ph_dict[j]["loss_red"]))
                for k in hb_ids:
                    cost_coeffs[idx_spot[(k, s["id"])]] = p * hb_dict[k]["spot_mt"] * s["spot_m"]
                    cost_coeffs[idx_short[(k, s["id"])]] = p * 42000.0

            cost_g = dict(cost_coeffs); cost_g[idx_dcost] = -1.0
            add_con(cost_g, -np.inf, cost_target_lakhs * 100000.0)
            spoil_g = dict(spoil_coeffs); spoil_g[idx_dspoil] = -1.0
            add_con(spoil_g, -np.inf, spoil_target_mt)
            short_coeffs = {idx_short[(k, s["id"])]: s["prob"] for s in scenarios for k in hb_ids}
            short_g = dict(short_coeffs); short_g[idx_dshort] = -1.0
            add_con(short_g, -np.inf, 0.0)

            integrality = np.zeros(n_vars); integrality[:len(ph_ids)] = 1
            ub_v = np.full(n_vars, np.inf); ub_v[:len(ph_ids)] = 1.0
            res = milp(c=c_obj, integrality=integrality, bounds=Bounds(np.zeros(n_vars), ub_v), constraints=LinearConstraint(np.array(rows_A), lb_A, ub_A))

            if res.success:
                open_ids = {ph_ids[i] for i in range(len(ph_ids)) if res.x[i] > 0.5}
                opened_names = [ph_dict[j]["name"] for j in ph_ids if j in open_ids]
                exp_cost_l = sum(res.x[idx] * val for idx, val in cost_coeffs.items()) / 100000.0
                exp_spoil_mt = sum(res.x[idx] * val for idx, val in spoil_coeffs.items())
                exp_short_mt = sum(res.x[idx] * val for idx, val in short_coeffs.items())
                tot_dem = sum(dem_base.values())
                fill_rate = (1.0 - exp_short_mt / tot_dem) * 100.0
                spoil_pct = (exp_spoil_mt / tot_dem) * 100.0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Packhouses Opened", f"{len(open_ids)} / {max_ph}")
                m2.metric("Expected Weekly Cost", f"₹{exp_cost_l:,.2f} L", f"Target ₹{cost_target_lakhs:.0f} L")
                m3.metric("Expected Spoilage", f"{spoil_pct:.2f}%", f"-{11.61 - spoil_pct:.2f}% vs Baseline")
                m4.metric("Expected Fill Rate", f"{fill_rate:.2f}%", f"{exp_short_mt:.1f} MT shortfall")

                if exp_short_mt > 50.0:
                    st.markdown(f"<div class='alert-banner'><b>Capacity Alert:</b> Expected shortfall is {exp_short_mt:.1f} MT/week. Increase Max Packhouses or raise the spot-market cap to protect retail SLAs.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='success-banner'><b>Recommended Network:</b> Open <b>{', '.join(opened_names)}</b> to achieve a {fill_rate:.1f}% fill rate and reduce spoilage to {spoil_pct:.2f}%.</div>", unsafe_allow_html=True)

                # Corridor Network Map (Clusters, Open/Closed Packhouses, Urban Hubs)
                fig_map = go.Figure()
                fig_map.add_trace(go.Scattergeo(
                    lat=[c[4] for c in CLUSTERS_META], lon=[c[5] for c in CLUSTERS_META],
                    text=[f"Cluster: {c[1]}" for c in CLUSTERS_META], mode="markers",
                    marker=dict(size=9, color="#0284C7", symbol="circle"), name="Farming Clusters (8)"
                ))
                fig_map.add_trace(go.Scattergeo(
                    lat=[p["lat"] for p in PACKHOUSES_META if p["id"] in open_ids],
                    lon=[p["lon"] for p in PACKHOUSES_META if p["id"] in open_ids],
                    text=[f"OPEN: {p['name']}" for p in PACKHOUSES_META if p["id"] in open_ids], mode="markers+text",
                    textposition="top right", marker=dict(size=13, color="#059669", symbol="square"), name="Opened Packhouses"
                ))
                closed_phs = [p for p in PACKHOUSES_META if p["id"] not in open_ids]
                if closed_phs:
                    fig_map.add_trace(go.Scattergeo(
                        lat=[p["lat"] for p in closed_phs], lon=[p["lon"] for p in closed_phs],
                        text=[f"Closed: {p['name']}" for p in closed_phs], mode="markers",
                        marker=dict(size=9, color="#CBD5E1", symbol="x"), name="Not Selected"
                    ))
                fig_map.add_trace(go.Scattergeo(
                    lat=[h["lat"] for h in HUBS_META], lon=[h["lon"] for h in HUBS_META],
                    text=[f"Hub: {h['name']}" for h in HUBS_META], mode="markers",
                    marker=dict(size=12, color="#D97706", symbol="diamond"), name="Urban Demand Hubs (4)"
                ))
                fig_map.update_geos(
                    scope="asia", center=dict(lat=16.8, lon=75.2), projection_scale=11.5,
                    showland=True, landcolor="#F1F5F9", countrycolor="#CBD5E1", coastlinewidth=0.8
                )
                fig_map.update_layout(title="Optimal Cold-Chain Corridor Footprint (MH – GA – KA)", height=380, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.05))
                st.plotly_chart(fig_map, use_container_width=True)

                ph_rows = []
                for j in ph_ids:
                    is_open = j in open_ids
                    flow_n = sum(res.x[idx_x[(i, j, m, "S2_Normal")]] for i in cl_ids for m in modes) if p_normal > 0 else 0.0
                    ph_rows.append({
                        "Packhouse_ID": j,
                        "Facility_Name": ph_dict[j]["name"],
                        "Decision": "OPEN" if is_open else "CLOSED",
                        "Capacity_MT_Wk": ph_dict[j]["cap"],
                        "Normal_Flow_MT": round(flow_n, 1),
                        "Utilization_Pct": round((flow_n / ph_dict[j]["cap"]) * 100, 1) if is_open else 0.0,
                    })
                df_ph_out = pd.DataFrame(ph_rows)
                st.dataframe(df_ph_out, use_container_width=True)
                st.download_button("Export Optimal Decisions (CSV)", df_ph_out.to_csv(index=False).encode("utf-8"), "optimal_packhouse_decisions.csv", "text/csv")
            else:
                st.error(f"Optimization did not converge: {res.message}")

# -------------------------------------------------------------------------
# 4. DATA & INPUT TEMPLATES
# -------------------------------------------------------------------------
elif page == "Data & Input Templates":
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.download_button(
            "Download Blank Input Template (CSV)",
            pd.DataFrame(columns=df_sample_ts.columns).to_csv(index=False).encode("utf-8"),
            "blank_input_template.csv",
            "text/csv",
            use_container_width=True,
        )
    with col_u2:
        if st.button("Reset to Built-In 104-Week Dataset", use_container_width=True):
            st.session_state["active_ts"] = df_sample_ts.copy()
            st.session_state["data_source_label"] = "104-Week Corridor Dataset (Nov 2024 – Oct 2026)"
            st.success("Restored built-in 104-week corridor dataset.")

    uploaded = st.file_uploader("Upload Custom Weekly Time-Series CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            req_cols = set(df_sample_ts.columns)
            missing_cols = req_cols - set(df_up.columns)
            if missing_cols:
                st.error(f"Schema Validation Error: Missing columns {sorted(missing_cols)}")
            elif len(df_up) < 12:
                st.error("Range Validation Error: At least 12 weekly rows are required.")
            elif (df_up["Total_Cluster_Supply_MT"] < 0).any() or (df_up["Total_Hub_Demand_MT"] < 0).any():
                st.error("Value Validation Error: Supply and demand values must be non-negative.")
            else:
                st.session_state["active_ts"] = df_up
                st.session_state["data_source_label"] = f"Custom Upload ({uploaded.name}, {len(df_up)} rows)"
                st.success(f"Validated and loaded {len(df_up)} rows from {uploaded.name}.")
        except Exception as e:
            st.error(f"File Parsing Error: {e}")

    st.dataframe(st.session_state["active_ts"].head(20), use_container_width=True)

# -------------------------------------------------------------------------
# 5. SUPPLY CHAIN STRATEGY & CONTEXT
# -------------------------------------------------------------------------
elif page == "Supply Chain Strategy & Context":
    t1, t2, t3 = st.tabs(["Supply Chain Strategy & Systems View", "Quantitative Problem Framing", "Technology Outlook"])

    with t1:
        st.markdown("""
        <div class='section-card'>
        <h4>End-to-End Corridor Value Chain & Strategic Fit</h4>
        <ul>
            <li><b>Key Players:</b> 8 horticulture farming clusters (Nashik, Narayangaon, Satara, Sangli, Kolhapur, Belagavi, Dharwad, Kolar), integrated pre-cooling packhouses, reefer & ambient fleets, and 4 urban consumption hubs (Mumbai, Pune, Goa, Bengaluru).</li>
            <li><b>Flows:</b> <i>Material</i> (fresh tomato/produce farm-to-DC), <i>Information</i> (weekly Agmarknet arrival & weather telemetry, DC fill-rate SLAs), and <i>Money</i> (farm payouts, packhouse CapEx/OpEx, wholesale revenue, shortfall penalties).</li>
            <li><b>Strategic Fit:</b> Competes on <b>Responsive, Low-Wastage Fulfilment</b>. Pre-cooling packhouses and reefer routing trade higher fixed/freight cost to slash transit spoilage from the 11.61% NABCONS ambient benchmark to under 2.5%.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Clean Inline SVG Systems View & Business Architecture
        st.markdown("""
        <div class='section-card'>
        <h4>Systems View (Causal Loops) & Business Architecture</h4>
        <svg viewBox="0 0 860 220" width="100%" xmlns="http://www.w3.org/2000/svg" style="font-family:Inter,sans-serif;">
            <rect x="10" y="15" width="250" height="190" rx="8" fill="#F8FAFC" stroke="#CBD5E1"/>
            <text x="135" y="40" text-anchor="middle" font-size="12" font-weight="700" fill="#0F172A">1. Data & Signals Layer</text>
            <text x="135" y="75" text-anchor="middle" font-size="11" fill="#334155">104W Supply & Demand History</text>
            <text x="135" y="105" text-anchor="middle" font-size="11" fill="#334155">IMD Temp & Monsoon Regimes</text>
            <text x="135" y="135" text-anchor="middle" font-size="11" fill="#334155">72-Arc Distance & Spoilage Rates</text>
            <text x="135" y="165" text-anchor="middle" font-size="11" fill="#334155">Packhouse CapEx / OpEx Limits</text>

            <line x1="265" y1="110" x2="300" y2="110" stroke="#0F766E" stroke-width="2.5" marker-end="url(#arrow)"/>

            <rect x="305" y="15" width="250" height="190" rx="8" fill="#ECFDF5" stroke="#6EE7B7"/>
            <text x="430" y="40" text-anchor="middle" font-size="12" font-weight="700" fill="#065F46">2. Analytics & Optimization Core</text>
            <text x="430" y="78" text-anchor="middle" font-size="11" fill="#065F46">Seasonal-AR Forecaster (4–16W)</text>
            <text x="430" y="112" text-anchor="middle" font-size="11" fill="#065F46">Two-Stage Stochastic Scenarios</text>
            <text x="430" y="146" text-anchor="middle" font-size="11" fill="#065F46">Goal Programming MILP (HiGHS)</text>
            <text x="430" y="176" text-anchor="middle" font-size="10" fill="#047857">Balances Cost · Spoilage · SLA</text>

            <line x1="560" y1="110" x2="595" y2="110" stroke="#0F766E" stroke-width="2.5"/>

            <rect x="600" y="15" width="250" height="190" rx="8" fill="#EFF6FF" stroke="#93C5FD"/>
            <text x="725" y="40" text-anchor="middle" font-size="12" font-weight="700" fill="#1E3A8A">3. Managerial Decisions</text>
            <text x="725" y="78" text-anchor="middle" font-size="11" fill="#1E40AF">Packhouse Open / Close (y_j)</text>
            <text x="725" y="112" text-anchor="middle" font-size="11" fill="#1E40AF">Reefer vs. Ambient Allocation</text>
            <text x="725" y="146" text-anchor="middle" font-size="11" fill="#1E40AF">Spot-Market Hedging Caps</text>
            <text x="725" y="176" text-anchor="middle" font-size="10" fill="#1D4ED8">CSV Export & SLA Alerts</text>
        </svg>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class='section-card'>
        <h4>Mathematical Formulation & Modeling Scope</h4>
        <ul>
            <li><b>Decision Variables:</b> Binary facility selection <code>y_j ∈ {0,1}</code>; continuous first-mile flows <code>x_{ijms} ≥ 0</code> and line-haul flows <code>z_{jkms} ≥ 0</code> across ambient/reefer modes and weather scenarios; spot purchases <code>u_{ks}</code>; shortfall <code>v_{ks}</code>; goal deviations <code>d_1^+, d_2^+, d_3^+</code>.</li>
            <li><b>Objective:</b> Minimize weighted normalized deviations across Weekly Cost, Expected Spoilage (MT), and Unmet Demand Shortfall.</li>
            <li><b>Key Constraints:</b> Cluster supply limits, packhouse capacity linking (<code>∑ x ≤ Cap_j · y_j</code>), spoilage-adjusted flow conservation, urban hub demand balance, max packhouses CapEx limit, and spot-market traceability cap.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class='section-card'>
        <h4>Technology Outlook: Next-Generation Cold-Chain Enablers</h4>
        <p><b>1. IoT Cold-Chain Telemetry & Ethylene Sensors:</b> Real-time temperature and humidity logging inside reefer trucks enables Dynamic FEFO (First-Expired-First-Out) rerouting. <i>Challenges & Ethics:</i> Rural cellular dead-zones, hardware CapEx for smallholder FPOs, and fair data-ownership governance so farmers are not unfairly penalized for transit delays outside their control.</p>
        <p><b>2. Digital Twins & Agentic AI Procurement:</b> A corridor Digital Twin mirrors packhouse queues and highway congestion, while Agentic AI monitors IMD heatwave alerts to trigger re-forecasts and draft dispatch updates for manager sign-off. <i>Challenges & Ethics:</i> ERP integration complexity, hallucination risks during extreme climate shocks, and maintaining clear human-in-the-loop accountability.</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 6. USER GUIDE & METHODOLOGY
# -------------------------------------------------------------------------
else:
    st.markdown("""
    <div class='section-card'>
    <h4>Step-by-Step User Guide</h4>
    <ol>
        <li><b>Executive Overview:</b> Filter by season to inspect historical supply/demand balance and ambient vs. cold-chain spoilage.</li>
        <li><b>Demand & Supply Forecast:</b> Choose a series, set the forecast horizon (4–16 weeks), review holdout MAPE/RMSE, and export forecasts.</li>
        <li><b>Network & Cold-Chain Optimizer:</b> Adjust goal weights, CapEx packhouse limits, and weather probabilities, then click <b>Run Network Optimization</b> to view the corridor map, utilization table, and exportable decisions.</li>
        <li><b>Data & Input Templates:</b> Download the blank CSV template, enter custom data, and upload it with automatic schema and range validation.</li>
    </ol>
    <h4>Visible Assumptions & Limitations</h4>
    <ul>
        <li>The built-in 104-week dataset is synthetically generated (<code>seed=42</code>) and calibrated to NABCONS (2022) tomato loss benchmarks (11.61%) and NCCD cold-chain parameters.</li>
        <li>The MILP uses linear scenario-adjusted arc spoilage rates rather than continuous nonlinear thermal decay, and aggregates operations to a weekly cadence.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)