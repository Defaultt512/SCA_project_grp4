import io
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
from scipy.optimize import milp, LinearConstraint, Bounds
import streamlit as st

st.set_page_config(
    page_title="AgriColdChain DSS | GIM Supply Chain Analytics",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# CUSTOM STYLING
# -------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1B4D3E 0%, #2C6E49 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .main-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
    .main-header p { margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }
    .kpi-box {
        background-color: #F4F9F6;
        border-left: 4px solid #1B4D3E;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .alert-box {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .rec-box {
        background-color: #ECFDF5;
        border-left: 4px solid #059669;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# LOAD BUILT-IN SAMPLE DATA & MASTER PARAMETERS
# -------------------------------------------------------------------------
import base64, gzip, os
_TS_GZ_B64 = "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"
_RT_GZ_B64 = "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"

@st.cache_data
def load_sample_data():
    if not os.path.exists("agri_coldchain_104w_timeseries.csv"):
        with open("agri_coldchain_104w_timeseries.csv", "wb") as f:
            f.write(gzip.decompress(base64.b64decode(_TS_GZ_B64)))
    if not os.path.exists("agri_coldchain_network_routes.csv"):
        with open("agri_coldchain_network_routes.csv", "wb") as f:
            f.write(gzip.decompress(base64.b64decode(_RT_GZ_B64)))
    df_ts = pd.read_csv("agri_coldchain_104w_timeseries.csv")
    df_rt = pd.read_csv("agri_coldchain_network_routes.csv")
    return df_ts, df_rt

df_sample_ts, df_sample_rt = load_sample_data()

if "active_ts" not in st.session_state:
    st.session_state["active_ts"] = df_sample_ts.copy()
if "data_source_label" not in st.session_state:
    st.session_state["data_source_label"] = "Built-in Sample Dataset (104 Weeks, Nov 2024 – Oct 2026)"

CLUSTERS_META = [
    ("FC01", "Nashik (Pimpalgaon)", "Maharashtra", "FC01_Nashik_MT"),
    ("FC02", "Narayangaon (Junner)", "Maharashtra", "FC02_Narayangaon_MT"),
    ("FC03", "Satara (Wai-Karad)", "Maharashtra", "FC03_Satara_MT"),
    ("FC04", "Sangli (Miraj-Tasgaon)", "Maharashtra", "FC04_Sangli_MT"),
    ("FC05", "Kolhapur (Shirol)", "Maharashtra", "FC05_Kolhapur_MT"),
    ("FC06", "Belagavi (Gokak)", "Karnataka", "FC06_Belagavi_MT"),
    ("FC07", "Dharwad (Hubballi)", "Karnataka", "FC07_Dharwad_MT"),
    ("FC08", "Kolar (Mulbagal)", "Karnataka", "FC08_Kolar_MT"),
]

PACKHOUSES_META = [
    {"id": "PH01", "name": "Ozarkhed Packhouse Hub (Nashik)", "cap": 850.0, "capex_l": 68.0, "fixed_wk": 145000.0, "var_mt": 620.0, "loss_red": 0.68},
    {"id": "PH02", "name": "Chakan-Narayangaon Agro Hub (Pune N)", "cap": 750.0, "capex_l": 64.0, "fixed_wk": 135000.0, "var_mt": 590.0, "loss_red": 0.66},
    {"id": "PH03", "name": "Karad-Satara Integrated Packhouse", "cap": 650.0, "capex_l": 54.0, "fixed_wk": 115000.0, "var_mt": 560.0, "loss_red": 0.65},
    {"id": "PH04", "name": "Kagal-Kolhapur Cold Chain Park", "cap": 600.0, "capex_l": 50.0, "fixed_wk": 108000.0, "var_mt": 550.0, "loss_red": 0.64},
    {"id": "PH05", "name": "Belagavi-Kittur Border Logistics Hub", "cap": 700.0, "capex_l": 56.0, "fixed_wk": 118000.0, "var_mt": 540.0, "loss_red": 0.67},
    {"id": "PH06", "name": "Hoskote-Kolar Mega Packhouse", "cap": 800.0, "capex_l": 66.0, "fixed_wk": 140000.0, "var_mt": 600.0, "loss_red": 0.68},
]

HUBS_META = [
    {"id": "DH01", "name": "Mumbai Metro DC (Vashi)", "col": "DH01_Mumbai_Demand_MT", "spot_mt": 31500.0},
    {"id": "DH02", "name": "Pune Metro DC (Marketyard)", "col": "DH02_Pune_Demand_MT", "spot_mt": 29500.0},
    {"id": "DH03", "name": "Goa Coastal Hub (Panaji/Margao)", "col": "DH03_Goa_Demand_MT", "spot_mt": 33800.0},
    {"id": "DH04", "name": "Bengaluru Metro DC (Yeshwanthpur)", "col": "DH04_Bengaluru_Demand_MT", "spot_mt": 30500.0},
]

# -------------------------------------------------------------------------
# HEADER BANNER & SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🥬 AgriColdChain DSS — Farm-to-Fork Cold-Chain & Stochastic Supply Planner</h1>
    <p>Goa Institute of Management | PGDM-BDA Supply Chain Analytics | Western India Horticulture Corridor (MH–GA–KA)</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "1. Executive Dashboard (CLO 3)",
        "2. Demand & Supply Forecasting (CLO 4)",
        "3. Cold-Chain Network Optimizer (CLO 5)",
        "4. Data Upload & Templates",
        "5. Context, Strategy & Tech Outlook (CLO 1,2,6)",
        "6. User Guide & Limitations",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(f"**Active Data:**\n{st.session_state['data_source_label']}")
df_ts = st.session_state["active_ts"]

# -------------------------------------------------------------------------
# PAGE 1: EXECUTIVE DASHBOARD (DESCRIPTIVE ANALYTICS - CLO 3)
# -------------------------------------------------------------------------
if page == "1. Executive Dashboard (CLO 3)":
    st.subheader("Descriptive Analytics: Corridor Supply, Demand & Spoilage Diagnostics")
    seasons = st.multiselect("Filter by Agricultural Season", df_ts["Agri_Season"].unique(), default=list(df_ts["Agri_Season"].unique()))
    df_f = df_ts[df_ts["Agri_Season"].isin(seasons)] if seasons else df_ts

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Weekly Supply", f"{df_f['Total_Cluster_Supply_MT'].mean():,.1f} MT", f"Peak: {df_f['Total_Cluster_Supply_MT'].max():,.0f} MT")
    c2.metric("Avg Weekly Demand", f"{df_f['Total_Hub_Demand_MT'].mean():,.1f} MT", f"Peak: {df_f['Total_Hub_Demand_MT'].max():,.0f} MT")
    amb_avg = df_f["Ambient_Spoilage_Pct"].mean() * 100
    cold_avg = df_f["ColdChain_Spoilage_Pct"].mean() * 100
    c3.metric("Ambient Spoilage Rate", f"{amb_avg:.2f}%", f"+{amb_avg - cold_avg:.2f}% vs Cold Chain", delta_color="inverse")
    c4.metric("Avg Mandi Modal Price", f"₹{df_f['Mandi_Modal_Price_INR_Kg'].mean():.2f}/kg", f"Range: ₹{df_f['Mandi_Modal_Price_INR_Kg'].min():.1f}–₹{df_f['Mandi_Modal_Price_INR_Kg'].max():.1f}")

    col_left, col_right = st.columns(2)
    with col_left:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=df_f["Week_Start_Date"], y=df_f["Total_Cluster_Supply_MT"], name="Total Cluster Supply (MT)", line=dict(color="#1B4D3E", width=2.2)))
        fig_ts.add_trace(go.Scatter(x=df_f["Week_Start_Date"], y=df_f["Total_Hub_Demand_MT"], name="Total Urban Demand (MT)", line=dict(color="#D97706", width=2.2, dash="dot")))
        fig_ts.update_layout(title="Weekly Supply vs. Urban Hub Demand (MT)", xaxis_title="Week", yaxis_title="Metric Tonnes (MT)", height=380, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_right:
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(x=df_f["Week_Start_Date"], y=df_f["Ambient_Spoilage_Pct"] * 100, name="Ambient Transport Spoilage (%)", line=dict(color="#DC2626", width=2)))
        fig_sp.add_trace(go.Scatter(x=df_f["Week_Start_Date"], y=df_f["ColdChain_Spoilage_Pct"] * 100, name="Integrated Cold-Chain Spoilage (%)", line=dict(color="#059669", width=2)))
        fig_sp.update_layout(title="Post-Harvest Spoilage Trade-off: Ambient vs. Cold Chain (%)", xaxis_title="Week", yaxis_title="Spoilage (%)", height=380, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_sp, use_container_width=True)

    # Cluster & Hub Breakdown
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        cl_means = [{"Cluster": name, "Mean Weekly Supply (MT)": round(df_f[col].mean(), 1)} for _, name, _, col in CLUSTERS_META]
        fig_cl = px.bar(pd.DataFrame(cl_means), x="Cluster", y="Mean Weekly Supply (MT)", title="Mean Weekly Supply by Farming Cluster (MT)", color_discrete_sequence=["#2C6E49"])
        st.plotly_chart(fig_cl, use_container_width=True)
    with col_b2:
        hb_means = [{"Demand Hub": h["name"], "Mean Weekly Demand (MT)": round(df_f[h["col"]].mean(), 1)} for h in HUBS_META]
        fig_hb = px.bar(pd.DataFrame(hb_means), x="Demand Hub", y="Mean Weekly Demand (MT)", title="Mean Weekly Demand by Urban Hub (MT)", color_discrete_sequence=["#0284C7"])
        st.plotly_chart(fig_hb, use_container_width=True)

# -------------------------------------------------------------------------
# PAGE 2: FORECASTING (PREDICTIVE ANALYTICS - CLO 4)
# -------------------------------------------------------------------------
elif page == "2. Demand & Supply Forecasting (CLO 4)":
    st.subheader("Predictive Analytics: Seasonal Time-Series Forecasting & Validation")
    target_choice = st.selectbox("Select Series to Forecast", ["Total_Hub_Demand_MT", "Total_Cluster_Supply_MT", "DH01_Mumbai_Demand_MT", "DH03_Goa_Demand_MT", "Mandi_Modal_Price_INR_Kg"])
    horizon = st.slider("Forecast Horizon (Weeks Ahead)", min_value=4, max_value=16, value=8)
    train_split = st.slider("Validation Split (Training Weeks)", min_value=70, max_value=min(96, len(df_ts)-6), value=min(92, len(df_ts)-12))

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
        sigma = np.std(resid[1:] - phi * resid[:-1])

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

        # Full fit for future forecast
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
    k1.metric("Out-of-Sample MAPE", f"{mape:.2f}%", "Excellent (<10%)" if mape < 10 else "Moderate")
    k2.metric("Out-of-Sample RMSE", f"{rmse:,.1f}")
    k3.metric(f"Next {horizon}-Wk Avg Forecast", f"{fut_preds.mean():,.1f}")

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=list(range(1, n_tot + 1)), y=y, name="Actual Historical", line=dict(color="#1F2937", width=2)))
    fig_fc.add_trace(go.Scatter(x=list(range(train_split + 1, n_tot + 1)), y=val_preds, name="Validation Fit (Test)", line=dict(color="#2563EB", width=2, dash="dash")))
    fut_x = list(range(n_tot + 1, n_tot + horizon + 1))
    fig_fc.add_trace(go.Scatter(x=fut_x, y=fut_preds, name="Point Forecast", line=dict(color="#059669", width=2.5)))
    fig_fc.add_trace(go.Scatter(x=fut_x + fut_x[::-1], y=list(ci_high) + list(ci_low[::-1]), fill="toself", fillcolor="rgba(5,150,105,0.18)", line=dict(color="rgba(255,255,255,0)"), name="95% Confidence Interval"))
    fig_fc.update_layout(title=f"SARIMA / Seasonal-AR Forecast & Validation: {target_choice}", xaxis_title="Week Index", yaxis_title=target_choice, height=420)
    st.plotly_chart(fig_fc, use_container_width=True)

    fc_df = pd.DataFrame({
        "Future_Week_Index": fut_x,
        "Point_Forecast": np.round(fut_preds, 2),
        "Lower_95_CI": np.round(ci_low, 2),
        "Upper_95_CI": np.round(ci_high, 2),
    })
    st.dataframe(fc_df, use_container_width=True)
    st.download_button("Download Forecast CSV", fc_df.to_csv(index=False).encode("utf-8"), "forecast_results.csv", "text/csv")

# -------------------------------------------------------------------------
# PAGE 3: PRESCRIPTIVE OPTIMIZER (GOAL PROGRAMMING + STOCHASTIC MILP - CLO 5)
# -------------------------------------------------------------------------
elif page == "3. Cold-Chain Network Optimizer (CLO 5)":
    st.subheader("Prescriptive Analytics: Goal Programming & Two-Stage Stochastic Optimizer")
    st.caption("Adjust managerial goal weights, CapEx limits, and weather scenario probabilities, then click Run Optimizer.")

    with st.expander("Solver Parameters & Managerial Goals", expanded=True):
        g1, g2, g3 = st.columns(3)
        w_cost = g1.slider("Goal 1 Weight: Cost Control", 0.0, 1.0, 0.35, 0.05)
        w_spoil = g2.slider("Goal 2 Weight: Spoilage Reduction", 0.0, 1.0, 0.35, 0.05)
        w_short = g3.slider("Goal 3 Weight: Service Fill-Rate SLA", 0.0, 1.0, 0.30, 0.05)

        p1, p2, p3, p4 = st.columns(4)
        max_ph = p1.slider("Max Packhouses to Open", 1, 6, 4)
        cost_target_lakhs = p2.number_input("Weekly Cost Target (₹ Lakhs)", 300.0, 900.0, 560.0, 10.0)
        spoil_target_mt = p3.number_input("Weekly Spoilage Target (MT)", 20.0, 300.0, 85.0, 5.0)
        spot_cap_pct = p4.slider("Max Spot-Market Share (%)", 0, 50, 20, 5) / 100.0

        s1, s2, s3 = st.columns(3)
        p_adverse = s1.slider("Prob: Adverse Weather (0.78x Yield)", 0.0, 0.8, 0.25, 0.05)
        p_normal = s2.slider("Prob: Normal Monsoon (1.00x Yield)", 0.0, 1.0, 0.50, 0.05)
        p_bumper = max(0.0, round(1.0 - p_adverse - p_normal, 2))
        s3.metric("Prob: Bumper Glut (1.24x Yield)", f"{p_bumper:.2f}", "Auto-balanced to 1.00")

    if st.button("🚀 Run Goal Programming + Stochastic MILP Solver", type="primary"):
        if abs(p_adverse + p_normal + p_bumper - 1.0) > 0.01:
            st.error("Scenario probabilities must sum to 1.00. Please adjust Adverse and Normal probabilities.")
        else:
            scenarios = [
                {"id": "S1_Adverse", "prob": p_adverse, "yield": 0.78, "amb_sp_m": 1.35, "ref_sp_m": 1.08, "spot_m": 1.42, "dem_m": 1.03},
                {"id": "S2_Normal",  "prob": p_normal,  "yield": 1.00, "amb_sp_m": 1.00, "ref_sp_m": 1.00, "spot_m": 1.00, "dem_m": 1.00},
                {"id": "S3_Bumper",  "prob": p_bumper,  "yield": 1.24, "amb_sp_m": 1.12, "ref_sp_m": 1.02, "spot_m": 0.76, "dem_m": 0.98},
            ]
            sup_base = {cid: df_ts[col].mean() for cid, _, _, col in CLUSTERS_META}
            dem_base = {h["id"]: df_ts[h["col"]].mean() for h in HUBS_META}
            rt_map = {(r["Origin_ID"], r["Destination_ID"]): r for _, r in df_sample_rt.iterrows()}
            ph_ids = [p["id"] for p in PACKHOUSES_META]
            ph_dict = {p["id"]: p for p in PACKHOUSES_META}
            cl_ids = [c[0] for c in CLUSTERS_META]
            hb_ids = [h["id"] for h in HUBS_META]
            hb_dict = {h["id"]: h for h in HUBS_META}
            modes = ["AMB", "REF"]

            # Build MILP using SciPy HiGHS (fast, deterministic)
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
                    # Flow balance
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
                opened = [ph_dict[ph_ids[i]]["name"] for i in range(len(ph_ids)) if res.x[i] > 0.5]
                exp_cost_l = sum(res.x[idx] * val for idx, val in cost_coeffs.items()) / 100000.0
                exp_spoil_mt = sum(res.x[idx] * val for idx, val in spoil_coeffs.items())
                exp_short_mt = sum(res.x[idx] * val for idx, val in short_coeffs.items())
                tot_dem = sum(dem_base.values())
                fill_rate = (1.0 - exp_short_mt / tot_dem) * 100.0
                spoil_pct = (exp_spoil_mt / tot_dem) * 100.0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Packhouses Opened", f"{len(opened)} / {max_ph}")
                m2.metric("Expected Weekly Cost", f"₹{exp_cost_l:,.2f} L", f"Target: ₹{cost_target_lakhs:.0f} L")
                m3.metric("Expected Spoilage", f"{spoil_pct:.2f}% ({exp_spoil_mt:.1f} MT)", f"-{11.61 - spoil_pct:.2f}% vs NABCONS Avg")
                m4.metric("Expected Fill Rate SLA", f"{fill_rate:.2f}%", f"Shortfall: {exp_short_mt:.1f} MT")

                st.success(f"**Optimal Packhouse Locations Selected:** {', '.join(opened)}")

                # Managerial Recommendations & Alerts
                if exp_short_mt > 50.0:
                    st.markdown(f"<div class='alert-box'><b>⚠️ SLA Capacity Alert:</b> Expected shortfall is {exp_short_mt:.1f} MT/week. Consider increasing Max Packhouses or relaxing the spot-market cap.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='rec-box'><b>✅ Managerial Recommendation:</b> Opening {len(opened)} strategic packhouses achieves a {fill_rate:.1f}% fill rate while cutting post-harvest spoilage from the 11.61% NABCONS ambient baseline down to {spoil_pct:.2f}%.</div>", unsafe_allow_html=True)

                # Packhouse Utilization Table (Normal Scenario)
                ph_rows = []
                for j in ph_ids:
                    is_open = res.x[idx_y[j]] > 0.5
                    flow_n = sum(res.x[idx_x[(i, j, m, "S2_Normal")]] for i in cl_ids for m in modes) if p_normal > 0 else 0.0
                    ph_rows.append({
                        "Packhouse_ID": j,
                        "Facility_Name": ph_dict[j]["name"],
                        "Status": "OPEN" if is_open else "CLOSED",
                        "Capacity_MT_Wk": ph_dict[j]["cap"],
                        "Normal_Scenario_Flow_MT": round(flow_n, 1),
                        "Utilization_Pct": round((flow_n / ph_dict[j]["cap"]) * 100, 1) if is_open else 0.0,
                    })
                df_ph_out = pd.DataFrame(ph_rows)
                st.dataframe(df_ph_out, use_container_width=True)
                fig_util = px.bar(df_ph_out[df_ph_out["Status"] == "OPEN"], x="Facility_Name", y="Utilization_Pct", title="Opened Packhouse Capacity Utilization (% - Normal Scenario)", color_discrete_sequence=["#1B4D3E"])
                st.plotly_chart(fig_util, use_container_width=True)

                st.download_button("Download Optimization Results (CSV)", df_ph_out.to_csv(index=False).encode("utf-8"), "optimal_packhouse_decisions.csv", "text/csv")
            else:
                st.error(f"Optimization failed: {res.message}")

# -------------------------------------------------------------------------
# PAGE 4: DATA UPLOAD, VALIDATION & TEMPLATES
# -------------------------------------------------------------------------
elif page == "4. Data Upload & Templates":
    st.subheader("Data Input Management, Validation & Blank Template")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.download_button(
            "📥 Download Blank Input Template (CSV)",
            pd.DataFrame(columns=df_sample_ts.columns).to_csv(index=False).encode("utf-8"),
            "blank_input_template.csv",
            "text/csv",
        )
    with col_u2:
        if st.button("🔄 Reset to Built-In 104-Week Sample Dataset"):
            st.session_state["active_ts"] = df_sample_ts.copy()
            st.session_state["data_source_label"] = "Built-in Sample Dataset (104 Weeks)"
            st.success("Reset to built-in 104-week sample dataset.")

    uploaded = st.file_uploader("Upload Custom Weekly Time-Series CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            req_cols = set(df_sample_ts.columns)
            missing_cols = req_cols - set(df_up.columns)
            if missing_cols:
                st.error(f"Schema Validation Error: Missing required columns: {sorted(missing_cols)}")
            elif len(df_up) < 12:
                st.error("Range Validation Error: Dataset must contain at least 12 weekly rows for analysis.")
            elif (df_up["Total_Cluster_Supply_MT"] < 0).any() or (df_up["Total_Hub_Demand_MT"] < 0).any():
                st.error("Value Validation Error: Supply and Demand quantities must be non-negative.")
            else:
                st.session_state["active_ts"] = df_up
                st.session_state["data_source_label"] = f"Custom Upload ({uploaded.name}, {len(df_up)} rows)"
                st.success(f"Validated and loaded {len(df_up)} rows from {uploaded.name}!")
        except Exception as e:
            st.error(f"File Parsing Error: {e}")

    st.dataframe(st.session_state["active_ts"].head(15), use_container_width=True)

# -------------------------------------------------------------------------
# PAGE 5: CONTEXT, PROBLEM FRAMING & TECH OUTLOOK (CLO 1, 2, 6)
# -------------------------------------------------------------------------
elif page == "5. Context, Strategy & Tech Outlook (CLO 1,2,6)":
    st.subheader("Supply Chain Context, Problem Framing & Technology Outlook")
    tab_c1, tab_c2, tab_c6 = st.tabs(["CLO 1: Context & Strategy", "CLO 2: Problem Framing", "CLO 6: Technology Outlook"])

    with tab_c1:
        st.markdown("""
        #### Supply Chain Context & Strategic Fit
        - **Key Players:** Smallholder farmers & FPOs (8 horticulture clusters across Nashik, Pune, Satara, Sangli, Kolhapur, Belagavi, Dharwad, Kolar), Integrated Packhouses/Pre-Cooling Hubs, Reefer/Ambient Fleet Operators, and 4 Urban Demand Hubs (Mumbai, Pune, Goa, Bengaluru).
        - **Flows:** *Material* (fresh produce from farmgate to urban DCs), *Information* (weekly Agmarknet arrivals, weather alerts, DC fill-rate SLAs), and *Money* (farmgate procurement payouts, packhouse CapEx/OpEx, wholesale revenue, shortfall penalties).
        - **Strategic Fit:** Competes on **Responsive, Low-Wastage Fulfilment** (Chopra's Strategic Fit). Investing in pre-cooling packhouses and reefer dispatch trades higher fixed/freight cost for a ~9% reduction in post-harvest spoilage (from 11.61% NABCONS ambient baseline to <2.5%).
        """)
        st.markdown("#### Systems View (Causal Loop Diagram) & Business Architecture")
        st.markdown("""
        - **Reinforcing Spoilage Loop:** Higher Ambient Temp → Faster Respiration/Decay → Higher Transit Spoilage → Lower Net Supply → Higher Spot Prices & SLA Penalties.
        - **Balancing Cold-Chain Loop:** Packhouse Pre-Cooling + Reefer Allocation → Lower Field Heat (42°C → 4°C) → Extended Shelf Life → Higher Fill Rate & Margin Stability.
        - **Business Architecture:** Historical Mandi/Weather Data + Forecast Engine (SARIMA/Seasonal-AR) → Goal Programming & Stochastic MILP Engine (PuLP/HiGHS) → Manager Decision Dashboard (Facility Open/Close & Reefer Routing).
        """)

    with tab_c2:
        st.markdown("""
        #### Quantitative Problem Framing (CLO 2)
        - **Decision Variables:** Binary facility location $y_j \in \{0,1\}$ (open packhouse $j \in \{1..6\}$); continuous first-mile flows $x_{ijms} \ge 0$ and line-haul flows $z_{jkms} \ge 0$ across modes $m \in \{\text{AMB}, \text{REF}\}$ and weather scenarios $s \in \{\text{Adverse}, \text{Normal}, \text{Bumper}\}$; spot purchases $u_{ks}$ and shortfall $v_{ks}$; goal deviations $d_1^+, d_2^+, d_3^+$.
        - **Objective Function (Weighted Goal Programming):** Minimize normalized deviations $\frac{w_1}{T_{\text{cost}}} d_1^+ + \frac{w_2}{T_{\text{spoil}}} d_2^+ + \frac{w_3}{T_{\text{short}}} d_3^+$ across expected weekly cost, spoilage MT, and SLA shortfall.
        - **Constraints:** Cluster supply limits, packhouse capacity & linking ($\sum x \le C_j y_j$), spoilage-adjusted flow conservation at packhouses, demand satisfaction at hubs, max packhouses budget cap, and spot-market traceability cap.
        """)

    with tab_c6:
        st.markdown("""
        #### Technology Outlook (~1 Page Equivalent, CLO 6)
        1. **IoT Real-Time Cold-Chain Telemetry & Ethylene/Temperature Sensors:**
           - *Benefits:* Continuous temperature/humidity logging inside reefer trucks and solar cold rooms enables dynamic shelf-life estimation and Dynamic FEFO (First-Expired-First-Out) rerouting.
           - *Limitations & Challenges:* Rural cellular dead-zones along state highways, sensor calibration drift, and upfront hardware cost for smallholder FPOs.
           - *Ethical Issues:* Data ownership disputes (who owns farmgate quality logs—the aggregator or the FPO?) and algorithmic price-penalization of farmers for transit delays outside their control.
        2. **Agentic AI & Digital Twin Supply Network Simulation:**
           - *Benefits:* A live Digital Twin mirrors packhouse queues and highway congestion, while Agentic AI monitors IMD heatwave alerts, automatically triggers re-forecasts, and drafts reefer dispatch changes for manager approval.
           - *Limitations & Challenges:* High integration effort across fragmented mandi/ERP systems and risk of hallucinated or overconfident automated dispatch during unprecedented climate shocks.
           - *Ethical Issues:* Human-in-the-loop accountability (who bears liability if an autonomous agent misroutes a perishable load?) and transparency of automated vendor/FPO selection.
        """)

# -------------------------------------------------------------------------
# PAGE 6: USER GUIDE & LIMITATIONS
# -------------------------------------------------------------------------
else:
    st.subheader("User Guide, Assumptions & Model Limitations")
    st.markdown("""
    #### How to Use This App
    1. **Explore Historical Trends (Tab 1):** Filter by agricultural season (`Rabi`, `Zaid`, `Kharif`) to inspect supply/demand and ambient vs. cold-chain spoilage.
    2. **Run Forecasts (Tab 2):** Choose a series, adjust the forecast horizon (4–16 weeks), inspect out-of-sample MAPE/RMSE, and download the forecast CSV.
    3. **Optimize the Network (Tab 3):** Set your Goal Programming weights (Cost, Spoilage, SLA), CapEx limit (Max Packhouses), and weather probabilities, then click **Run Optimizer** to get actionable packhouse & utilization recommendations.
    4. **Test Custom Data (Tab 4):** Download the blank CSV template, enter your own weekly numbers, and upload it with automatic schema/range validation.

    #### Key Assumptions & Limitations
    - **Disclosed Synthetic Data:** The 104-week dataset is synthetically generated (`seed=42`) and calibrated to NABCONS (2022) tomato loss benchmarks (11.61%) and NCCD cold-chain parameters.
    - **Static Linear Freight & Spoilage:** The MILP uses deterministic arc-level spoilage percentages per scenario rather than continuous nonlinear Arrhenius thermal decay.
    - **Weekly Aggregation:** Decisions are modeled at a weekly cadence; intra-week daily dispatch scheduling is outside the current model scope.
    """)