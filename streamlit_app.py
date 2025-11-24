# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -------------------------- Config --------------------------
st.set_page_config(page_title="Distribution Fitter Pro", layout="wide")
st.title("Distribution Fitter Pro")
st.markdown("### Fit 12 statistical distributions – auto or manual")

# -------------------------- Distributions --------------------------
distributions = {
    "Normal": stats.norm,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Log-Normal": stats.lognorm,
    "Beta": stats.beta,
    "Chi-Squared": stats.chi2,
    "Uniform": stats.uniform,
    "Cauchy": stats.cauchy,
    "Laplace": stats.laplace,
    "Gumbel Right": stats.gumbel_r,
    "Pareto": stats.pareto,
}

# -------------------------- Sidebar: Data Input --------------------------
st.sidebar.header("Data Input")
option = st.sidebar.radio("How to input data?", ["Type / paste", "Upload CSV"])

data = None

if option == "Type / paste":
    text = st.sidebar.text_area(
        "Enter numbers (comma, space or newline)",
        "5, 8, 12, 15, 18, 20, 23, 25, 28, 30, 35, 40, 50, 60, 80",
        height=120
    )
    try:
        data = np.fromstring(text.replace(',', ' '), sep=' ')
        data = data[~np.isnan(data)]
    except:
        st.sidebar.error("Please enter valid numbers")

else:  # Upload CSV
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        col = st.sidebar.selectbox("Select column", df.columns)
        data = pd.to_numeric(df[col], errors='coerce').dropna().values

if data is None or len(data) < 5:
    st.info("Please provide at least 5 data points")
    st.stop()

data = data[np.isfinite(data)]
st.sidebar.success(f"Loaded {len(data)} points")

# -------------------------- Tabs --------------------------
tab1, tab2 = st.tabs(["Auto Fit", "Manual Fit"])

# ============================= AUTO FIT =============================
with tab1:
    st.header("Automatic Fitting")
    choices = st.multiselect("Select distributions", list(distributions.keys()),
                             default=["Normal", "Gamma", "Weibull", "Log-Normal"])

    results = []
    best_ks = 999
    best_name = best_params = best_dist = None

    prog = st.progress(0)
    for i, name in enumerate(choices):
        d = distributions[name]
        try:
            if name in ["Beta", "Pareto"]:
                p = d.fit(data, floc=0 if np.all(data > 0) else None)
            else:
                p = d.fit(data)
            ks, _ = stats.kstest(data, d.cdf, args=p)
            results.append((name, p, ks, d))
            if ks < best_ks:
                best_ks, best_name, best_params, best_dist = ks, name, p, d
        except:
            results.append((name, None, 999, d))
        prog.progress((i + 1) / len(choices))

    # Table
    table = []
    for name, p, ks, _ in results:
        ps = ", ".join([f"{x:.4f}" for x in p]) if p else "Failed"
        table.append({"Distribution": name, "Parameters": ps, "KS-stat": f"{ks:.5f}" if ks < 999 else "Failed"})
    st.dataframe(pd.DataFrame(table), use_container_width=True)

    if best_name:
        st.success(f"Best fit: **{best_name}** (KS = {best_ks:.5f})")

        fig, ax = plt.subplots()
        ax.hist(data, bins=30, density=True, alpha=0.7, color="#3498db", label="Data")
        x = np.linspace(data.min(), data.max(), 1000)
        ax.plot(x, best_dist.pdf(x, *best_params), 'r-', lw=3, label=best_name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

# ============================= MANUAL FIT =============================
with tab2:
    st.header("Manual Parameter Tuning")
    name = st.selectbox("Distribution", list(distributions.keys()))
    d = distributions[name]

    try:
        default = d.fit(data)
    except:
        default = (1,) * d.numargs + (data.mean(), data.std())

    shapes = d.shapes.split(", ") if d.shapes else []
    labels = [s.strip() for s in shapes] + ["loc", "scale"]

    params = []
    for i, lab in enumerate(labels):
        if lab == "loc":
            val = st.slider(lab, float(data.min()-data.std()), float(data.max()+data.std()), float(default[i]))
        elif lab == "scale":
            val = st.slider(lab, 0.01, float(10*data.std()), float(default[i]))
        else:
            val = st.slider(lab, 0.01, 50.0, float(default[i]))
        params.append(val)

    fig2, ax2 = plt.subplots()
    ax2.hist(data, bins=30, density=True, alpha=0.7, color="#9b59b6", label="Data")
    x2 = np.linspace(data.min(), data.max(), 1000)
    try:
        ax2.plot(x2, d.pdf(x2, *params), 'g-', lw=3, label=f"{name} (manual)")
    except:
        st.error("Invalid parameters for this distribution")
    else:
        ax2.legend()
        st.pyplot(fig2)


st.caption("Built for your class project – good luck!")

