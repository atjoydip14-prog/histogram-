import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
import warnings
warnings.filterwarnings("ignore")

# -------------------------- Page Config --------------------------
st.set_page_config(page_title="Distribution Fitter Pro", layout="wide")
st.title("🔍 Distribution Fitter Pro")
st.markdown("### Fit 12+ statistical distributions to your data — automatically or manually!")

# -------------------------- Available Distributions --------------------------
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
    "Gumbel (Right)": stats.gumbel_r,
    "Pareto": stats.pareto,
}

# -------------------------- Sidebar: Data Input --------------------------
st.sidebar.header("📊 Data Input")
data_option = st.sidebar.radio("Input Method:", ["Manual Entry", "Upload CSV"])

data = None
if data_option == "Manual Entry":
    manual_input = st.sidebar.text_area(
        "Enter values (comma/space/newline separated):", 
        value="10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 45, 50, 60, 70, 100",
        height=100
    )
    try:
        data = np.fromstring(manual_input.replace(',', ' '), sep=' ').astype(float)
        data = data[~np.isnan(data)]  # Clean early
        if len(data) < 5:
            st.sidebar.warning("⚠️ Need at least 5 points!")
            data = None
    except ValueError:
        st.sidebar.error("❌ Invalid numbers only!")

elif data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose CSV:", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] == 1:
                data = df.iloc[:, 0].dropna().values
            else:
                col = st.sidebar.selectbox("Select Column:", df.columns)
                data = df[col].dropna().values
            data = data.astype(float)
            data = data[~np.isnan(data)]
        except Exception as e:
            st.sidebar.error(f"❌ CSV Error: {e}")

if data is None or len(data) < 5:
    st.info("👆 Please enter/upload valid data (5+ numbers) in the sidebar to start!")
    st.stop()

# Final clean
data = data[np.isfinite(data)]
if len(data) == 0:
    st.error("❌ No valid data after cleaning!")
    st.stop()

# Data summary
col1, col2, col3, col4 = st.sidebar.columns(4)
col1.metric("Points", len(data))
col2.metric("Mean", f"{np.mean(data):.2f}")
col3.metric("Std Dev", f"{np.std(data):.2f}")
col4.metric("Min/Max", f"{np.min(data):.1f} / {np.max(data):.1f}")

# -------------------------- Main Tabs --------------------------
tab1, tab2 = st.tabs(["🤖 Auto-Fit (Recommended)", "🎛️ Manual Tuning"])

# ===================================================================
# TAB 1: Auto Fitting
# ===================================================================
with tab1:
    st.header("Automatic Fitting")
    selected_dists = st.multiselect(
        "Select Distributions (multi-select OK):",
        options=list(distributions.keys()),
        default=["Normal", "Exponential", "Gamma", "Weibull", "Log-Normal"]
    )

    if not selected_dists:
        st.info("Select at least one distribution!")
        st.stop()

    results = []
    best_ks = np.inf
    best_name, best_params, best_dist = None, None, None

    progress = st.progress(0)
    for i, name in enumerate(selected_dists):
        dist = distributions[name]
        try:
            # Robust fitting with floc/fscale for better convergence
            if name in ["Beta", "Pareto"]:
                params = dist.fit(data, floc=0)  # Fix loc=0 for positive data
            else:
                params = dist.fit(data)
            
            # KS test (adjusted for loc/scale)
            if len(params) >= 2:
                scaled_data = (data - params[-2]) / params[-1]
                ks_stat, _ = stats.kstest(scaled_data, dist.cdf, args=params[:-2])
            else:
                ks_stat, _ = stats.kstest(data, dist.cdf, args=params)
            
            results.append((name, params, ks_stat, dist))
            if ks_stat < best_ks:
                best_ks = ks_stat
                best_name, best_params, best_dist = name, params, dist
        except Exception:
            results.append((name, None, np.inf, dist))
        
        progress.progress((i + 1) / len(selected_dists))

    # Results Table
    st.subheader("Fit Results")
    table_data = []
    for name, params, ks, _ in results:
        param_str = ", ".join([f"{p:.4f}" for p in params]) if params is not None else "Failed"
        status = "🏆 Best!" if name == best_name else ""
        table_data.append({"Distribution": name, "Parameters": param_str, "KS Stat": f"{ks:.4f}" if ks < np.inf else "Failed", "Status": status})
    
    df_results = pd.DataFrame(table_data)
    st.dataframe(df_results.style.apply(lambda x: ['background-color: #d4edda' if 'Best' in str(val) else '' for val in x], axis=1))

    if best_name:
        st.success(f"🏆 Best Fit: **{best_name}** (KS = {best_ks:.4f})")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        hist_n, bins, _ = ax.hist(data, bins='auto', density=True, alpha=0.6, color='skyblue', label='Data', edgecolor='black')
        x_fit = np.linspace(min(bins), max(bins), 1000)
        y_fit = best_dist.pdf(x_fit, *best_params)
        ax.plot(x_fit, y_fit, 'r-', linewidth=3, label=f'{best_name} Fit')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Metrics
        st.subheader("📈 Fit Quality")
        # Better error calc: compare to histogram densities
        bin_centers = (bins[:-1] + bins[1:]) / 2
        hist_density = np.histogram(data, bins=bins, density=True)[0]
        fit_at_centers = best_dist.pdf(bin_centers, *best_params)
        mae = np.mean(np.abs(hist_density - fit_at_centers))
        rmse = np.sqrt(np.mean((hist_density - fit_at_centers)**2))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.4f}")
        c2.metric("RMSE", f"{rmse:.4f}")
        c3.metric("KS Stat", f"{best_ks:.4f}")

# ===================================================================
# TAB 2: Manual Fitting
# ===================================================================
with tab2:
    st.header("Manual Parameter Tuning")
    manual_dist_name = st.selectbox("Choose Distribution:", options=list(distributions.keys()))
    dist = distributions[manual_dist_name]

    # Get default params
    try:
        default_params = dist.fit(data)
    except:
        default_params = [1.0] * (dist.numargs + 2)  # shape(s) + loc + scale fallback
        default_params[-2] = np.min(data) - 1
        default_params[-1] = np.std(data)

    # Dynamic sliders based on dist
    st.subheader("Adjust Parameters:")
    params = []
    param_labels = dist.shapes.split(',') + ['loc', 'scale'] if dist.shapes else ['loc', 'scale']
    
    for i, label in enumerate(param_labels):
        min_val = 0.01
        max_val = 100.0
        step = 0.01
        default = default_params[i] if i < len(default_params) else 1.0
        
        if label == 'loc':
            min_val = data.min() - data.std()
            max_val = data.max() + data.std()
            step = (max_val - min_val) / 100
        elif label == 'scale':
            min_val = 0.01
            max_val = 10 * np.std(data)
            step = (max_val - min_val) / 100
        
        val = st.slider(f"{label.capitalize()}", min_val, max_val, default, step=step)
        params.append(val)

    # Plot manual fit
    col_left, col_right = st.columns([3, 1])
    with col_left:
        fig_man, ax_man = plt.subplots(figsize=(10, 6))
        ax_man.hist(data, bins='auto', density=True, alpha=0.6, color='lightcoral', label='Data')
        x_man = np.linspace(data.min(), data.max(), 1000)
        try:
            y_man = dist.pdf(x_man, *params)
            ax_man.plot(x_man, y_man, 'g-', linewidth=3, label=f'{manual_dist_name} (Manual)')
            ax_man.set_title(f'Manual Fit: {manual_dist_name}')
            ax_man.set_xlabel('Value')
            ax_man.set_ylabel('Density')
            ax_man.legend()
            ax_man.grid(True, alpha=0.3)
            st.pyplot(fig_man)
        except Exception as e:
            st.error(f"⚠️ Plot Error: {e} (Try adjusting params)")

    with col_right:
        st.subheader("Current Params")
        for lbl, val in zip(param_labels, params):
            st.write(f"**{lbl}**: {val:.4f}")

# Footer
st.markdown("---")
st.markdown("*Powered by Streamlit & SciPy • Ready for your class submission!*")