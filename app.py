# app.py  (النسخة المدمجة)
import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# الدوال الأساسية من utils (تأكد أن ملف utils.py موجود في نفس المجلد)
from utils import (
    parse_function,
    solve_bisection,
    solve_newton,
    solve_secant,
    plot_function,
    plot_convergence,
)

# ---------------- page config ----------------
st.set_page_config(page_title="Numerical Analysis Project", page_icon="🔢", layout="wide")

# ---------------- helpers ----------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_from_file(image_file):
    """If the file exists, set it as background via base64-embedded CSS."""
    if not os.path.exists(image_file):
        return  # nothing to do
    try:
        img_b64 = get_base64_of_bin_file(image_file)
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{img_b64}");
            background-size: cover;
            background-position: center;
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: rgba(0,0,0,0);
        }}
        .landing-overlay {{
            background: rgba(255, 255, 255, 0.78);
            padding: 40px;
            border-radius: 16px;
            text-align: center;
            margin: 60px auto;
            max-width: 900px;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except Exception as e:
        # fail silently but show in console for debugging
        print("Background load error:", e)

# ---------------- global CSS (applies to entire app) ----------------
st.markdown(
    """
    <style>
      /* RTL inputs + alignment tweaks for better Arabic layout */
      .stTextInput>div>div>input {text-align: right;}
      .stNumberInput>div>div>input {text-align: right;}
      .right-align {text-align: right;}
      .dataframe tbody td { text-align: right; }
      /* make matplotlib canvases use full width container */
      .stPlotlyChart, .stImage, .stCanvas { max-width: 100% !important; }
      /* tighten header spacing */
      header ~ div { padding-top: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- session state for page navigation & defaults ----------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

# Keep other session state keys used by main UI to persist between navigations
defaults = {
    "ready": False,
    "params": {},
    "errors": [],
    "eq": "x**3 - 2*x - 1",
    "method_select": "جميع الطرق",
    "a_str": "",
    "b_str": "",
    "x0_newton_str": "",
    "x0_sec_str": "",
    "x1_sec_str": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go_to(page_name: str):
    st.session_state.page = page_name

# ---------------- Landing Page ----------------
if st.session_state.page == "landing":
    # attempt to set background if background.jpg exists in same folder
    set_background_from_file("background.jpg")  # optional: put background.jpg in same folder

    st.markdown(
        """
        <div style="text-align: center; padding: 50px;">
            <h1 style="font-size: 2em; color: #2E86C1;">Numerical Analysis Project</h1>
            <hr>
            <h3 style="color: #555;">حل المعادلات غير الخطية باستخدام طرق التحليل العددي</h3>
            <p style="font-size: 1.1em; color: #444; margin-top: 20px;">
                هذا المشروع يهدف إلى استكشاف طرق مختلفة لإيجاد حلول تقريبية 
                للمعادلات غير الخطية مثل نيوتن-رافسون والطرق العددية الأخرى.
            </p>
            <hr>
            <h6 style="font-size: 2em; color: #269091;">أعداد : رياض عبدالله الشاوش</h6>
            <h6 style="font-size: 2em; color: #269091;">اشراف الدكتور : صالح الصباري</h6>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.button("جرب النظام", use_container_width=True):
        go_to("home")

# ---------------- Main App Page ----------------
elif st.session_state.page == "home":
# ---------- Header + description مع التحكم بالحجم ----------
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="font-size: 3em; font-weight: bold; color: #2C850;">
                برنامج حل المعادلات غير الخطية
            </h1>
            <h2 style="font-size: 2em; color: #555;">
                Bisection · Newton · Secant
            </h2>
            <p style="font-size: 1.2em; color: #333; margin-top: 10px;">
                في هذه الواجهة يمكنك إدخال المعادلة، اختيار الطريقة، وإظهار خطوات الحل والرسوم البيانية.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # layout: main plot area (left) and controls (right)
    left_col, right_col = st.columns([2, 1])

    # ---------- validator callback ----------
    def validate_inputs():
        """Validate current session_state widget values and set session_state['ready']"""
                # Back to landing

        ready = True
        errs = []
        params = {}

        eq = st.session_state.get("eq", "").strip()
        if not eq:
            ready = False

        method = st.session_state.get("method_select", "جميع الطرق")

        # constants
        TOL = 1e-6
        MAX_ITER_BISECTION = 100
        MAX_ITER_NEWTON = 50
        MAX_ITER_SECANT = 50
        if st.button("العودة لواجهة الترحيب"):
            go_to("landing")

        # Bisection
        if method in ["التنصيف", "جميع الطرق"]:
            a_s = st.session_state.get("a_str", "")
            b_s = st.session_state.get("b_str", "")
            if not a_s or not b_s:
                ready = False
            else:
                try:
                    a = float(a_s); b = float(b_s)
                    if a >= b:
                        ready = False
                        errs.append("⚠️ For bisection: a must be < b.")
                    else:
                        params['bisection'] = {'a': a, 'b': b, 'tol': TOL, 'max_iter': MAX_ITER_BISECTION}
                except Exception:
                    ready = False
                    errs.append("⚠️ Bisection: please enter valid numeric values for a and b.")

        # Newton
        if method in ["نيوتن–رافسون", "جميع الطرق"]:
            x0n_s = st.session_state.get("x0_newton_str", "")
            if not x0n_s:
                ready = False
            else:
                try:
                    x0n = float(x0n_s)
                    params['newton'] = {'x0': x0n, 'tol': TOL, 'max_iter': MAX_ITER_NEWTON}
                except Exception:
                    ready = False
                    errs.append("⚠️ Newton: please enter a valid numeric x0.")

        # Secant
        if method in ["القاطع", "جميع الطرق"]:
            x0s_s = st.session_state.get("x0_sec_str", "")
            x1s_s = st.session_state.get("x1_sec_str", "")
            if not x0s_s or not x1s_s:
                ready = False
            else:
                try:
                    x0s = float(x0s_s); x1s = float(x1s_s)
                    if abs(x1s - x0s) < 1e-15:
                        ready = False
                        errs.append("⚠️ Secant: x0 and x1 must be distinct.")
                    else:
                        params['secant'] = {'x0': x0s, 'x1': x1s, 'tol': TOL, 'max_iter': MAX_ITER_SECANT}
                except Exception:
                    ready = False
                    errs.append("⚠️ Secant: please enter valid numeric x0 and x1.")

        st.session_state["ready"] = ready
        st.session_state["params"] = params
        st.session_state["errors"] = errs

    # ---------- Controls (right column) ----------
    with right_col:
        st.header("Controls")

        # equation input (default example)
        equation = st.text_input(
            "📌 Enter equation (in x):",
            key="eq",
            value=st.session_state.get("eq", "x**3 - 2*x - 1"),
            placeholder="example: x**3 - 2*x - 1",
            on_change=validate_inputs
        )

        # method selector
        method = st.selectbox(
            "Choose method:",
            ["جميع الطرق", "التنصيف", "نيوتن–رافسون", "القاطع"],
            index=0,
            key="method_select",
            on_change=validate_inputs
        )

        st.markdown("**Method parameters** — placeholders vanish on typing")

        # Bisection fields (placeholders)
        if method in ["التنصيف", "جميع الطرق"]:
            st.subheader("Bisection")
            a_str = st.text_input("a (lower bound)", placeholder="Enter a", key="a_str", on_change=validate_inputs)
            b_str = st.text_input("b (upper bound)", placeholder="Enter b", key="b_str", on_change=validate_inputs)

        # Newton fields
        if method in ["نيوتن–رافسون", "جميع الطرق"]:
            st.subheader("Newton–Raphson")
            x0_newton_str = st.text_input("x0 (initial guess)", placeholder="Enter x0", key="x0_newton_str", on_change=validate_inputs)

        # Secant fields
        if method in ["القاطع", "جميع الطرق"]:
            st.subheader("Secant")
            x0_sec_str = st.text_input("x0 (first guess)", placeholder="Enter x0", key="x0_sec_str", on_change=validate_inputs)
            x1_sec_str = st.text_input("x1 (second guess)", placeholder="Enter x1", key="x1_sec_str", on_change=validate_inputs)

        # display small hints/errors
        if st.session_state.get("errors"):
            for e in st.session_state["errors"]:
                st.warning(e)
        else:
            if not st.session_state.get("ready", False):
                st.info("Fill required fields to enable Solve button.")

        # Smart Solve button (enabled/disabled by session_state['ready'])
        solve_btn = st.button("🔍 Solve", disabled=not st.session_state.get("ready", False))


        # Placeholder container for results (will show below button)
        results_container = st.container()  # results + solution steps + plots will render here

    # ---------- Outputs: results & plots now rendered inside results_container (under inputs/button) ----------
    if solve_btn:
        params = st.session_state.get("params", {})
        f, df, expr, parse_err = parse_function(st.session_state.get("eq", ""))
        with results_container:
            if parse_err:
                st.error(parse_err)
            else:
                # compute each selected method
                results = {}
                run_errors = []

                if "bisection" in params:
                    try:
                        r = solve_bisection(f, **params["bisection"])
                        results["Bisection"] = r
                    except Exception as e:
                        run_errors.append(f"Bisection: {e}")

                if "newton" in params:
                    try:
                        r = solve_newton(f, df, **params["newton"])
                        results["Newton–Raphson"] = r
                    except Exception as e:
                        run_errors.append(f"Newton–Raphson: {e}")

                if "secant" in params:
                    try:
                        r = solve_secant(f, **params["secant"])
                        results["Secant"] = r
                    except Exception as e:
                        run_errors.append(f"Secant: {e}")

                # show runtime errors (if any)
                if run_errors:
                    for e in run_errors:
                        st.error(e)

                # show final results under inputs and button
                if results:
                    st.subheader("Final Results")
                    for name, info in results.items():
                        root_val = info.get("root", None)
                        if root_val is None:
                            st.markdown(f"<div class='right-align'>{name}: root not found (error)</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='right-align'>{name}: approximate root = {root_val:.8f}</div>", unsafe_allow_html=True)
                else:
                    st.info("No results computed. Check errors above.")

                # Tabs for solution steps (display under results)
                if results:
                    st.subheader("Solution Steps")
                    labels = list(results.keys())
                    if labels:
                        tabs = st.tabs(labels)
                        for tab_obj, label in zip(tabs, labels):
                            with tab_obj:
                                iterations = results[label].get("iterations", [])
                                # normalize to DataFrame
                                def iterations_to_df(iters):
                                    if not iters:
                                        return pd.DataFrame()
                                    first = iters[0]
                                    if isinstance(first, dict):
                                        return pd.DataFrame(iters)
                                    if isinstance(first, (list, tuple)):
                                        max_len = max(len(x) if isinstance(x,(list,tuple)) else 1 for x in iters)
                                        if max_len == 2:
                                            return pd.DataFrame(iters, columns=["iteration", "error"])
                                        return pd.DataFrame(iters)
                                    return pd.DataFrame(iters)
                                df_iter = iterations_to_df(iterations)
                                if df_iter.empty:
                                    st.write("No iteration data to display.")
                                else:
                                    st.dataframe(df_iter, width="stretch")

                # ---- Plots: now show directly under solution steps (inside results_container) ----
                if results:
                    st.subheader("Plots")
                    plot_labels = list(results.keys()) + ["Convergence", "Comparison"]
                    p_tabs = st.tabs(plot_labels)
                    for ptab, label in zip(p_tabs, plot_labels):
                        with ptab:
                            if label in results:
                                rv = results[label].get("root", None)
                                if rv is None:
                                    st.write("No root to plot for this method.")
                                else:
                                    fig = plot_function(f, root=rv, method_name=label)
                                    # style for dark theme and readable English labels
                                    ax = fig.axes[0]
                                    ax.set_xlabel("x")
                                    ax.set_ylabel("f(x)")
                                    ax.set_title(f"{label} - f(x)")
                                    fig.patch.set_facecolor("#2b2b2b")
                                    ax.set_facecolor("#2b2b2b")
                                    ax.tick_params(colors="white", which="both")
                                    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
                                    ax.title.set_color("white")
                                    for spine in ax.spines.values():
                                        spine.set_color("white")
                                    legend = ax.get_legend()
                                    if legend:
                                        frame = legend.get_frame()
                                        frame.set_facecolor("#3a3a3a")
                                        for text in legend.get_texts():
                                            text.set_color("white")
                                    st.pyplot(fig)
                            elif label == "Convergence":
                                conv = {}
                                for m, info in results.items():
                                    iters = info.get("iterations", [])
                                    errs = []
                                    for idx, step in enumerate(iters):
                                        if isinstance(step, dict) and "error" in step and step["error"] is not None:
                                            errs.append((step.get("iteration", idx+1), step["error"]))
                                        elif isinstance(step, (list,tuple)) and len(step) >= 2:
                                            errs.append((step[0], step[1]))
                                    conv[m] = errs
                                if any(conv.values()):
                                    fig = plot_convergence(conv)
                                    ax = fig.axes[0]
                                    fig.patch.set_facecolor("#2b2b2b")
                                    ax.set_facecolor("#2b2b2b")
                                    ax.tick_params(colors="white", which="both")
                                    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
                                    ax.title.set_color("white")
                                    for spine in ax.spines.values():
                                        spine.set_color("white")
                                    legend = ax.get_legend()
                                    if legend:
                                        for text in legend.get_texts(): text.set_color("white")
                                    st.pyplot(fig)
                                else:
                                    st.write("Not enough convergence data to plot.")
                            elif label == "Comparison":
                                fig, ax = plt.subplots(figsize=(8,4))
                                x_vals = np.linspace(-10,10,800)
                                try:
                                    y_vals = f(x_vals)
                                except Exception:
                                    y_vals = np.array([f(x) for x in x_vals])
                                ax.plot(x_vals, y_vals, label="f(x)")
                                ax.axhline(0, linestyle="--", color="black", linewidth=1)
                                colors = ["r","g","m","c"]
                                for (m, info), col in zip(results.items(), colors):
                                    rv = info.get("root", None)
                                    if rv is not None:
                                        ax.scatter(rv, f(rv), color=col, s=80, label=f"{m}: {rv:.6f}")
                                ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.set_title("Comparison of Roots")
                                ax.legend(); ax.grid(True)
                                fig.patch.set_facecolor('#2b2b2b'); ax.set_facecolor('#2b2b2b')
                                ax.tick_params(colors="white", which="both")
                                ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
                                ax.title.set_color("white")
                                for spine in ax.spines.values(): spine.set_color("white")
                                st.pyplot(fig)
                else:
                    st.info("No results computed.")

# ---------------- end of app.py ----------------
