# utils.py
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) تحليل المعادلة وتحويلها لدالة + مشتقة
# ---------------------------
def parse_function(expr_str):
    x = sp.symbols('x')
    try:
        expr = sp.sympify(expr_str.replace("^", "**"))
        f = sp.lambdify(x, expr, "numpy")
        df = sp.lambdify(x, sp.diff(expr, x), "numpy")
        return f, df, expr, None
    except Exception as e:
        return None, None, None, f"⚠️ خطأ في تحليل المعادلة: {e}"

# ---------------------------
# 2) طريقة التنصيف (Bisection)
# ---------------------------
def solve_bisection(f, a, b, tol=1e-6, max_iter=100):
    fa, fb = f(a), f(b)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("قيم f(a) أو f(b) غير صالحة (NaN).")
    if fa * fb > 0:
        raise ValueError("f(a) و f(b) يجب أن تكونا ذات إشارات مختلفة.")
    
    iterations = []
    for i in range(1, max_iter+1):
        c = (a + b)/2
        fc = f(c)
        error = abs(b - a)/2
        iterations.append((i, error))
        if abs(fc) < tol or error < tol:
            return {"root": c, "iterations": iterations}
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return {"root": c, "iterations": iterations}

# ---------------------------
# 3) طريقة نيوتن-رافسون (Newton-Raphson)
# ---------------------------
def solve_newton(f, df, x0, tol=1e-6, max_iter=50):
    x = x0
    iterations = []
    for i in range(1, max_iter+1):
        fx, dfx = f(x), df(x)
        if np.isnan(fx) or np.isnan(dfx) or dfx == 0:
            raise ValueError(f"المشتقة صفرية أو غير معرفة عند x = {x:.6f}")
        x_new = x - fx / dfx
        error = abs(x_new - x)
        iterations.append((i, error))
        if error < tol:
            return {"root": x_new, "iterations": iterations}
        x = x_new
    return {"root": x, "iterations": iterations}

# ---------------------------
# 4) طريقة القاطع (Secant)
# ---------------------------
def solve_secant(f, x0, x1, tol=1e-6, max_iter=50):
    iterations = []
    for i in range(1, max_iter+1):
        f0, f1 = f(x0), f(x1)
        if np.isnan(f0) or np.isnan(f1):
            raise ValueError(f"f(x) غير معرفة عند إحدى النقاط.")
        if f1 - f0 == 0:
            raise ValueError("فرق f(x1)-f(x0) صفر. جرب نقاط أخرى.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)
        iterations.append((i, error))
        if error < tol:
            return {"root": x2, "iterations": iterations}
        x0, x1 = x1, x2
    return {"root": x2, "iterations": iterations}

# ---------------------------
# 5) رسم الدالة مع الجذر (Plot Function)
# ---------------------------
def plot_function(f, root=None, method_name=""):
    x_vals = np.linspace(-10, 10, 400)
    y_vals = f(x_vals)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_vals, y_vals, label="f(x)", color="blue")
    ax.axhline(0, color="black", linewidth=1)

    if root is not None:
        ax.scatter(root, f(root), color="red", zorder=5,
                   label=f"Root ({method_name})")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Graph of f(x) {method_name}")
    ax.legend()
    ax.grid(True)

    return fig

# ---------------------------
# 6) رسم تحليل التقارب (Error vs Iteration)
# ---------------------------
def plot_convergence(results_dict):
    """
    results_dict = {
        "Bisection": [(iter, error), ...],
        "Newton": [(iter, error), ...],
        "Secant": [(iter, error), ...]
    }
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for method, steps in results_dict.items():
        if not steps:
            continue
        iterations = [s[0] for s in steps]
        errors = [s[1] for s in steps]
        ax.semilogy(iterations, errors, marker="o", label=method)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error (log scale)")
    ax.set_title("Convergence Analysis")
    ax.legend()
    ax.grid(True)

    return fig
