import numpy as np

def bisection(f, a, b, tol=1e-6, max_iter=100):
    steps = []
    if f(a) * f(b) > 0:
        return None, steps, "⚠️ لا يوجد تغيير إشارة بين a و b"

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        error = abs(b - a) / 2
        steps.append((i, a, b, c, fc, error))

        if error < tol or abs(fc) < tol:
            return c, steps, None

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return c, steps, "⚠️ لم يتم الوصول إلى الجذر بالدقة المطلوبة"


def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    steps = []
    x = x0

    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            return None, steps, "⚠️ المشتقة ≈ 0، الطريقة فشلت"

        x_new = x - fx / dfx
        error = abs(x_new - x)
        steps.append((i, x, fx, dfx, x_new, error))

        if error < tol or abs(fx) < tol:
            return x_new, steps, None

        x = x_new

    return x, steps, "⚠️ لم يتم الوصول إلى الجذر بالدقة المطلوبة"


def secant(f, x0, x1, tol=1e-6, max_iter=100):
    steps = []

    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)

        if abs(f1 - f0) < 1e-12:
            return None, steps, "⚠️ المقام ≈ 0، الطريقة فشلت"

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)
        steps.append((i, x0, x1, f0, f1, x2, error))

        if error < tol or abs(f(x2)) < tol:
            return x2, steps, None

        x0, x1 = x1, x2

    return x2, steps, "⚠️ لم يتم الوصول إلى الجذر بالدقة المطلوبة"
