from __future__ import annotations


def bland_altman_stats(a, b):
    np = __import__("numpy")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    return {"bias": float(diff.mean()), "loa_low": float(diff.mean() - 1.96 * diff.std()), "loa_high": float(diff.mean() + 1.96 * diff.std())}
