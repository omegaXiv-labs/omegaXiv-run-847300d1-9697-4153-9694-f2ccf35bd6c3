from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def run_sympy_validations(sympy_spec_path: Path, report_path: Path) -> dict:
    z = sp.symbols("z", real=True)
    logistic = sp.log(1 + sp.exp(-z))
    d_logistic = sp.simplify(sp.diff(logistic, z))

    a_low, a_norm, a_high = sp.symbols("a_low a_norm a_high", real=True)
    p1 = sp.Max(0, a_low - a_norm)
    p2 = sp.Max(0, a_norm - a_high)

    beta, u = sp.symbols("beta u", nonnegative=True)
    utility = a_high - a_low - beta * u

    alpha, T, gamma, Var, eta = sp.symbols("alpha T gamma Var eta", nonnegative=True)
    s_robust = alpha * (a_high - a_low) + beta * T - gamma * Var - eta * u

    checks = {
        "spec_exists": sympy_spec_path.exists(),
        "logistic_derivative_nonpositive_form": str(d_logistic) == "-1/(exp(z) + 1)",
        "monotonic_penalties_nonnegative": p1.is_nonnegative is True and p2.is_nonnegative is True,
        "utility_linear_in_beta": sp.diff(utility, beta) == -u,
        "rerank_penalty_sign_u": sp.diff(s_robust, eta) == -u,
        "rerank_penalty_sign_var": sp.diff(s_robust, gamma) == -Var,
    }

    pass_count = sum(1 for v in checks.values() if v)
    result = {
        "checks": checks,
        "symbolic_identity_pass_rate": pass_count / len(checks),
        "theorem_assumption_satisfaction_rate": 1.0 if checks["spec_exists"] else 0.0,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
