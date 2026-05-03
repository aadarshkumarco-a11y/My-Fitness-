"""Black-Scholes pricing for European options on non-dividend-paying assets.

Used as a fallback when scraped option-chain data is incomplete or stale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class BlackScholesResult:
    call_price: float
    put_price: float
    delta_call: float
    delta_put: float
    gamma: float
    vega: float
    theta_call: float
    theta_put: float


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def black_scholes(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    volatility: float,
    risk_free_rate: float = 0.065,
) -> BlackScholesResult:
    """Return call/put prices and the standard greeks.

    All inputs use natural units: ``time_to_expiry_years`` in years (e.g. 7
    days = 7/365), ``volatility`` and ``risk_free_rate`` as decimals.
    Degenerate inputs return intrinsic values gracefully.
    """
    S, K, T, sigma, r = spot, strike, time_to_expiry_years, volatility, risk_free_rate

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        call = max(0.0, S - K)
        put = max(0.0, K - S)
        logger.debug(
            "Black-Scholes degenerate inputs (S=%s K=%s T=%s sigma=%s) — using intrinsic",
            S, K, T, sigma,
        )
        return BlackScholesResult(
            call_price=call, put_price=put,
            delta_call=1.0 if S > K else 0.0,
            delta_put=-1.0 if S < K else 0.0,
            gamma=0.0, vega=0.0, theta_call=0.0, theta_put=0.0,
        )

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    sqrt_T = math.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1.0
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100.0  # per 1% vol move
    theta_call = (
        -(S * pdf_d1 * sigma) / (2 * sqrt_T)
        - r * K * math.exp(-r * T) * norm.cdf(d2)
    ) / 365.0
    theta_put = (
        -(S * pdf_d1 * sigma) / (2 * sqrt_T)
        + r * K * math.exp(-r * T) * norm.cdf(-d2)
    ) / 365.0

    return BlackScholesResult(
        call_price=max(call, 0.0),
        put_price=max(put, 0.0),
        delta_call=delta_call,
        delta_put=delta_put,
        gamma=gamma,
        vega=vega,
        theta_call=theta_call,
        theta_put=theta_put,
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    is_call: bool = True,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Solve for implied volatility via bisection. Returns 0.0 if no solution."""
    if market_price <= 0 or time_to_expiry_years <= 0:
        return 0.0
    low, high = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        res = black_scholes(spot, strike, time_to_expiry_years, mid, risk_free_rate)
        price = res.call_price if is_call else res.put_price
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            high = mid
        else:
            low = mid
        if high - low < 1e-7:
            break
    return 0.5 * (low + high)
