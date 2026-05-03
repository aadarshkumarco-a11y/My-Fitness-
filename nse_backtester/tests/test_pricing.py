import math

from nse_backtester.pricing import black_scholes, implied_volatility


def test_atm_call_put_parity():
    spot, strike, t, vol, r = 18000, 18000, 7 / 365, 0.18, 0.065
    res = black_scholes(spot, strike, t, vol, r)
    parity = res.call_price - res.put_price
    expected = spot - strike * math.exp(-r * t)
    assert abs(parity - expected) < 1.0


def test_degenerate_inputs_return_intrinsic():
    res = black_scholes(spot=100, strike=80, time_to_expiry_years=0, volatility=0.2)
    assert res.call_price == 20
    assert res.put_price == 0


def test_implied_volatility_round_trip():
    target_vol = 0.25
    spot, strike, t, r = 18000, 18000, 30 / 365, 0.065
    res = black_scholes(spot, strike, t, target_vol, r)
    iv = implied_volatility(res.call_price, spot, strike, t, r, is_call=True)
    assert abs(iv - target_vol) < 0.01
