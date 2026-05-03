from nse_backtester.data_engine import DataEngine
from nse_backtester.database import Database


def test_synthetic_ohlcv_shape():
    df = DataEngine.synthetic_ohlcv(days=50, seed=1)
    assert len(df) > 0
    assert {"open", "high", "low", "close", "volume", "timestamp"}.issubset(df.columns)
    assert (df["close"] > 0).all()


def test_parse_option_chain_handles_missing_fields():
    de = DataEngine.__new__(DataEngine)
    payload = {
        "records": {
            "underlyingValue": 17000,
            "expiryDates": ["29-May-2024"],
            "data": [
                {"strikePrice": 17000, "expiryDate": "29-May-2024",
                 "CE": {"lastPrice": 200, "openInterest": 1000, "impliedVolatility": 18},
                 "PE": {"lastPrice": "-", "openInterest": None, "impliedVolatility": ""}},
            ],
        }
    }
    snap = de._parse_option_chain("NIFTY", payload)
    assert snap.symbol == "NIFTY"
    assert snap.spot == 17000
    assert snap.atm_strike == 17000
    assert len(snap.rows) == 2  # CE + PE both attempted
    pe_row = [r for r in snap.rows if r["option_type"] == "PE"][0]
    assert pe_row["ltp"] is None
    assert pe_row["iv"] is None


def test_database_round_trip(tmp_path):
    db = Database(path=tmp_path / "t.db")
    rows = [
        {"timestamp": "2024-01-01T00:00:00Z", "symbol": "NIFTY", "expiry": "29-May",
         "strike": 18000, "option_type": "CE", "ltp": 100, "oi": 500, "iv": 15, "spot": 18000}
    ]
    assert db.insert_option_chain_rows(rows) == 1
    df = db.fetch_option_chain("NIFTY")
    assert len(df) == 1
