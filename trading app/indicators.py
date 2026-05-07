import numpy as np # type: ignore
import pandas as pd
import logging
from typing import List, Optional
from models import (RSIResult, MACDResult, BollingerResult, EMAResult,
                    ATRResult, VWAPResult, IndicatorBundle, SignalStrength)
from cache import cache_get, cache_set
import data as data_svc

logger = logging.getLogger(__name__)

_SCORE = {
    SignalStrength.STRONG_BUY: +1.0, SignalStrength.BUY: +0.5,
    SignalStrength.NEUTRAL:     0.0,
    SignalStrength.SELL:       -0.5, SignalStrength.STRONG_SELL: -1.0,
}


def _lst(s: pd.Series) -> List[Optional[float]]:
    return [None if pd.isna(v) else round(float(v), 6) for v in s]


def _r(v, d=4):
    return round(float(v), d)


def _sig(score: float) -> SignalStrength:
    if score >= 0.6:  return SignalStrength.STRONG_BUY
    if score >= 0.2:  return SignalStrength.BUY
    if score <= -0.6: return SignalStrength.STRONG_SELL
    if score <= -0.2: return SignalStrength.SELL
    return SignalStrength.NEUTRAL


def rsi(df: pd.DataFrame, period=14) -> RSIResult:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    vals  = 100 - (100 / (1 + rs))
    lat   = _r(vals.iloc[-1], 2)
    if lat >= 70:    sig = SignalStrength.STRONG_SELL
    elif lat >= 60:  sig = SignalStrength.BUY
    elif lat <= 30:  sig = SignalStrength.STRONG_BUY
    elif lat <= 40:  sig = SignalStrength.SELL
    else:            sig = SignalStrength.NEUTRAL
    return RSIResult(values=_lst(vals), latest=lat, signal=sig,
                     overbought=lat > 70, oversold=lat < 30)


def macd(df: pd.DataFrame, fast=12, slow=26, sig_p=9) -> MACDResult:
    c    = df["close"]
    ml   = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sl   = ml.ewm(span=sig_p, adjust=False).mean()
    hist = ml - sl
    lm, ls, lh = _r(ml.iloc[-1]), _r(sl.iloc[-1]), _r(hist.iloc[-1])
    cross = None
    if len(hist) >= 2:
        if hist.iloc[-2] < 0 and lh >= 0:  cross = "bullish"
        elif hist.iloc[-2] > 0 and lh <= 0: cross = "bearish"
    if cross == "bullish" or (lm > 0 and lh > 0):
        sig = SignalStrength.STRONG_BUY if cross else SignalStrength.BUY
    elif cross == "bearish" or (lm < 0 and lh < 0):
        sig = SignalStrength.STRONG_SELL if cross else SignalStrength.SELL
    else:
        sig = SignalStrength.NEUTRAL
    return MACDResult(macd_line=_lst(ml), signal_line=_lst(sl), histogram=_lst(hist),
                      latest_macd=lm, latest_signal=ls, latest_histogram=lh,
                      signal=sig, crossover=cross)


def bollinger(df: pd.DataFrame, period=20, std=2.0) -> BollingerResult:
    c   = df["close"]
    mid = c.rolling(period).mean()
    s   = c.rolling(period).std(ddof=0)
    up  = mid + std * s
    lo  = mid - std * s
    lu, lm, ll, lc = _r(up.iloc[-1]), _r(mid.iloc[-1]), _r(lo.iloc[-1]), _r(c.iloc[-1])
    bw  = _r((lu - ll) / lm, 4) if lm else 0.0
    pb  = _r((lc - ll) / (lu - ll), 4) if (lu - ll) else 0.5
    sig = (SignalStrength.STRONG_SELL if pb > 1.0 else
           SignalStrength.SELL        if pb > 0.8 else
           SignalStrength.STRONG_BUY  if pb < 0.0 else
           SignalStrength.BUY         if pb < 0.2 else
           SignalStrength.NEUTRAL)
    return BollingerResult(upper=_lst(up), middle=_lst(mid), lower=_lst(lo),
                           latest_upper=lu, latest_middle=lm, latest_lower=ll,
                           latest_close=lc, bandwidth=bw, percent_b=pb, signal=sig)


def ema(df: pd.DataFrame, period=20) -> EMAResult:
    c  = df["close"]
    e  = c.ewm(span=period, adjust=False).mean()
    return EMAResult(period=period, values=_lst(e), latest=_r(e.iloc[-1]),
                     price_above=_r(c.iloc[-1]) > _r(e.iloc[-1]))


def atr(df: pd.DataFrame, period=14) -> ATRResult:
    h, l, c = df["high"], df["low"], df["close"]
    tr  = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    a   = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    lat = _r(a.iloc[-1]); lc = _r(c.iloc[-1])
    return ATRResult(period=period, values=_lst(a), latest=lat,
                     atr_pct=_r((lat/lc)*100, 2) if lc else 0.0)


def vwap(df: pd.DataFrame) -> VWAPResult:
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    vol  = df["volume"].replace(0, np.nan)
    vw   = (tp * vol).cumsum() / vol.cumsum()
    lat  = _r(vw.iloc[-1]); lc = _r(df["close"].iloc[-1])
    return VWAPResult(values=_lst(vw), latest=lat, price_above=lc > lat)


def compute_all(df: pd.DataFrame, symbol: str, timeframe: str) -> IndicatorBundle:
    if len(df) < 26:
        raise ValueError(f"Need at least 26 candles, got {len(df)}")
    df = df.copy(); df.columns = [c.lower() for c in df.columns]
    r   = rsi(df); m = macd(df); b = bollinger(df)
    e20 = ema(df, 20); e50 = ema(df, 50); a = atr(df); v = vwap(df)
    sigs = [r.signal, m.signal, b.signal,
            SignalStrength.BUY  if e20.price_above else SignalStrength.SELL,
            SignalStrength.BUY  if e50.price_above else SignalStrength.SELL,
            SignalStrength.BUY  if v.price_above   else SignalStrength.SELL]
    score = round(sum(_SCORE[s] for s in sigs) / len(sigs), 3)
    bull  = sum(1 for s in sigs if s in (SignalStrength.BUY, SignalStrength.STRONG_BUY))
    bear  = sum(1 for s in sigs if s in (SignalStrength.SELL, SignalStrength.STRONG_SELL))
    return IndicatorBundle(symbol=symbol, timeframe=timeframe, candle_count=len(df),
                           rsi=r, macd=m, bollinger=b, ema_20=e20, ema_50=e50, atr=a, vwap=v,
                           composite_signal=_sig(score), composite_score=score,
                           bullish_count=bull, bearish_count=bear, neutral_count=len(sigs)-bull-bear)


async def get_indicators(symbol: str, timeframe, days=180) -> IndicatorBundle:
    from models import Timeframe
    key    = f"ind:{symbol}:{timeframe.value}"
    cached = await cache_get(key)
    if cached:
        return IndicatorBundle(**cached)
    market = await data_svc.get_historical(symbol, timeframe, days)
    if not market.candles:
        raise ValueError(f"No candle data for {symbol}")
    df = pd.DataFrame([{"open":c.open,"high":c.high,"low":c.low,
                         "close":c.close,"volume":c.volume} for c in market.candles])
    bundle = compute_all(df, symbol.upper(), timeframe.value)
    await cache_set(key, bundle.model_dump(), ttl=60)
    return bundle
