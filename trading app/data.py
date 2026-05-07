import yfinance as yf
import httpx
import pandas as pd
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List
from models import (MarketData, LivePrice, OHLCVCandle, Timeframe,
                    AssetType, AssetInfo, ASSET_MAP)
from cache import cache_get, cache_set

logger = logging.getLogger(__name__)

YF_INTERVAL = {
    Timeframe.ONE_MIN:"1m", Timeframe.FIVE_MIN:"5m", Timeframe.FIFTEEN_MIN:"15m",
    Timeframe.ONE_HOUR:"1h", Timeframe.FOUR_HOUR:"1h", Timeframe.ONE_DAY:"1d",
    Timeframe.ONE_WEEK:"1wk", Timeframe.ONE_MONTH:"1mo",
}
BN_INTERVAL = {
    Timeframe.ONE_MIN:"1m", Timeframe.FIVE_MIN:"5m", Timeframe.FIFTEEN_MIN:"15m",
    Timeframe.ONE_HOUR:"1h", Timeframe.FOUR_HOUR:"4h", Timeframe.ONE_DAY:"1d",
    Timeframe.ONE_WEEK:"1w", Timeframe.ONE_MONTH:"1M",
}
BINANCE = "https://api.binance.com"


def _f(val) -> Optional[float]:
    try:
        v = float(val)
        return None if pd.isna(v) else v
    except:
        return None


def _resolve(symbol: str) -> AssetInfo:
    asset = ASSET_MAP.get(symbol.upper())
    if not asset:
        raise ValueError(f"Unknown symbol '{symbol}'. Supported: {', '.join(ASSET_MAP.keys())}")
    return asset


async def fetch_history_yf(asset: AssetInfo, timeframe: Timeframe, days: int) -> MarketData:
    end   = datetime.utcnow()
    start = end - timedelta(days=min(days, 3650))
    ticker = yf.Ticker(asset.yf_ticker)
    df = ticker.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=YF_INTERVAL[timeframe],
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No data from Yahoo Finance for {asset.symbol}")
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    candles = [
        OHLCVCandle(timestamp=idx.to_pydatetime(),
                    open=_f(r["Open"]) or 0, high=_f(r["High"]) or 0,
                    low=_f(r["Low"]) or 0,  close=_f(r["Close"]) or 0,
                    volume=_f(r["Volume"]) or 0)
        for idx, r in df.iterrows()
    ]
    return MarketData(symbol=asset.symbol, asset_type=asset.asset_type,
                      timeframe=timeframe, candles=candles, source="yfinance")


async def fetch_history_binance(asset: AssetInfo, timeframe: Timeframe, limit: int) -> MarketData:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{BINANCE}/api/v3/klines",
                             params={"symbol": asset.binance_symbol,
                                     "interval": BN_INTERVAL[timeframe],
                                     "limit": min(limit, 1000)})
        r.raise_for_status()
        raw = r.json()
    candles = [
        OHLCVCandle(timestamp=datetime.utcfromtimestamp(k[0]/1000),
                    open=float(k[1]), high=float(k[2]),
                    low=float(k[3]),  close=float(k[4]), volume=float(k[5]))
        for k in raw
    ]
    return MarketData(symbol=asset.symbol, asset_type=asset.asset_type,
                      timeframe=timeframe, candles=candles, source="binance")


async def fetch_price_yf(asset: AssetInfo) -> LivePrice:
    info   = yf.Ticker(asset.yf_ticker).fast_info
    price  = _f(getattr(info, "last_price", None))
    prev   = _f(getattr(info, "previous_close", None))
    change = (price - prev) if price and prev else None
    chgpct = (change / prev * 100) if change and prev else None
    return LivePrice(symbol=asset.symbol, asset_type=asset.asset_type,
                     price=price or 0, change_24h=change, change_pct_24h=chgpct)


async def fetch_price_binance(asset: AssetInfo) -> LivePrice:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{BINANCE}/api/v3/ticker/24hr",
                             params={"symbol": asset.binance_symbol})
        r.raise_for_status()
        d = r.json()
    return LivePrice(symbol=asset.symbol, asset_type=asset.asset_type,
                     price=float(d["lastPrice"]),
                     change_24h=float(d["priceChange"]),
                     change_pct_24h=float(d["priceChangePercent"]),
                     volume_24h=float(d["volume"]),
                     high_24h=float(d["highPrice"]),
                     low_24h=float(d["lowPrice"]))


# ── Public interface ──────────────────────────────────────────────────────────

async def get_historical(symbol: str, timeframe: Timeframe = Timeframe.ONE_DAY, days: int = 90) -> MarketData:
    asset = _resolve(symbol)
    key   = f"hist:{symbol}:{timeframe.value}:{days}"
    hit   = await cache_get(key)
    if hit:
        return MarketData(**hit)
    if asset.asset_type == AssetType.CRYPTO and asset.binance_symbol:
        data = await fetch_history_binance(asset, timeframe, limit=min(days * 24, 1000))
    else:
        data = await fetch_history_yf(asset, timeframe, days)
    await cache_set(key, data.model_dump(), ttl=3600)
    return data


async def get_live_price(symbol: str) -> LivePrice:
    asset = _resolve(symbol)
    key   = f"live:{symbol}"
    hit   = await cache_get(key)
    if hit:
        return LivePrice(**hit)
    if asset.asset_type == AssetType.CRYPTO and asset.binance_symbol:
        price = await fetch_price_binance(asset)
    else:
        price = await fetch_price_yf(asset)
    await cache_set(key, price.model_dump(), ttl=30)
    return price


async def get_multiple_prices(symbols: List[str]) -> List[LivePrice]:
    results = await asyncio.gather(*[get_live_price(s) for s in symbols], return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]


def list_assets(asset_type=None) -> List[AssetInfo]:
    assets = list(ASSET_MAP.values())
    if asset_type:
        assets = [a for a in assets if a.asset_type == asset_type]
    return assets
