from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timezone
from enum import Enum


class AssetType(str, Enum):
    CRYPTO = "crypto"
    FOREX  = "forex"
    STOCK  = "stock"
    INDEX  = "index"


class Timeframe(str, Enum):
    ONE_MIN     = "1m"
    FIVE_MIN    = "5m"
    FIFTEEN_MIN = "15m"
    ONE_HOUR    = "1h"
    FOUR_HOUR   = "4h"
    ONE_DAY     = "1d"
    ONE_WEEK    = "1wk"
    ONE_MONTH   = "1mo"


class OHLCVCandle(BaseModel):
    timestamp: datetime
    open: float; high: float; low: float; close: float; volume: float


class MarketData(BaseModel):
    symbol: str; asset_type: AssetType; timeframe: Timeframe
    candles: List[OHLCVCandle]; source: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LivePrice(BaseModel):
    symbol: str; asset_type: AssetType; price: float
    change_24h: Optional[float] = None
    change_pct_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AssetInfo(BaseModel):
    symbol: str; name: str; asset_type: AssetType
    exchange: Optional[str] = None
    currency: str = "INR"
    yf_ticker: Optional[str] = None
    binance_symbol: Optional[str] = None


class SignalStrength(str, Enum):
    STRONG_BUY  = "strong_buy"
    BUY         = "buy"
    NEUTRAL     = "neutral"
    SELL        = "sell"
    STRONG_SELL = "strong_sell"


class RSIResult(BaseModel):
    latest: float; signal: SignalStrength; overbought: bool; oversold: bool
    values: List[Optional[float]] = []


class MACDResult(BaseModel):
    latest_macd: float; latest_signal: float; latest_histogram: float
    signal: SignalStrength; crossover: Optional[str] = None
    macd_line: List[Optional[float]] = []
    signal_line: List[Optional[float]] = []
    histogram: List[Optional[float]] = []


class BollingerResult(BaseModel):
    latest_upper: float; latest_middle: float; latest_lower: float; latest_close: float
    bandwidth: float; percent_b: float; signal: SignalStrength
    upper: List[Optional[float]] = []
    middle: List[Optional[float]] = []
    lower: List[Optional[float]] = []


class EMAResult(BaseModel):
    period: int; latest: float; price_above: bool
    values: List[Optional[float]] = []


class ATRResult(BaseModel):
    period: int; latest: float; atr_pct: float
    values: List[Optional[float]] = []


class VWAPResult(BaseModel):
    latest: float; price_above: bool
    values: List[Optional[float]] = []


class IndicatorBundle(BaseModel):
    symbol: str; timeframe: str; candle_count: int
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rsi: RSIResult; macd: MACDResult; bollinger: BollingerResult
    ema_20: EMAResult; ema_50: EMAResult; atr: ATRResult; vwap: VWAPResult
    composite_signal: SignalStrength; composite_score: float
    bullish_count: int; bearish_count: int; neutral_count: int


class RiskLevel(str, Enum):
    LOW = "low"; MEDIUM = "medium"; HIGH = "high"; EXTREME = "extreme"


class RiskProfile(BaseModel):
    entry_price: float; stop_loss: float; take_profit: float; risk_reward: float
    stop_loss_pct: float; take_profit_pct: float
    suggested_position_pct: float; max_loss_per_trade_pct: float
    atr_abs: float; atr_pct: float; risk_level: RiskLevel


class IndicatorVote(BaseModel):
    name: str; signal: str; value: str; weight: float


class SignalExplanation(BaseModel):
    summary: str
    bullish_reasons: List[str]
    bearish_reasons: List[str]
    key_levels: Dict[str, float]
    watch_out_for: str


class SignalAction(str, Enum):
    STRONG_BUY  = "strong_buy"
    BUY         = "buy"
    HOLD        = "hold"
    SELL        = "sell"
    STRONG_SELL = "strong_sell"


class FusedSignal(BaseModel):
    symbol: str; asset_type: str; timeframe: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: SignalAction; confidence: float; score: float
    indicator_score: float; ml_score: float
    indicator_votes: List[IndicatorVote]
    risk: RiskProfile; live_price: float
    explanation: SignalExplanation
    trend: str; volatility: str; volume_bias: str


# ── Asset registry ────────────────────────────────────────────────────────────
KNOWN_ASSETS: List[AssetInfo] = [
    AssetInfo(symbol="BTC",       name="Bitcoin",             asset_type=AssetType.CRYPTO, yf_ticker="BTC-USD",     binance_symbol="BTCUSDT",  currency="USD"),
    AssetInfo(symbol="ETH",       name="Ethereum",            asset_type=AssetType.CRYPTO, yf_ticker="ETH-USD",     binance_symbol="ETHUSDT",  currency="USD"),
    AssetInfo(symbol="SOL",       name="Solana",              asset_type=AssetType.CRYPTO, yf_ticker="SOL-USD",     binance_symbol="SOLUSDT",  currency="USD"),
    AssetInfo(symbol="BNB",       name="BNB",                 asset_type=AssetType.CRYPTO, yf_ticker="BNB-USD",     binance_symbol="BNBUSDT",  currency="USD"),
    AssetInfo(symbol="USDINR",    name="USD/INR",             asset_type=AssetType.FOREX,  yf_ticker="USDINR=X",    currency="INR"),
    AssetInfo(symbol="EURINR",    name="EUR/INR",             asset_type=AssetType.FOREX,  yf_ticker="EURINR=X",    currency="INR"),
    AssetInfo(symbol="EURUSD",    name="EUR/USD",             asset_type=AssetType.FOREX,  yf_ticker="EURUSD=X",    currency="USD"),
    AssetInfo(symbol="NIFTY50",   name="Nifty 50",            asset_type=AssetType.INDEX,  yf_ticker="^NSEI",       exchange="NSE", currency="INR"),
    AssetInfo(symbol="SENSEX",    name="BSE Sensex",          asset_type=AssetType.INDEX,  yf_ticker="^BSESN",      exchange="BSE", currency="INR"),
    AssetInfo(symbol="BANKNIFTY", name="Bank Nifty",          asset_type=AssetType.INDEX,  yf_ticker="^NSEBANK",    exchange="NSE", currency="INR"),
    AssetInfo(symbol="RELIANCE",  name="Reliance Industries", asset_type=AssetType.STOCK,  yf_ticker="RELIANCE.NS", exchange="NSE", currency="INR"),
    AssetInfo(symbol="TCS",       name="Tata Consultancy",    asset_type=AssetType.STOCK,  yf_ticker="TCS.NS",      exchange="NSE", currency="INR"),
    AssetInfo(symbol="HDFCBANK",  name="HDFC Bank",           asset_type=AssetType.STOCK,  yf_ticker="HDFCBANK.NS", exchange="NSE", currency="INR"),
    AssetInfo(symbol="INFY",      name="Infosys",             asset_type=AssetType.STOCK,  yf_ticker="INFY.NS",     exchange="NSE", currency="INR"),
    AssetInfo(symbol="WIPRO",     name="Wipro",               asset_type=AssetType.STOCK,  yf_ticker="WIPRO.NS",    exchange="NSE", currency="INR"),
]
ASSET_MAP = {a.symbol: a for a in KNOWN_ASSETS}
