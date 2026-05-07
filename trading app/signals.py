import logging
from models import (FusedSignal, SignalAction, RiskProfile, RiskLevel,
                    IndicatorVote, SignalExplanation, SignalStrength,
                    AssetType, ASSET_MAP, Timeframe, IndicatorBundle)
from cache import cache_get, cache_set
import data as data_svc
import indicators as ind_svc

logger = logging.getLogger(__name__)

_SCORE = {
    SignalStrength.STRONG_BUY: +1.0, SignalStrength.BUY: +0.5,
    SignalStrength.NEUTRAL:     0.0,
    SignalStrength.SELL:       -0.5, SignalStrength.STRONG_SELL: -1.0,
}

_ATR_MULT   = {"crypto":(1.8,3.6), "stock":(1.5,2.5), "index":(1.2,2.0), "forex":(1.0,1.8)}
_BLEND      = {"crypto":(0.4,0.6), "stock":(0.5,0.5), "index":(0.55,0.45), "forex":(0.6,0.4)}
_ASSET_TYPE = {AssetType.CRYPTO:"crypto", AssetType.STOCK:"stock",
               AssetType.INDEX:"index",   AssetType.FOREX:"forex"}


def _action(score: float, rr: float) -> SignalAction:
    if rr < 1.4 and abs(score) < 0.8:
        score *= 0.7
    if score >= 0.55:  return SignalAction.STRONG_BUY
    if score >= 0.25:  return SignalAction.BUY
    if score <= -0.55: return SignalAction.STRONG_SELL
    if score <= -0.25: return SignalAction.SELL
    return SignalAction.HOLD


def _build_risk(entry: float, direction: str, atr_abs: float,
                atr_pct: float, asset_type: str, confidence: float) -> RiskProfile:
    sl_m, tp_m = _ATR_MULT.get(asset_type, (1.5, 2.5))
    if direction == "buy":
        sl = round(entry - sl_m * atr_abs, 6)
        tp = round(entry + tp_m * atr_abs, 6)
    elif direction == "sell":
        sl = round(entry + sl_m * atr_abs, 6)
        tp = round(entry - tp_m * atr_abs, 6)
    else:
        sl = round(entry - atr_abs, 6)
        tp = round(entry + atr_abs, 6)

    sl_d   = abs(entry - sl)
    tp_d   = abs(entry - tp)
    rr     = round(tp_d / sl_d, 2) if sl_d else 0.0
    sl_pct = round((sl_d / entry) * 100, 3) if entry else 0.0
    tp_pct = round((tp_d / entry) * 100, 3) if entry else 0.0
    kelly  = max(0.0, (confidence * rr - (1 - confidence)) / rr) * 0.25 * 100 if rr else 1.0
    pos    = min(round(kelly, 2), 5.0)

    thresholds = {"crypto":(2,4,7),"stock":(1,2,4),"index":(0.5,1.2,2.5),"forex":(0.3,0.7,1.5)}
    lo, med, hi = thresholds.get(asset_type, (1, 2.5, 5))
    rl = (RiskLevel.LOW    if atr_pct < lo  else
          RiskLevel.MEDIUM if atr_pct < med else
          RiskLevel.HIGH   if atr_pct < hi  else RiskLevel.EXTREME)

    return RiskProfile(entry_price=round(entry,6), stop_loss=sl, take_profit=tp,
                       risk_reward=rr, stop_loss_pct=sl_pct, take_profit_pct=tp_pct,
                       suggested_position_pct=pos, max_loss_per_trade_pct=round(pos*sl_pct/100,3),
                       atr_abs=round(atr_abs,6), atr_pct=round(atr_pct,3), risk_level=rl)


def _fuse(bundle: IndicatorBundle, live_price: float, asset_type: str) -> FusedSignal:
    ema20_sig = SignalStrength.BUY  if bundle.ema_20.price_above else SignalStrength.SELL
    ema50_sig = SignalStrength.BUY  if bundle.ema_50.price_above else SignalStrength.SELL
    vwap_sig  = SignalStrength.BUY  if bundle.vwap.price_above   else SignalStrength.SELL

    components = [
        ("RSI",       bundle.rsi.signal,       0.20, f"RSI {bundle.rsi.latest:.1f}"),
        ("MACD",      bundle.macd.signal,      0.20, f"Hist {bundle.macd.latest_histogram:+.4f}"),
        ("Bollinger", bundle.bollinger.signal, 0.15, f"B% {bundle.bollinger.percent_b:.2f}"),
        ("EMA 20",    ema20_sig,               0.15, f"{'above' if bundle.ema_20.price_above else 'below'} {bundle.ema_20.latest:.2f}"),
        ("EMA 50",    ema50_sig,               0.15, f"{'above' if bundle.ema_50.price_above else 'below'} {bundle.ema_50.latest:.2f}"),
        ("VWAP",      vwap_sig,                0.15, f"{'above' if bundle.vwap.price_above else 'below'} {bundle.vwap.latest:.2f}"),
    ]

    ind_score = sum(_SCORE[sig] * w for _, sig, w, _ in components)
    if bundle.macd.crossover == "bullish": ind_score += 0.05
    elif bundle.macd.crossover == "bearish": ind_score -= 0.05

    votes = [IndicatorVote(name=n,
                           signal="buy" if _SCORE[s]>0 else "sell" if _SCORE[s]<0 else "neutral",
                           value=v, weight=w)
             for n, s, w, v in components]

    iw, mw   = _BLEND.get(asset_type, (0.5, 0.5))
    ml_score = bundle.composite_score
    final    = round(ind_score * iw + ml_score * mw, 4)

    direction = "buy" if final > 0.1 else "sell" if final < -0.1 else "hold"
    risk      = _build_risk(live_price, direction, bundle.atr.latest,
                            bundle.atr.atr_pct, asset_type, abs(final))
    action    = _action(final, risk.risk_reward)
    conf      = round(min(abs(final) + min((risk.risk_reward - 1) / 3, 0.15), 0.99), 3)

    # Regime
    ema_bull = bundle.ema_20.price_above and bundle.ema_50.price_above
    ema_bear = not bundle.ema_20.price_above and not bundle.ema_50.price_above
    macd_pos = bundle.macd.latest_histogram > 0
    trend    = ("uptrend"   if ema_bull and macd_pos  else
                "downtrend" if ema_bear and not macd_pos else "sideways")
    atr_pct  = bundle.atr.atr_pct
    vol_str  = ("low" if atr_pct < 1 else "medium" if atr_pct < 3 else
                "high" if atr_pct < 6 else "extreme")

    # Explanation
    bull_r, bear_r = [], []
    rsi_v = bundle.rsi.latest
    if rsi_v < 35:   bull_r.append(f"RSI at {rsi_v:.1f} — oversold, bounce likely")
    elif rsi_v > 65: bear_r.append(f"RSI at {rsi_v:.1f} — overbought, watch for reversal")
    elif rsi_v > 55: bull_r.append(f"RSI at {rsi_v:.1f} — bullish momentum building")
    else:            bear_r.append(f"RSI at {rsi_v:.1f} — momentum weakening")

    if bundle.macd.crossover == "bullish":   bull_r.append("MACD bullish crossover just confirmed")
    elif bundle.macd.crossover == "bearish": bear_r.append("MACD bearish crossover just confirmed")
    elif bundle.macd.latest_histogram > 0:   bull_r.append("MACD histogram positive")
    else:                                    bear_r.append("MACD histogram negative")

    if ema_bull:   bull_r.append("Price above both EMA-20 and EMA-50 — trend is up")
    elif ema_bear: bear_r.append("Price below both EMA-20 and EMA-50 — trend is down")

    if bundle.vwap.price_above: bull_r.append("Price above VWAP — buyers in control")
    else:                       bear_r.append("Price below VWAP — sellers in control")

    watch = (f"Extreme volatility — ATR at {atr_pct:.1f}% of price. Size positions carefully."
             if atr_pct > 5 else
             "Monitor volume — a signal without volume confirmation can fail quickly.")

    action_name = action.value.replace("_", " ").title()
    expl = SignalExplanation(
        summary=f"{action_name} signal (score {final:+.2f}) — {len(bull_r)} bullish vs {len(bear_r)} bearish factors.",
        bullish_reasons=bull_r, bearish_reasons=bear_r,
        key_levels={
            "support":    round(bundle.bollinger.latest_lower,  4),
            "resistance": round(bundle.bollinger.latest_upper,  4),
            "ema_20":     round(bundle.ema_20.latest, 4),
            "ema_50":     round(bundle.ema_50.latest, 4),
            "vwap":       round(bundle.vwap.latest,   4),
            "bb_mid":     round(bundle.bollinger.latest_middle, 4),
        },
        watch_out_for=watch,
    )

    vol_bias = ("above_avg" if bundle.bullish_count >= 4 else
                "below_avg" if bundle.bearish_count >= 4 else "normal")

    return FusedSignal(
        symbol=bundle.symbol, asset_type=asset_type, timeframe=bundle.timeframe,
        action=action, confidence=conf, score=final,
        indicator_score=round(ind_score, 4), ml_score=round(ml_score, 4),
        indicator_votes=votes, risk=risk, live_price=round(live_price, 6),
        explanation=expl, trend=trend, volatility=vol_str, volume_bias=vol_bias,
    )


async def get_signal(symbol: str, timeframe: Timeframe = Timeframe.ONE_DAY,
                     force: bool = False) -> FusedSignal:
    sym = symbol.upper(); tf = timeframe.value
    key = f"signal:{sym}:{tf}"

    if not force:
        cached = await cache_get(key)
        if cached:
            return FusedSignal(**cached)

    asset_info = ASSET_MAP.get(sym)
    if not asset_info:
        raise ValueError(f"Unknown symbol: {sym}")

    asset_type = _ASSET_TYPE.get(asset_info.asset_type, "stock")

    import asyncio
    price_obj, bundle = await asyncio.gather(
        data_svc.get_live_price(sym),
        ind_svc.get_indicators(sym, timeframe),
    )

    sig = _fuse(bundle, price_obj.price, asset_type)
    await cache_set(key, sig.model_dump(), ttl=60)
    return sig
