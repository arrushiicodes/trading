from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

from models import (Timeframe, AssetType, AssetInfo, LivePrice, MarketData,
                    IndicatorBundle, FusedSignal)
import data as data_svc
import indicators as ind_svc
import signals as sig_svc


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TradeAI started — open index.html in your browser")
    yield


app = FastAPI(title="TradeAI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/v1/market/assets", response_model=List[AssetInfo])
async def list_assets(asset_type: Optional[AssetType] = None):
    return data_svc.list_assets(asset_type)


@app.get("/api/v1/market/price/{symbol}", response_model=LivePrice)
async def live_price(symbol: str):
    try:
        return await data_svc.get_live_price(symbol.upper())
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"Error: {e}")


@app.get("/api/v1/market/prices", response_model=List[LivePrice])
async def multi_prices(symbols: str = Query(...)):
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return await data_svc.get_multiple_prices(sym_list)


@app.get("/api/v1/market/history/{symbol}", response_model=MarketData)
async def history(
    symbol: str,
    timeframe: Timeframe = Query(Timeframe.ONE_DAY),
    days: int = Query(90, ge=1, le=3650),
):
    try:
        return await data_svc.get_historical(symbol.upper(), timeframe, days)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"Error: {e}")


@app.get("/api/v1/indicators/{symbol}", response_model=IndicatorBundle)
async def indicators(
    symbol: str,
    timeframe: Timeframe = Query(Timeframe.ONE_DAY),
    days: int = Query(180, ge=30),
):
    try:
        return await ind_svc.get_indicators(symbol.upper(), timeframe, days)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"Error: {e}")


@app.get("/api/v1/signal/{symbol}", response_model=FusedSignal)
async def get_signal(
    symbol: str,
    timeframe: Timeframe = Query(Timeframe.ONE_DAY),
    force_refresh: bool = Query(False),
):
    try:
        return await sig_svc.get_signal(symbol.upper(), timeframe, force_refresh)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"Error: {e}")


@app.get("/api/v1/signal/{symbol}/card")
async def signal_card(symbol: str, timeframe: Timeframe = Query(Timeframe.ONE_DAY)):
    try:
        s = await sig_svc.get_signal(symbol.upper(), timeframe)
        return {
            "symbol": s.symbol, "timeframe": s.timeframe,
            "action": s.action, "confidence": s.confidence, "score": s.score,
            "live_price": s.live_price,
            "summary": s.explanation.summary,
            "bullish_reasons": s.explanation.bullish_reasons,
            "bearish_reasons": s.explanation.bearish_reasons,
            "watch_out_for": s.explanation.watch_out_for,
            "entry": s.risk.entry_price,
            "stop_loss": s.risk.stop_loss,
            "take_profit": s.risk.take_profit,
            "risk_reward": s.risk.risk_reward,
            "position_pct": s.risk.suggested_position_pct,
            "risk_level": s.risk.risk_level,
            "trend": s.trend, "volatility": s.volatility,
            "indicator_votes": [v.model_dump() for v in s.indicator_votes],
            "key_levels": s.explanation.key_levels,
            "generated_at": s.generated_at,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/v1/signal/batch/cards")
async def batch_cards(symbols: str = Query(...), timeframe: Timeframe = Query(Timeframe.ONE_DAY)):
    import asyncio
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    results = await asyncio.gather(*[sig_svc.get_signal(s, timeframe) for s in sym_list], return_exceptions=True)
    return [
        {"symbol": s, "action": r.action, "confidence": r.confidence,
         "score": r.score, "live_price": r.live_price,
         "trend": r.trend, "risk_level": r.risk.risk_level}
        if not isinstance(r, Exception)
        else {"symbol": s, "error": str(r)}
        for s, r in zip(sym_list, results)
    ]
