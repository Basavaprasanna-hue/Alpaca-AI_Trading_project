# Live_options_entry_monitor.py

# Research/validation only. This script does NOT submit, cancel, modify,
# or close orders. It:
# 1) reads the later-model EOD DQN JSON intent;
# 2) resolves fresh active option contracts from Alpaca;
# 3) subscribes to their live option quotes;
# 4) applies DTE, price, quote-size, freshness, and spread-cost gates;
# 5) writes a ranked live_eligible_contracts.csv for review.

from __future__ import annotations
import msgpack
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import websockets
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest

API_KEY = ""
API_SECRET = ""

OPTION_FEED = os.getenv("ALPACA_OPTION_FEED", "indicative").lower()
PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

SCRIPT_DIR = Path(__file__).resolve().parent
DIRECTION_INTENTS_JSON = SCRIPT_DIR / "direction_intents" / "latest_direction_intents.json"
OUTPUT_CSV = SCRIPT_DIR / "order_intents" / "live_eligible_contracts.csv"
AUDIT_CSV = SCRIPT_DIR / "order_intents" / "live_contract_shortlist.csv"

MIN_DTE = 0
MAX_DTE = 180
MAX_ONE_CONTRACT_DEBIT = 500.00
MAX_ROUND_TRIP_SPREAD_COST = 0.05
MIN_QUOTE_SIZE = 1
MAX_QUOTE_AGE_SECONDS = 60
MAX_CONTRACTS_PER_SIGNAL = 5
MAX_DTE_DISTANCE_FROM_REFERENCE = 7
MAX_MONEYNESS_DISTANCE_FROM_REFERENCE = 0.10


@dataclass
class LiveQuote:
    option_symbol: str
    timestamp: Any
    bid: float
    ask: float
    bid_size: int
    ask_size: int

    @property
    def midpoint(self) -> float | None:
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def round_trip_spread_cost(self) -> float | None:
        if self.ask <= 0 or self.bid < 0 or self.ask < self.bid:
            return None
        return (self.ask - self.bid) / self.ask


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}. Run EOD inference first."
        )


def require_credentials() -> None:
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "Missing Alpaca credentials. Set API_KEY and API_SECRET in this script "
            "or set ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )
    if API_KEY == API_SECRET:
        raise RuntimeError("API_KEY and API_SECRET must be different values.")


def as_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None

    try:
        if hasattr(value, "seconds") and hasattr(value, "nanoseconds"):
            return pd.Timestamp(
                value.seconds * 1_000_000_000 + value.nanoseconds,
                unit="ns",
                tz="UTC",
            )

        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            return None

        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")

        return timestamp.tz_convert("UTC")

    except (TypeError, ValueError, OverflowError):
        return None


def normalize_contract_type(value: Any) -> str:
    value = str(value).lower()
    if "call" in value:
        return "call"
    if "put" in value:
        return "put"
    return value


def load_direction_signals() -> pd.DataFrame:
    require_file(DIRECTION_INTENTS_JSON, "Direction-intent JSON")

    with DIRECTION_INTENTS_JSON.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    rows: list[dict[str, Any]] = []
    for result in payload.get("results", []):
        decision = str(result.get("decision", "")).strip().upper()
        direction = str(result.get("direction", "")).strip().upper()
        if decision != "ENTER" or direction not in {"LONG_CALL", "LONG_PUT"}:
            continue

        constraints = result.get("live_selector_constraints") or {}
        candidate = result.get("selected_candidate") or {}
        ticker = str(result.get("ticker", "")).strip().upper()
        contract_type = normalize_contract_type(
            constraints.get("contract_type", candidate.get("contract_type", ""))
        )
        reference_dte = constraints.get("reference_dte", candidate.get("reference_dte"))
        reference_moneyness = constraints.get(
            "reference_moneyness", candidate.get("reference_moneyness")
        )

        if not ticker:
            raise ValueError(f"ENTER signal missing ticker: {result}")
        if contract_type not in {"call", "put"}:
            raise ValueError(f"{ticker}: invalid contract type {contract_type!r}")
        if reference_dte is None or reference_moneyness is None:
            raise ValueError(f"{ticker}: missing reference DTE or moneyness")

        rows.append(
            {
                "ticker": ticker,
                "direction": direction,
                "contract_type": contract_type,
                "reference_dte": float(reference_dte),
                "reference_moneyness": float(reference_moneyness),
                "max_round_trip_spread_cost": float(
                    constraints.get(
                        "max_round_trip_spread_cost", MAX_ROUND_TRIP_SPREAD_COST
                    )
                ),
                "quantity": int(constraints.get("quantity", 1)),
                "historical_option_symbol": candidate.get("historical_option_symbol"),
                "reference_strike_price": candidate.get("strike_price"),
                "reference_expiration_date": candidate.get("expiration_date"),
                "hard_flatten_deadline_ist": result.get("hard_flatten_deadline_ist"),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "direction",
            "contract_type",
            "reference_dte",
            "reference_moneyness",
            "max_round_trip_spread_cost",
            "quantity",
            "historical_option_symbol",
            "reference_strike_price",
            "reference_expiration_date",
            "hard_flatten_deadline_ist",
        ],
    )


def get_underlying_quotes(
    stock_client: StockHistoricalDataClient, tickers: list[str]
) -> dict[str, float]:
    if not tickers:
        return {}

    response = stock_client.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=tickers)
    )
    prices: dict[str, float] = {}
    for ticker in tickers:
        quote = response.get(ticker)
        if quote is None:
            continue
        bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
        ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
        if bid > 0 and ask > 0 and ask >= bid:
            prices[ticker] = (bid + ask) / 2.0
        elif ask > 0:
            prices[ticker] = ask
        elif bid > 0:
            prices[ticker] = bid
    return prices


def fetch_active_contracts(
    trading_client: TradingClient,
    ticker: str,
    contract_type: str,
    start_expiration: pd.Timestamp,
    end_expiration: pd.Timestamp,
) -> list[Any]:
    requested_type = ContractType.CALL if contract_type == "call" else ContractType.PUT
    request = GetOptionContractsRequest(
        underlying_symbols=[ticker],
        status=AssetStatus.ACTIVE,
        type=requested_type,
        expiration_date_gte=start_expiration.date(),
        expiration_date_lte=end_expiration.date(),
        limit=1000,
    )
    response = trading_client.get_option_contracts(request)
    return list(response.option_contracts)


def resolve_live_contracts(signals: pd.DataFrame) -> pd.DataFrame:
    trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    underlying_prices = get_underlying_quotes(
        stock_client, sorted(signals["ticker"].unique().tolist())
    )

    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    all_rows: list[dict[str, Any]] = []

    for _, signal in signals.iterrows():
        ticker = str(signal["ticker"])
        underlying_price = underlying_prices.get(ticker)
        if underlying_price is None or underlying_price <= 0:
            print(f"{ticker}: no valid underlying quote; skipping contract resolution.")
            continue

        reference_dte = float(signal["reference_dte"])
        reference_moneyness = float(signal["reference_moneyness"])
        target_min_dte = max(
            MIN_DTE, int(round(reference_dte - MAX_DTE_DISTANCE_FROM_REFERENCE))
        )
        target_max_dte = min(
            MAX_DTE, int(round(reference_dte + MAX_DTE_DISTANCE_FROM_REFERENCE))
        )
        if target_min_dte > target_max_dte:
            print(
                f"{ticker}: invalid target DTE range {target_min_dte}-{target_max_dte}; "
                "increase MAX_DTE or regenerate the signal."
            )
            continue

        contracts = fetch_active_contracts(
            trading_client=trading_client,
            ticker=ticker,
            contract_type=str(signal["contract_type"]),
            start_expiration=today + pd.Timedelta(days=target_min_dte),
            end_expiration=today + pd.Timedelta(days=target_max_dte),
        )
        print(
            f"{ticker}: underlying=${underlying_price:.2f}, "
            f"reference_dte={reference_dte:.0f}, "
            f"reference_moneyness={reference_moneyness:.6f}, "
            f"target_dte={target_min_dte}-{target_max_dte}, "
            f"contracts_returned={len(contracts)}"
        )

        candidates: list[dict[str, Any]] = []
        for contract in contracts:
            symbol = getattr(contract, "symbol", None)
            expiration_value = getattr(contract, "expiration_date", None)
            expiration = pd.Timestamp(expiration_value)
            strike = pd.to_numeric(
                getattr(contract, "strike_price", None), errors="coerce"
            )
            actual_type = normalize_contract_type(getattr(contract, "type", ""))

            if not symbol or pd.isna(expiration) or pd.isna(strike) or strike <= 0:
                continue
            if actual_type != signal["contract_type"]:
                continue

            live_dte = int((expiration.normalize() - today).days)
            if not (MIN_DTE <= live_dte <= MAX_DTE):
                continue

            # This matches the inference file's reference moneyness convention:
            # strike / underlying price.
            live_moneyness = float(strike) / float(underlying_price)
            dte_distance = abs(live_dte - reference_dte)
            moneyness_distance = abs(live_moneyness - reference_moneyness)

            if moneyness_distance > MAX_MONEYNESS_DISTANCE_FROM_REFERENCE:
                continue

            candidates.append(
                {
                    **signal.to_dict(),
                    "option_symbol": str(symbol),
                    "expiration_date": expiration.date().isoformat(),
                    "strike_price": float(strike),
                    "dte": live_dte,
                    "underlying_reference_price": float(underlying_price),
                    "live_moneyness": live_moneyness,
                    "dte_distance": dte_distance,
                    "moneyness_distance": moneyness_distance,
                    "profile_distance": dte_distance + 20.0 * moneyness_distance,
                }
            )

        print(f"{ticker}: contracts_matching_all_filters={len(candidates)}")
        if not candidates:
            print(
                f"{ticker}: no active {signal['contract_type']} contracts matched "
                "the EOD profile."
            )
            continue

        ranked = pd.DataFrame(candidates).sort_values(
            ["profile_distance", "dte_distance", "moneyness_distance"],
            ascending=[True, True, True],
        )
        all_rows.extend(ranked.head(MAX_CONTRACTS_PER_SIGNAL).to_dict(orient="records"))

    shortlist = pd.DataFrame(all_rows)
    if shortlist.empty:
        return shortlist
    return shortlist.drop_duplicates("option_symbol").reset_index(drop=True)


def quote_age_seconds(quote_timestamp: str) -> float | None:
    timestamp = as_utc_timestamp(quote_timestamp)
    if timestamp is None:
        return None
    return max(
        0.0,
        float((pd.Timestamp.now(tz="UTC") - timestamp).total_seconds()),
    )


def score_and_validate(candidate: pd.Series, quote: LiveQuote) -> dict[str, Any]:
    midpoint = quote.midpoint
    cost = quote.round_trip_spread_cost
    debit = quote.ask * 100.0 if quote.ask > 0 else None
    age = quote_age_seconds(quote.timestamp)
    max_cost = float(
        candidate.get("max_round_trip_spread_cost", MAX_ROUND_TRIP_SPREAD_COST)
    )

    passed = (
        midpoint is not None
        and cost is not None
        and cost <= max_cost
        and quote.bid_size >= MIN_QUOTE_SIZE
        and quote.ask_size >= MIN_QUOTE_SIZE
        and age is not None
        and age <= MAX_QUOTE_AGE_SECONDS
    )

    reasons: list[str] = []
    if midpoint is None:
        reasons.append("invalid bid/ask")
    if cost is None or cost > max_cost:
        reasons.append(f"round-trip spread cost exceeds {max_cost:.0%}")
    if quote.bid_size < MIN_QUOTE_SIZE or quote.ask_size < MIN_QUOTE_SIZE:
        reasons.append("insufficient displayed quote size")
    if age is None or age > MAX_QUOTE_AGE_SECONDS:
        reasons.append("quote is stale or timestamp missing")

    live_score = None
    if passed and cost is not None and debit is not None:
        live_score = (
            1000.0 * (1.0 - cost)
            + min(quote.bid_size, quote.ask_size)
            - 0.01 * debit
            - float(candidate.get("profile_distance", 0.0))
        )

    return {
        **candidate.to_dict(),
        **asdict(quote),
        "midpoint": midpoint,
        "round_trip_spread_cost": cost,
        "estimated_one_contract_debit": debit,
        "quote_age_seconds": age,
        "passed_live_gate": passed,
        "rejection_reason": "; ".join(reasons),
        "live_score": live_score,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_ranked_output(
    shortlist: pd.DataFrame, latest_quotes: dict[str, LiveQuote]
) -> None:
    rows: list[dict[str, Any]] = []
    for _, candidate in shortlist.iterrows():
        quote = latest_quotes.get(candidate["option_symbol"])
        if quote is not None:
            rows.append(score_and_validate(candidate, quote))

    if not rows:
        return

    ranked = pd.DataFrame(rows).sort_values(
        ["passed_live_gate", "live_score", "profile_distance"],
        ascending=[False, False, True],
        na_position="last",
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUTPUT_CSV, index=False)

    top = ranked.iloc[0]
    cost = top["round_trip_spread_cost"]
    cost_text = f"{cost:.2%}" if pd.notna(cost) else "n/a"
    debit = top["estimated_one_contract_debit"]
    debit_text = f"${debit:.2f}" if pd.notna(debit) else "n/a"
    print(
        f"{top['ticker']} {top['direction']} | {top['option_symbol']} | "
        f"PASS={bool(top['passed_live_gate'])} | "
        f"bid={top['bid']:.2f} ask={top['ask']:.2f} | "
        f"cost={cost_text} | debit={debit_text}"
    )


async def stream_quotes(shortlist: pd.DataFrame) -> None:
    symbols = shortlist["option_symbol"].tolist()

    if not symbols:
        raise RuntimeError(
            "No active live option contracts are available to stream."
        )

    url = f"wss://stream.data.alpaca.markets/v1beta1/{OPTION_FEED}"
    latest_quotes: dict[str, LiveQuote] = {}
    symbol_set = set(symbols)

    async with websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=20,
        additional_headers={
            "Content-Type": "application/msgpack",
        },
    ) as websocket:
        await websocket.send(
            msgpack.packb(
                {
                    "action": "auth",
                    "key": API_KEY,
                    "secret": API_SECRET,
                },
                use_bin_type=True,
            )
        )

        auth_reply = msgpack.unpackb(
            await websocket.recv(),
            raw=False,
        )
        print("AUTH:", auth_reply)

        await websocket.send(
            msgpack.packb(
                {
                    "action": "subscribe",
                    "quotes": symbols,
                },
                use_bin_type=True,
            )
        )

        subscribe_reply = msgpack.unpackb(
            await websocket.recv(),
            raw=False,
        )
        print("SUBSCRIBE:", subscribe_reply)

        errors = [
            message
            for message in subscribe_reply
            if message.get("T") == "error"
        ]

        if errors:
            raise RuntimeError(f"Option-stream subscription failed: {errors}")

        print(
            f"Streaming {len(symbols)} resolved option contracts "
            f"from {OPTION_FEED}."
        )

        while True:
            raw_message = await websocket.recv()

            if isinstance(raw_message, str):
                raise RuntimeError(
                    "Expected MessagePack binary option-stream data, "
                    "but received a text frame."
                )

            messages = msgpack.unpackb(raw_message, raw=False)

            if not isinstance(messages, list):
                messages = [messages]

            for message in messages:
                if message.get("T") != "q":
                    continue

                symbol = message.get("S")

                if symbol not in symbol_set:
                    continue

                latest_quotes[symbol] = LiveQuote(
                    option_symbol=symbol,
                    timestamp=message.get("t"),
                    bid=float(message.get("bp", 0.0) or 0.0),
                    ask=float(message.get("ap", 0.0) or 0.0),
                    bid_size=int(message.get("bs", 0) or 0),
                    ask_size=int(message.get("as", 0) or 0),
                )

                write_ranked_output(shortlist, latest_quotes)


async def main() -> None:
    require_credentials()
    signals = load_direction_signals()
    if signals.empty:
        raise RuntimeError(
            "No ENTER signals in latest_direction_intents.json; nothing to monitor."
        )

    print("Loaded EOD signals:")
    print(
        signals[
            ["ticker", "direction", "reference_dte", "reference_moneyness"]
        ].to_string(index=False)
    )

    shortlist = resolve_live_contracts(signals)
    if shortlist.empty:
        raise RuntimeError("No active contracts matched the EOD direction/reference profile.")

    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(AUDIT_CSV, index=False)
    print(f"Saved active-contract shortlist: {AUDIT_CSV}")
    print(
        shortlist[
            [
                "ticker",
                "direction",
                "option_symbol",
                "dte",
                "live_moneyness",
                "profile_distance",
            ]
        ].to_string(index=False)
    )

    await stream_quotes(shortlist)


if __name__ == "__main__":
    asyncio.run(main())
