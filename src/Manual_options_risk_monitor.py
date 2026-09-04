#This file contains a manual options risk monitoring script

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
from alpaca.trading.client import TradingClient

IST = ZoneInfo("Asia/Kolkata")
ET = ZoneInfo("America/New_York")

CONTRACT_MULTIPLIER = 100
DEFAULT_LOSS_EXIT_RETURN = -0.35
DEFAULT_TAKE_PROFIT_RETURN = 0.30
DEFAULT_FORCE_EXIT_AT_OR_BELOW_DTE = 1
DEFAULT_HARD_DEADLINE_IST = "2026-09-04T20:30:00+05:30"
DEFAULT_MAX_QUOTE_AGE_SECONDS = 60
DEFAULT_MAX_ROUND_TRIP_SPREAD_COST = 0.05


@dataclass
class OptionQuote:
    symbol: str
    timestamp: str | None
    bid: float | None
    ask: float | None
    bid_size: float | None
    ask_size: float | None

    @property
    def midpoint(self) -> float | None:
        if self.bid is None or self.ask is None or self.bid <= 0 or self.ask <= self.bid:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def round_trip_spread_cost(self) -> float | None:
        if self.bid is None or self.ask is None or self.bid <= 0 or self.ask <= self.bid:
            return None
        return (self.ask - self.bid) / self.ask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Non-executing manual risk overlay for long option positions. "
            "It reads open Alpaca paper positions, evaluates fixed loss/time/liquidity gates, "
            "and writes review files. It never submits or closes orders."
        )
    )
    parser.add_argument("--output-dir", default="risk_overlay_output")
    parser.add_argument("--loss-exit-return", type=float, default=DEFAULT_LOSS_EXIT_RETURN)
    parser.add_argument("--take-profit-return", type=float, default=DEFAULT_TAKE_PROFIT_RETURN)
    parser.add_argument(
        "--force-exit-at-or-below-dte",
        type=int,
        default=DEFAULT_FORCE_EXIT_AT_OR_BELOW_DTE,
    )
    parser.add_argument("--hard-deadline-ist", default=DEFAULT_HARD_DEADLINE_IST)
    parser.add_argument(
        "--max-quote-age-seconds",
        type=float,
        default=DEFAULT_MAX_QUOTE_AGE_SECONDS,
    )
    parser.add_argument(
        "--max-round-trip-spread-cost",
        type=float,
        default=DEFAULT_MAX_ROUND_TRIP_SPREAD_COST,
    )
    parser.add_argument(
        "--allowed-ticker",
        action="append",
        default=[],
        help="Optional ticker allow-list; repeat flag for multiple tickers, e.g. --allowed-ticker MRVL",
    )
    return parser.parse_args()


def require_credentials() -> tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET "
            "or APCA_API_KEY_ID and APCA_API_SECRET_KEY in the shell before running."
        )
    return key, secret


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def utc_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def quote_age_seconds(timestamp: str | None) -> float | None:
    parsed = utc_timestamp(timestamp)
    if parsed is None:
        return None
    return max(0.0, (pd.Timestamp.now(tz="UTC") - parsed).total_seconds())


def get_expiration_from_symbol(symbol: str) -> pd.Timestamp | None:
    clean = str(symbol).strip().upper()
    if len(clean) < 15:
        return None
    for index in range(0, max(0, len(clean) - 14)):
        candidate = clean[index:index + 6]
        if candidate.isdigit():
            try:
                return pd.to_datetime(candidate, format="%y%m%d").normalize()
            except ValueError:
                pass
    return None


def classify_position(position: Any) -> str:
    asset_class = str(getattr(position, "asset_class", "")).lower()
    symbol = str(getattr(position, "symbol", "")).upper()
    if "option" in asset_class or len(symbol) >= 15:
        return "OPTION"
    return "OTHER"


def underlying_from_option_symbol(symbol: str) -> str:
    clean = str(symbol).strip().upper()
    for index, char in enumerate(clean):
        if char.isdigit() and len(clean[index:index + 6]) == 6:
            return clean[:index]
    return clean


def fetch_option_quotes(
    client: OptionHistoricalDataClient,
    symbols: list[str],
) -> dict[str, OptionQuote]:
    if not symbols:
        return {}
    response = client.get_option_latest_quote(
        OptionLatestQuoteRequest(symbol_or_symbols=symbols)
    )
    output: dict[str, OptionQuote] = {}
    for symbol in symbols:
        raw = response.get(symbol)
        if raw is None:
            continue
        output[symbol] = OptionQuote(
            symbol=symbol,
            timestamp=str(getattr(raw, "timestamp", "")) or None,
            bid=finite_float(getattr(raw, "bid_price", None)),
            ask=finite_float(getattr(raw, "ask_price", None)),
            bid_size=finite_float(getattr(raw, "bid_size", None)),
            ask_size=finite_float(getattr(raw, "ask_size", None)),
        )
    return output


def estimate_return_at_bid(
    avg_entry_price: float | None,
    bid: float | None,
) -> float | None:
    if avg_entry_price is None or bid is None or avg_entry_price <= 0 or bid <= 0:
        return None
    return (bid / avg_entry_price) - 1.0


def determine_gate(
    estimated_return: float | None,
    dte: int | None,
    now_ist: pd.Timestamp,
    hard_deadline: pd.Timestamp,
    quote: OptionQuote | None,
    max_quote_age_seconds: float,
    max_round_trip_spread_cost: float,
    loss_exit_return: float,
    take_profit_return: float,
    force_exit_at_or_below_dte: int,
) -> tuple[str, bool, list[str]]:
    flags: list[str] = []

    if now_ist >= hard_deadline:
        return "MANUAL_PROJECT_DEADLINE_EXIT", True, ["project deadline reached"]

    if dte is not None and dte <= force_exit_at_or_below_dte:
        return "MANUAL_TIME_EXIT", True, [f"DTE {dte} <= {force_exit_at_or_below_dte}"]

    if estimated_return is not None and estimated_return <= loss_exit_return:
        return "MANUAL_LOSS_EXIT", True, [
            f"bid-based return {estimated_return:.2%} <= loss threshold {loss_exit_return:.2%}"
        ]

    if estimated_return is not None and estimated_return >= take_profit_return:
        return "MANUAL_TAKE_PROFIT_REVIEW", True, [
            f"bid-based return {estimated_return:.2%} >= take-profit threshold {take_profit_return:.2%}"
        ]

    if quote is None:
        return "MANUAL_LIQUIDITY_REVIEW", True, ["no current option quote returned"]

    age = quote_age_seconds(quote.timestamp)
    spread_cost = quote.round_trip_spread_cost
    if age is None or age > max_quote_age_seconds:
        flags.append("quote stale or timestamp missing")
    if quote.bid is None or quote.ask is None or quote.bid <= 0 or quote.ask <= quote.bid:
        flags.append("invalid bid/ask")
    if spread_cost is None or spread_cost > max_round_trip_spread_cost:
        flags.append("spread exceeds configured limit")

    if flags:
        return "MANUAL_LIQUIDITY_REVIEW", True, flags

    return "HOLD_MONITORING", False, ["all fixed manual overlay gates pass"]


def evaluate_positions(args: argparse.Namespace) -> list[dict[str, Any]]:
    key, secret = require_credentials()
    trading_client = TradingClient(key, secret, paper=True)
    option_client = OptionHistoricalDataClient(key, secret)

    allowed = {ticker.upper() for ticker in args.allowed_ticker}
    hard_deadline = pd.Timestamp(args.hard_deadline_ist)
    if hard_deadline.tzinfo is None:
        hard_deadline = hard_deadline.tz_localize(IST)
    else:
        hard_deadline = hard_deadline.tz_convert(IST)
    now_ist = pd.Timestamp.now(tz=IST)
    today_et = pd.Timestamp.now(tz=ET).normalize().tz_localize(None)

    all_positions = list(trading_client.get_all_positions())
    option_positions = [
        position
        for position in all_positions
        if classify_position(position) == "OPTION"
        and finite_float(getattr(position, "qty", None)) not in (None, 0.0)
    ]

    if allowed:
        option_positions = [
            position
            for position in option_positions
            if underlying_from_option_symbol(str(getattr(position, "symbol", ""))) in allowed
        ]

    symbols = [str(getattr(position, "symbol")) for position in option_positions]
    quotes = fetch_option_quotes(option_client, symbols)

    rows: list[dict[str, Any]] = []
    for position in option_positions:
        symbol = str(getattr(position, "symbol"))
        quote = quotes.get(symbol)
        expiration = get_expiration_from_symbol(symbol)
        dte = int((expiration - today_et).days) if expiration is not None else None
        avg_entry_price = finite_float(getattr(position, "avg_entry_price", None))
        quantity = finite_float(getattr(position, "qty", None))
        current_market_value = finite_float(getattr(position, "market_value", None))
        unrealized_pl = finite_float(getattr(position, "unrealized_pl", None))
        unrealized_plpc = finite_float(getattr(position, "unrealized_plpc", None))
        estimated_return = estimate_return_at_bid(
            avg_entry_price,
            quote.bid if quote else None,
        )

        gate, exit_review_required, reasons = determine_gate(
            estimated_return=estimated_return,
            dte=dte,
            now_ist=now_ist,
            hard_deadline=hard_deadline,
            quote=quote,
            max_quote_age_seconds=args.max_quote_age_seconds,
            max_round_trip_spread_cost=args.max_round_trip_spread_cost,
            loss_exit_return=args.loss_exit_return,
            take_profit_return=args.take_profit_return,
            force_exit_at_or_below_dte=args.force_exit_at_or_below_dte,
        )

        entry_debit = (
            avg_entry_price * CONTRACT_MULTIPLIER * abs(quantity)
            if avg_entry_price is not None and quantity is not None
            else None
        )
        bid_exit_value = (
            quote.bid * CONTRACT_MULTIPLIER * abs(quantity)
            if quote and quote.bid is not None and quantity is not None
            else None
        )

        rows.append(
            {
                "checked_at_ist": now_ist.isoformat(),
                "overlay_version": "manual-risk-overlay-v1",
                "auto_exit_enabled": False,
                "recommended_action": (
                    "REVIEW_AND_MANUALLY_CLOSE"
                    if exit_review_required
                    else "HOLD_AND_CONTINUE_MONITORING"
                ),
                "risk_gate": gate,
                "risk_reasons": "; ".join(reasons),
                "option_symbol": symbol,
                "underlying_ticker": underlying_from_option_symbol(symbol),
                "side": str(getattr(position, "side", "")),
                "quantity": quantity,
                "avg_entry_price": avg_entry_price,
                "entry_debit_estimate": entry_debit,
                "broker_market_value": current_market_value,
                "broker_unrealized_pl": unrealized_pl,
                "broker_unrealized_plpc": unrealized_plpc,
                "expiration_date": expiration.date().isoformat() if expiration is not None else None,
                "dte": dte,
                "bid": quote.bid if quote else None,
                "ask": quote.ask if quote else None,
                "bid_size": quote.bid_size if quote else None,
                "ask_size": quote.ask_size if quote else None,
                "quote_timestamp": quote.timestamp if quote else None,
                "quote_age_seconds": quote_age_seconds(quote.timestamp) if quote else None,
                "round_trip_spread_cost": quote.round_trip_spread_cost if quote else None,
                "estimated_bid_exit_value": bid_exit_value,
                "estimated_bid_return": estimated_return,
                "configured_loss_exit_return": args.loss_exit_return,
                "configured_take_profit_return": args.take_profit_return,
                "configured_force_exit_at_or_below_dte": args.force_exit_at_or_below_dte,
                "hard_deadline_IST": hard_deadline.isoformat(),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = evaluate_positions(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checked_at = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"manual_risk_overlay_{checked_at}.csv"
    json_path = output_dir / "latest_manual_risk_overlay.json"

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(rows, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )

    if dataframe.empty:
        print("No open option positions matched the selected scope.")
    else:
        visible_columns = [
            "underlying_ticker",
            "option_symbol",
            "quantity",
            "dte",
            "bid",
            "ask",
            "estimated_bid_return",
            "risk_gate",
            "recommended_action",
        ]
        print(dataframe[visible_columns].to_string(index=False))

    print(f"Saved review CSV: {csv_path}")
    print(f"Saved latest review JSON: {json_path}")
    print("No orders were submitted or closed by this script.")


if __name__ == "__main__":
    main()
