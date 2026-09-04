"""Read-only Alpaca paper-account performance report.

This script never submits, replaces, cancels, or closes orders. It retrieves
account, portfolio-history, order, and activity records and writes an auditable
JSON report plus CSV exports.

Metrics:
- reporting period, start/base equity, current equity, period account P&L
- completed round trips, wins/losses/breakevens, observed win rate
- reconstructed realized P&L from matched buy-to-open / sell-to-close fills
- current unrealized P&L from open positions
- maximum drawdown from timestamped equity history

Important limitations:
- The trade ledger is designed for single-leg, long-option round trips.
- It uses FIFO lot matching by exact option symbol.
- It flags unmatched/partial lots rather than silently treating them as closed.
- It reports Alpaca account-level P&L separately from reconstructed strategy
  P&L. Account P&L can include activity outside this strategy.

Install:
    pip install alpaca-py pandas

Run (paper account only):
    export ALPACA_API_KEY='...'
    export ALPACA_API_SECRET='...'
    python alpaca_paper_performance_report.py \
      --start 2026-08-28T00:00:00Z \
      --end 2026-09-04T23:59:59Z \
      --output-dir evidence/performance_report

Optional filters:
    --underlyings MRVL DELL AVGO
    --only-options

Never commit a .env file, API keys, or secret keys.
"""

from __future__ import annotations
import requests

import argparse
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import UUID


import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ActivityType, AssetClass, OrderStatus, QueryOrderStatus
from alpaca.trading.requests import GetPortfolioHistoryRequest, GetOrdersRequest


EPSILON = Decimal("0.000001")
MULTIPLIER = Decimal("100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Alpaca paper-account performance and trade-ledger report"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Reporting-period start, ISO-8601, e.g. 2026-08-28T00:00:00Z",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Reporting-period end, ISO-8601; defaults to now (UTC)",
    )
    parser.add_argument(
        "--output-dir",
        default="evidence/performance_report",
        help="Directory for JSON and CSV outputs",
    )
    parser.add_argument(
        "--underlyings",
        nargs="*",
        default=[],
        help="Optional underlying filters, e.g. MRVL DELL AVGO",
    )
    parser.add_argument(
        "--only-options",
        action="store_true",
        help="Use only option fills/positions in the reconstructed trade ledger",
    )
    parser.add_argument(
        "--period",
        default=None,
        help="Alpaca portfolio-history period such as 1D, 1W, 1M, 3M, 1A, all; default is all",
    )
    parser.add_argument(
        "--timeframe",
        default="1H",
        choices=["1Min", "5Min", "15Min", "1H", "1D"],
        help="Portfolio-history observation frequency; default 1H",
    )
    return parser.parse_args()


def get_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def parse_timestamp(value: str | None, label: str) -> pd.Timestamp:
    if not value:
        return pd.Timestamp.now(tz="UTC")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise ValueError(f"Invalid {label} timestamp: {value!r}")
    return timestamp


def as_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return default


def json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if hasattr(value, "value"):
        return value.value

    if isinstance(value, Path):
        return str(value)

    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def activity_to_dict(activity: Any) -> dict[str, Any]:
    data = activity.model_dump() if hasattr(activity, "model_dump") else dict(activity)
    return {key: json_safe_or_value(value) for key, value in data.items()}

def json_safe_or_value(value: Any) -> Any:
    if isinstance(value, (Decimal, pd.Timestamp, datetime, UUID, Path)):
        return json_safe(value)

    if hasattr(value, "value"):
        return value.value

    if isinstance(value, list):
        return [json_safe_or_value(item) for item in value]

    if isinstance(value, tuple):
        return [json_safe_or_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): json_safe_or_value(item)
            for key, item in value.items()
        }

    return value

def is_option_symbol(symbol: str) -> bool:
    # Alpaca/OCC option symbols contain a 6-digit expiration followed by C/P and strike.
    # This is only a fallback heuristic; asset_class is preferred when present.
    symbol = str(symbol).strip().upper()
    return len(symbol) >= 15 and any(marker in symbol for marker in ("C", "P"))


def underlying_from_option_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    for index in range(max(0, len(symbol) - 15)):
        suffix = symbol[index:]
        if len(suffix) >= 15 and suffix[:6].isdigit() and suffix[6:7] in {"C", "P"}:
            return symbol[:index]
    return symbol


def normalize_side(value: Any) -> str:
    return enum_value(value).strip().lower()


def normalize_asset_class(value: Any) -> str:
    return enum_value(value).strip().lower()


def order_to_dict(order: Any) -> dict[str, Any]:
    data = order.model_dump() if hasattr(order, "model_dump") else dict(order)
    return {key: json_safe_or_value(value) for key, value in data.items()}


def fetch_orders(
    client: TradingClient, start: pd.Timestamp, end: pd.Timestamp
) -> list[dict[str, Any]]:
    request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=start.to_pydatetime(),
        until=end.to_pydatetime(),
        direction="asc",
        nested=True,
        limit=500,
    )
    orders = client.get_orders(filter=request)
    return [order_to_dict(order) for order in orders]


def fetch_activities(
    client: TradingClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    """
    Read-only direct REST request for Alpaca paper-account FILL activities.

    TradingClient does not provide get_activities() in the installed alpaca-py
    version. This endpoint returns actual fill records required for FIFO
    entry/exit matching.
    """
    api_key = get_env("ALPACA_API_KEY")
    api_secret = get_env("ALPACA_API_SECRET")

    url = "https://paper-api.alpaca.markets/v2/account/activities"

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    params: dict[str, Any] = {
        "activity_types": "FILL",
        "after": start.isoformat().replace("+00:00", "Z"),
        "until": end.isoformat().replace("+00:00", "Z"),
        "direction": "asc",
        "page_size": 100,
    }

    all_rows: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        request_params = dict(params)

        if page_token:
            request_params["page_token"] = page_token

        response = requests.get(
            url,
            headers=headers,
            params=request_params,
            timeout=30,
        )

        response.raise_for_status()

        rows = response.json()

        if not rows:
            break

        if not isinstance(rows, list):
            raise RuntimeError(
                f"Unexpected activities response type: {type(rows).__name__}. "
                f"Response: {rows}"
            )

        all_rows.extend(rows)

        if len(rows) < 100:
            break

        next_token = rows[-1].get("id")

        if not next_token or next_token == page_token:
            break

        page_token = str(next_token)

    filtered: list[dict[str, Any]] = []

    for row in all_rows:
        timestamp_value = row.get("transaction_time") or row.get("date")

        if not timestamp_value:
            continue

        timestamp = parse_timestamp(str(timestamp_value), "activity")

        if start <= timestamp <= end:
            row["transaction_time_utc"] = timestamp.isoformat()
            filtered.append(row)

    return filtered


def fetch_portfolio_history(
    client: TradingClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period: str | None,
    timeframe: str,
) -> pd.DataFrame:
    # Alpaca computes portfolio-history P&L relative to the chosen base value.
    # We calculate maximum drawdown ourselves from returned timestamped equity.
    request_kwargs: dict[str, Any] = {
        "timeframe": timeframe,
        "intraday_reporting": "continuous",
        "pnl_reset": "per_day",
        "date_start": start.date(),
        "date_end": end.date(),
    }
    if period:
        request_kwargs["period"] = period

    history = client.get_portfolio_history(
        GetPortfolioHistoryRequest(**request_kwargs)
    )
    timestamps = list(getattr(history, "timestamp", []) or [])
    equities = list(getattr(history, "equity", []) or [])
    pnl_values = list(getattr(history, "profit_loss", []) or [])
    pnl_pct_values = list(getattr(history, "profit_loss_pct", []) or [])

    rows: list[dict[str, Any]] = []
    for index, timestamp_value in enumerate(timestamps):
        timestamp = pd.to_datetime(timestamp_value, unit="s", utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        equity = equities[index] if index < len(equities) else None
        pnl = pnl_values[index] if index < len(pnl_values) else None
        pnl_pct = pnl_pct_values[index] if index < len(pnl_pct_values) else None
        if equity is None:
            continue
        rows.append(
            {
                "timestamp_utc": timestamp,
                "equity": float(equity),
                "alpaca_profit_loss": float(pnl) if pnl is not None else None,
                "alpaca_profit_loss_pct": float(pnl_pct) if pnl_pct is not None else None,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.dropna(subset=["equity"]).sort_values("timestamp_utc")
    return frame.drop_duplicates(subset=["timestamp_utc"], keep="last").reset_index(drop=True)


def calculate_drawdown(equity_history: pd.DataFrame) -> dict[str, Any]:
    if len(equity_history) < 2:
        return {
            "available": False,
            "reason": "Fewer than two equity observations were returned.",
            "maximum_drawdown_dollars": None,
            "maximum_drawdown_percent": None,
            "peak_equity": None,
            "peak_timestamp_utc": None,
            "trough_equity": None,
            "trough_timestamp_utc": None,
        }

    frame = equity_history.copy()
    frame["running_peak_equity"] = frame["equity"].cummax()
    frame["drawdown_dollars"] = frame["equity"] - frame["running_peak_equity"]
    frame["drawdown_percent"] = 100.0 * (
        frame["equity"] / frame["running_peak_equity"] - 1.0
    )
    trough_index = frame["drawdown_dollars"].idxmin()
    trough = frame.loc[trough_index]

    # The relevant peak must occur on or before the trough, not necessarily the
    # final/current all-period peak.
    preceding = frame.loc[:trough_index]
    peak_index = preceding["equity"].idxmax()
    peak = frame.loc[peak_index]

    return {
        "available": True,
        "maximum_drawdown_dollars": float(abs(trough["drawdown_dollars"])),
        "maximum_drawdown_percent": float(abs(trough["drawdown_percent"])),
        "peak_equity": float(peak["equity"]),
        "peak_timestamp_utc": peak["timestamp_utc"].isoformat(),
        "trough_equity": float(trough["equity"]),
        "trough_timestamp_utc": trough["timestamp_utc"].isoformat(),
        "current_equity_is_high_water_mark": bool(
            abs(frame.iloc[-1]["equity"] - frame["equity"].max()) < 1e-9
        ),
        "observations": int(len(frame)),
    }


def activity_is_option(fill: dict[str, Any]) -> bool:
    asset_class = normalize_asset_class(fill.get("asset_class", ""))
    if asset_class:
        return asset_class == AssetClass.US_OPTION.value or "option" in asset_class
    return is_option_symbol(str(fill.get("symbol", "")))


def filter_fills(
    fills: list[dict[str, Any]], underlyings: set[str], only_options: bool
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for fill in fills:
        symbol = str(fill.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        is_option = activity_is_option(fill)
        if only_options and not is_option:
            continue
        underlying = underlying_from_option_symbol(symbol) if is_option else symbol
        if underlyings and underlying not in underlyings:
            continue
        fill = dict(fill)
        fill["symbol"] = symbol
        fill["underlying"] = underlying
        fill["is_option"] = is_option
        selected.append(fill)
    return selected


def build_trade_ledger(fills: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FIFO-match long opening buys with later sells of the exact same symbol.

    Returns completed round trips and unmatched lot records. This intentionally
    does not attempt to infer short-option or multi-leg-spread P&L.
    """
    sorted_fills = sorted(
        fills,
        key=lambda row: parse_timestamp(
            str(row.get("transaction_time_utc") or row.get("transaction_time")), "fill"
        ),
    )
    open_lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    completed: list[dict[str, Any]] = []
    unmatched_sells: list[dict[str, Any]] = []
    sequence = 0

    for fill in sorted_fills:
        symbol = str(fill["symbol"])
        side = normalize_side(fill.get("side"))
        quantity = as_decimal(fill.get("qty") or fill.get("quantity"))
        price = as_decimal(fill.get("price"))
        timestamp = parse_timestamp(
            str(fill.get("transaction_time_utc") or fill.get("transaction_time")),
            "fill",
        )
        order_id = str(fill.get("order_id") or "")
        activity_id = str(fill.get("id") or "")

        if quantity <= EPSILON or price < 0:
            continue

        if side == "buy":
            open_lots[symbol].append(
                {
                    "remaining_qty": quantity,
                    "entry_price": price,
                    "entry_time": timestamp,
                    "entry_order_id": order_id,
                    "entry_activity_id": activity_id,
                    "underlying": fill["underlying"],
                    "is_option": bool(fill["is_option"]),
                }
            )
            continue

        if side != "sell":
            continue

        remaining_to_match = quantity
        while remaining_to_match > EPSILON and open_lots[symbol]:
            lot = open_lots[symbol][0]
            matched_qty = min(remaining_to_match, lot["remaining_qty"])
            multiplier = MULTIPLIER if lot["is_option"] else Decimal("1")
            realized_pnl = (price - lot["entry_price"]) * matched_qty * multiplier
            sequence += 1
            completed.append(
                {
                    "trade_id": f"{symbol}-{sequence:04d}",
                    "underlying": lot["underlying"],
                    "option_symbol": symbol if lot["is_option"] else None,
                    "symbol": symbol,
                    "asset_type": "option" if lot["is_option"] else "non_option",
                    "entry_timestamp_utc": lot["entry_time"].isoformat(),
                    "exit_timestamp_utc": timestamp.isoformat(),
                    "entry_quantity": float(matched_qty),
                    "exit_quantity": float(matched_qty),
                    "entry_vwap": float(lot["entry_price"]),
                    "exit_vwap": float(price),
                    "contract_multiplier": float(multiplier),
                    "realized_pnl_before_fees": float(realized_pnl),
                    "entry_order_id": lot["entry_order_id"],
                    "exit_order_id": order_id,
                    "entry_activity_id": lot["entry_activity_id"],
                    "exit_activity_id": activity_id,
                    "holding_duration_hours": round(
                        (timestamp - lot["entry_time"]).total_seconds() / 3600.0, 4
                    ),
                    "pnl_classification": (
                        "win"
                        if realized_pnl > EPSILON
                        else "loss"
                        if realized_pnl < -EPSILON
                        else "breakeven"
                    ),
                }
            )
            lot["remaining_qty"] -= matched_qty
            remaining_to_match -= matched_qty
            if lot["remaining_qty"] <= EPSILON:
                open_lots[symbol].popleft()

        if remaining_to_match > EPSILON:
            unmatched_sells.append(
                {
                    "symbol": symbol,
                    "underlying": fill["underlying"],
                    "timestamp_utc": timestamp.isoformat(),
                    "side": "sell",
                    "unmatched_quantity": float(remaining_to_match),
                    "fill_price": float(price),
                    "order_id": order_id,
                    "activity_id": activity_id,
                    "reason": "No matching long buy lot in reporting-period data",
                }
            )

    unmatched_open_lots: list[dict[str, Any]] = []
    for symbol, lots in open_lots.items():
        for lot in lots:
            unmatched_open_lots.append(
                {
                    "symbol": symbol,
                    "underlying": lot["underlying"],
                    "timestamp_utc": lot["entry_time"].isoformat(),
                    "side": "buy",
                    "unmatched_quantity": float(lot["remaining_qty"]),
                    "fill_price": float(lot["entry_price"]),
                    "order_id": lot["entry_order_id"],
                    "activity_id": lot["entry_activity_id"],
                    "reason": "Long buy lot remains unmatched/open at report end",
                }
            )

    ledger = pd.DataFrame(completed)
    unmatched = pd.DataFrame(unmatched_sells + unmatched_open_lots)
    return ledger, unmatched


def summarize_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {
            "completed_round_trips": 0,
            "winning_completed_trades": 0,
            "losing_completed_trades": 0,
            "breakeven_completed_trades": 0,
            "observed_win_rate_percent": None,
            "realized_strategy_pnl_before_fees": 0.0,
        }

    wins = int((ledger["pnl_classification"] == "win").sum())
    losses = int((ledger["pnl_classification"] == "loss").sum())
    breakevens = int((ledger["pnl_classification"] == "breakeven").sum())
    total = int(len(ledger))
    return {
        "completed_round_trips": total,
        "winning_completed_trades": wins,
        "losing_completed_trades": losses,
        "breakeven_completed_trades": breakevens,
        "observed_win_rate_percent": round(100.0 * wins / total, 2) if total else None,
        "realized_strategy_pnl_before_fees": round(
            float(ledger["realized_pnl_before_fees"].sum()), 2
        ),
    }


def summarize_orders(orders: list[dict[str, Any]], only_options: bool) -> dict[str, Any]:
    filtered = orders
    if only_options:
        filtered = [
            row
            for row in orders
            if normalize_asset_class(row.get("asset_class", ""))
            in {AssetClass.US_OPTION.value, "us_option", "option"}
            or is_option_symbol(str(row.get("symbol", "")))
        ]

    entries_submitted = 0
    entries_filled = 0
    exceptional: list[dict[str, Any]] = []
    for order in filtered:
        side = normalize_side(order.get("side"))
        status = enum_value(order.get("status")).lower()
        qty = as_decimal(order.get("qty"))
        filled_qty = as_decimal(order.get("filled_qty"))
        if side == "buy":
            entries_submitted += 1
            if filled_qty > EPSILON or status == OrderStatus.FILLED.value:
                entries_filled += 1
        if status in {
            OrderStatus.REJECTED.value,
            OrderStatus.CANCELED.value,
            OrderStatus.EXPIRED.value,
        } or (filled_qty > EPSILON and filled_qty < qty):
            exceptional.append(
                {
                    "id": order.get("id"),
                    "client_order_id": order.get("client_order_id"),
                    "symbol": order.get("symbol"),
                    "side": side,
                    "status": status,
                    "qty": float(qty),
                    "filled_qty": float(filled_qty),
                    "submitted_at": order.get("submitted_at"),
                    "filled_at": order.get("filled_at"),
                    "failed_at": order.get("failed_at"),
                    "canceled_at": order.get("canceled_at"),
                    "expired_at": order.get("expired_at"),
                    "reject_reason": order.get("reject_reason"),
                }
            )

    return {
        "orders_retrieved": len(filtered),
        "entries_submitted_buy_orders": entries_submitted,
        "filled_entry_buy_orders": entries_filled,
        "exceptional_orders": exceptional,
    }


def positions_to_frame(positions: list[Any], only_options: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position in positions:
        data = position.model_dump() if hasattr(position, "model_dump") else dict(position)
        asset_class = normalize_asset_class(data.get("asset_class", ""))
        symbol = str(data.get("symbol", "")).strip().upper()
        is_option = asset_class == AssetClass.US_OPTION.value or "option" in asset_class or is_option_symbol(symbol)
        if only_options and not is_option:
            continue
        rows.append(
            {
                "symbol": symbol,
                "underlying": underlying_from_option_symbol(symbol) if is_option else symbol,
                "asset_type": "option" if is_option else asset_class,
                "qty": float(as_decimal(data.get("qty"))),
                "side": data.get("side"),
                "avg_entry_price": float(as_decimal(data.get("avg_entry_price"))),
                "current_price": float(as_decimal(data.get("current_price"))),
                "market_value": float(as_decimal(data.get("market_value"))),
                "cost_basis": float(as_decimal(data.get("cost_basis"))),
                "unrealized_pl": float(as_decimal(data.get("unrealized_pl"))),
                "unrealized_plpc": float(as_decimal(data.get("unrealized_plpc"))),
            }
        )
    return pd.DataFrame(rows)


def make_report(
    account: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    equity_history: pd.DataFrame,
    drawdown: dict[str, Any],
    order_summary: dict[str, Any],
    ledger_summary: dict[str, Any],
    positions: pd.DataFrame,
    unmatched: pd.DataFrame,
) -> dict[str, Any]:
    current_equity = as_decimal(getattr(account, "equity", None))
    current_cash = as_decimal(getattr(account, "cash", None))
    current_buying_power = as_decimal(getattr(account, "buying_power", None))

    if equity_history.empty:
        start_equity = None
        ending_equity_from_series = None
        period_pnl = None
        period_pnl_pct = None
        base_note = "Portfolio history returned no equity observations for the selected period."
    else:
        start_equity = float(equity_history.iloc[0]["equity"])
        ending_equity_from_series = float(equity_history.iloc[-1]["equity"])
        period_pnl = float(current_equity) - start_equity
        period_pnl_pct = 100.0 * period_pnl / start_equity if start_equity else None
        base_note = (
            "Start equity is the first timestamped equity observation returned by "
            "Alpaca portfolio history for this report window."
        )

    unrealized = 0.0
    if not positions.empty:
        unrealized = round(float(positions["unrealized_pl"].sum()), 2)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "account_environment": "ALPACA_PAPER",
            "reporting_period_start_utc": start.isoformat(),
            "reporting_period_end_utc": end.isoformat(),
            "strategy_filter_note": (
                "Optional underlying filter was applied only to fill/order ledger reconstruction. "
                "Account equity and drawdown remain account-level metrics."
            ),
        },
        "account_level_metrics": {
            "starting_equity_base_value": start_equity,
            "current_equity": float(current_equity),
            "equity_at_last_portfolio_history_observation": ending_equity_from_series,
            "period_account_pnl_dollars": round(period_pnl, 2) if period_pnl is not None else None,
            "period_account_pnl_percent": round(period_pnl_pct, 4) if period_pnl_pct is not None else None,
            "cash": float(current_cash),
            "buying_power": float(current_buying_power),
            "base_value_note": base_note,
        },
        "trade_statistics": {
            **order_summary,
            **ledger_summary,
            "unrealized_pnl_open_positions": unrealized,
            "unmatched_lot_records": int(len(unmatched)),
            "ledger_method": (
                "FIFO matching of long buy fills to later sell fills by exact symbol; "
                "intended for single-leg long-option round trips."
            ),
        },
        "maximum_drawdown": drawdown,
        "interpretation_notes": [
            "Account-level P&L and drawdown may include account activity outside the filtered strategy ledger.",
            "Realized strategy P&L is reconstructed before fees from matched fills; it is not an Alpaca tax-lot or broker statement.",
            "Open positions are excluded from completed-round-trip count and observed win rate; their current unrealized P&L is reported separately.",
            "If trades were opened before the report start, widen --start so opening fills are available for correct FIFO matching.",
            "A small number of completed trades produces a descriptive observed win rate, not a statistically reliable performance estimate.",
        ],
    }


def main() -> None:
    args = parse_args()
    api_key = get_env("ALPACA_API_KEY")
    api_secret = get_env("ALPACA_API_SECRET")
    start = parse_timestamp(args.start, "start")
    end = parse_timestamp(args.end, "end")
    if end <= start:
        raise ValueError("--end must be after --start")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    underlyings = {item.upper().strip() for item in args.underlyings if item.strip()}

    client = TradingClient(api_key, api_secret, paper=True)
    account = client.get_account()
    orders = fetch_orders(client, start, end)
    activities = fetch_activities(client, start, end)
    fills = filter_fills(activities, underlyings, args.only_options)
    positions = positions_to_frame(client.get_all_positions(), args.only_options)
    equity_history = fetch_portfolio_history(
        client=client,
        start=start,
        end=end,
        period=args.period,
        timeframe=args.timeframe,
    )

    ledger, unmatched = build_trade_ledger(fills)
    ledger_summary = summarize_ledger(ledger)
    order_summary = summarize_orders(orders, args.only_options)
    drawdown = calculate_drawdown(equity_history)
    report = make_report(
        account=account,
        start=start,
        end=end,
        equity_history=equity_history,
        drawdown=drawdown,
        order_summary=order_summary,
        ledger_summary=ledger_summary,
        positions=positions,
        unmatched=unmatched,
    )

    pd.DataFrame(orders).to_csv(output_dir / "orders_raw.csv", index=False)
    pd.DataFrame(activities).to_csv(output_dir / "fills_raw.csv", index=False)
    ledger.to_csv(output_dir / "completed_trade_ledger.csv", index=False)
    unmatched.to_csv(output_dir / "unmatched_lots.csv", index=False)
    positions.to_csv(output_dir / "open_positions.csv", index=False)
    equity_history.to_csv(output_dir / "equity_history.csv", index=False)
    pd.DataFrame(order_summary["exceptional_orders"]).to_csv(
        output_dir / "rejected_canceled_expired_or_partial_orders.csv", index=False
    )
    (output_dir / "performance_report.json").write_text(
        json.dumps(report, indent=2, default=json_safe), encoding="utf-8"
    )

    account_metrics = report["account_level_metrics"]
    trade_metrics = report["trade_statistics"]
    dd = report["maximum_drawdown"]
    print("\nALPACA PAPER PERFORMANCE REPORT")
    print(f"Reporting period: {start.isoformat()} to {end.isoformat()}")
    print(f"Starting equity/base value: {account_metrics['starting_equity_base_value']}")
    print(f"Current equity: ${account_metrics['current_equity']:,.2f}")
    period_pnl = account_metrics.get("period_account_pnl_dollars")
    period_pnl_pct = account_metrics.get("period_account_pnl_percent")

    if period_pnl is not None and period_pnl_pct is not None:
        print(
            "Period account P&L: "
            f"${period_pnl:,.2f} "
            f"({period_pnl_pct:.4f}%)"
        )
    elif period_pnl is not None:
        print(f"Period account P&L: ${period_pnl:,.2f} (percentage unavailable)")
    else:
        print("Period account P&L: unavailable (no usable portfolio equity baseline)")
    print(f"Completed round trips: {trade_metrics['completed_round_trips']}")
    print(
        "Winning / losing / break-even: "
        f"{trade_metrics['winning_completed_trades']} / "
        f"{trade_metrics['losing_completed_trades']} / "
        f"{trade_metrics['breakeven_completed_trades']}"
    )
    print(f"Observed win rate: {trade_metrics['observed_win_rate_percent']}")
    print(
        "Realized strategy P&L before fees: "
        f"${trade_metrics['realized_strategy_pnl_before_fees']:,.2f}"
    )
    print(
        "Unrealized P&L on current open positions: "
        f"${trade_metrics['unrealized_pnl_open_positions']:,.2f}"
    )
    if dd.get("available"):
        print(
            "Maximum drawdown: "
            f"${dd['maximum_drawdown_dollars']:,.2f} "
            f"({dd['maximum_drawdown_percent']:.4f}%)"
        )
        print(f"Peak → trough: ${dd['peak_equity']:,.2f} → ${dd['trough_equity']:,.2f}")
    else:
        print(f"Maximum drawdown: unavailable ({dd.get('reason')})")
    print(f"\nSaved auditable report files to: {output_dir}")


if __name__ == "__main__":
    main()
