"""
OptionRelay — EOD DQN Inference Reference Implementation

This script is inference-only: it never connects to Alpaca and never submits,
cancels, modifies, or closes orders.

It supports the current feature_matrix.csv naming convention and the downloaded
artifact naming convention:
    scaler bundle keys: available_features, scaled_feature_columns, medians, scaler
    config keys: state_size, action_size, max_candidates, take_profit_return, ...

The public repository intentionally excludes the historical feature matrix,
saved scaler bundle, trained DQN weights, and private model artifacts.

Usage example:
python options_eod_inference_later_model.py \
    --features feature_matrix.csv \
    --models-root ./optionsdqnresults \
    --decision-date 2026-08-31 \
    --ticker MRVL
"""

from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_TICKERS = ["MRVL", "DELL", "AVGO"]
DEFAULT_MAX_CANDIDATES = 10
CONTRACTS_PER_EXPIRY_TYPE_BUCKET = 1
HARD_FLATTEN_DEADLINE_IST = "2026-09-04T20:30:00+05:30"
MIN_ENTRY_DTE = 1
MAX_ENTRY_DTE = 7

# Current dataset column names → legacy aliases used internally by the old
# training environment. These are ADDED as duplicate aliases; current columns
# remain untouched because the saved scaler expects their current names.
LEGACY_ALIASES = {
    "option_symbol": "optionsymbol",
    "quote_date": "quotedate",
    "option_open": "optionopen",
    "option_high": "optionhigh",
    "option_low": "optionlow",
    "option_close": "optionclose",
    "option_volume": "optionvolume",
    "option_vwap": "optionvwap",
    "option_transactions": "optiontransactions",
    "underlying_ticker": "underlyingticker",
    "strike_price": "strikeprice",
    "contract_type": "contracttype",
    "expiration_date": "expirationdate",
    "stock_open": "stockopen",
    "stock_high": "stockhigh",
    "stock_low": "stocklow",
    "stock_close": "stockclose",
    "stock_volume": "stockvolume",
    "stock_vwap": "stockvwap",
    "stock_transactions": "stocktransactions",
    "DTE_Bucket": "DTEBucket",
    "Log_Moneyness": "LogMoneyness",
    "Option_Premium_to_Stock": "OptionPremiumtoStock",
    "Intrinsic_Value": "IntrinsicValue",
    "Time_Value": "TimeValue",
    "Option_Dollar_Volume": "OptionDollarVolume",
    "Log_Option_Volume": "LogOptionVolume",
    "Log_Dollar_Volume": "LogDollarVolume",
    "EOD_Proxy_Price": "EODProxyPrice",
    "Option_Close_Return_1D": "OptionCloseReturn1D",
    "Previous_Option_Close": "PreviousOptionClose",
    "Previous_Option_Volume": "PreviousOptionVolume",
    "Underlying_Return_1D": "UnderlyingReturn1D",
    "Previous_Stock_Close": "PreviousStockClose",
    "Underlying_Volatility_20D": "UnderlyingVolatility20D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run later-model options DQN EOD inference")
    parser.add_argument("--features", required=True, help="Path to feature_matrix.csv")
    parser.add_argument("--models-root", required=True, help="Directory containing MRVL/ DELL/ AVGO artifact folders")
    parser.add_argument("--decision-date", default="2026-09-01", help="Completed EOD date YYYY-MM-DD")
    parser.add_argument("--ticker", action="append", default=[], help="Ticker to evaluate; repeat flag for several tickers")
    parser.add_argument("--output-dir", default="direction_intents", help="Directory for JSON output")
    return parser.parse_args()


def load_feature_matrix(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    # Current data files use snake_case. Keep those intact for the scaler,
    # but create separate aliases used by the historical environment logic.
    for modern, legacy in LEGACY_ALIASES.items():
        if modern in df.columns and legacy not in df.columns:
            df[legacy] = df[modern]

    # Support a legacy dataset without destroying any current column names.
    if "Ticker" in df.columns and "underlyingticker" not in df.columns:
        df["underlyingticker"] = df["Ticker"]
    if "Date" in df.columns and "quotedate" not in df.columns:
        df["quotedate"] = df["Date"]

    required_legacy = [
        "underlyingticker", "optionsymbol", "quotedate", "expirationdate",
        "contracttype", "strikeprice", "optionopen", "optionclose",
        "optionvolume", "stockclose",
    ]
    missing_legacy = [name for name in required_legacy if name not in df.columns]
    if missing_legacy:
        raise ValueError(f"Feature matrix lacks required identifiers/fields: {missing_legacy}")

    # The downloaded scaler expects is_call. The supplied current feature matrix
    # has contract_type, from which this exact binary feature is deterministically
    # reconstructed. 
    if "is_call" not in df.columns:
        source = "contract_type" if "contract_type" in df.columns else "contracttype"
        df["is_call"] = df[source].astype(str).str.lower().str.strip().eq("call").astype(float)

    df["quotedate"] = pd.to_datetime(df["quotedate"], errors="coerce").dt.normalize()
    df["expirationdate"] = pd.to_datetime(df["expirationdate"], errors="coerce").dt.normalize()
    df["underlyingticker"] = df["underlyingticker"].astype(str).str.upper().str.strip()
    df["optionsymbol"] = df["optionsymbol"].astype(str)
    df["contracttype"] = df["contracttype"].astype(str).str.lower().str.strip()

    numeric_names = [
        "strikeprice", "optionopen", "optionclose", "optionvolume", "optionvwap",
        "optiontransactions", "stockclose", "stockvolume", "stockvwap", "DTE", "Tau",
        "Moneyness", "LogMoneyness", "OptionPremiumtoStock", "IntrinsicValue", "TimeValue",
        "OptionCloseReturn1D", "UnderlyingReturn1D", "UnderlyingVolatility20D",
        "OptionDollarVolume", "LogOptionVolume", "LogDollarVolume", "EODProxyPrice", "is_call",
    ]
    for name in numeric_names:
        if name in df.columns:
            df[name] = pd.to_numeric(df[name], errors="coerce")

    # Training-compatible fallback calculations are used only when a necessary
    # field is absent. Current feature_matrix.csv already supplies these fields.
    if "EODProxyPrice" not in df.columns:
        df["EODProxyPrice"] = df["optionvwap"] if "optionvwap" in df.columns else df["optionclose"]
        df["EODProxyPrice"] = df["EODProxyPrice"].where(df["EODProxyPrice"] > 0, df["optionclose"])
    if "DTE" not in df.columns:
        df["DTE"] = (df["expirationdate"] - df["quotedate"]).dt.days
    if "Tau" not in df.columns:
        df["Tau"] = df["DTE"] / 365.0
    if "Moneyness" not in df.columns:
        df["Moneyness"] = df["stockclose"] / df["strikeprice"]
    if "LogMoneyness" not in df.columns:
        df["LogMoneyness"] = np.log(df["Moneyness"].where(df["Moneyness"] > 0))
    if "OptionDollarVolume" not in df.columns:
        df["OptionDollarVolume"] = df["optionclose"] * df["optionvolume"]
    if "DTEBucket" not in df.columns:
        df["DTEBucket"] = pd.cut(
            df["DTE"],
            bins=[-1, 0, 2, 4, 7, 14, 30, 60, 90, 180, 365, np.inf],
            labels=["0DTE", "1-2DTE", "3-4DTE", "5-7DTE", "8-14DTE", "15-30DTE", "31-60DTE", "61-90DTE", "91-180DTE", "181-365DTE", "365DTE"],
        )

    df = df.dropna(subset=["underlyingticker", "optionsymbol", "quotedate", "expirationdate", "strikeprice", "stockclose", "EODProxyPrice"])
    df = df[(df["DTE"] >= 0) & (df["optionclose"] > 0) & (df["stockclose"] > 0) & (df["strikeprice"] > 0) & (df["EODProxyPrice"] > 0)].copy()
    return df.drop_duplicates(subset=["underlyingticker", "optionsymbol", "quotedate"]).sort_values(["underlyingticker", "quotedate", "optionsymbol"]).reset_index(drop=True)


def artifact_get(mapping: dict[str, Any], modern: str, legacy: str) -> Any:
    if modern in mapping:
        return mapping[modern]
    if legacy in mapping:
        return mapping[legacy]
    raise KeyError(f"Artifact contains neither {modern!r} nor {legacy!r}")


def locate_artifact(ticker_dir: Path, ticker: str, suffix: str) -> Path:
    expected = ticker_dir / f"{ticker}_{suffix}"
    if expected.exists():
        return expected
    matches = sorted(ticker_dir.glob(f"*{suffix}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"{ticker}: cannot find artifact ending {suffix!r} in {ticker_dir}")
    raise FileNotFoundError(f"{ticker}: multiple artifacts ending {suffix!r}: {matches}")


def load_artifacts(models_root: Path, ticker: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    ticker_dir = models_root / ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"{ticker}: ticker directory not found: {ticker_dir}")
    weights_path = locate_artifact(ticker_dir, ticker, "dqn.weights.h5")
    scaler_path = locate_artifact(ticker_dir, ticker, "scaler.joblib")
    config_path = locate_artifact(ticker_dir, ticker, "config.json")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    bundle = joblib.load(scaler_path)
    if not isinstance(bundle, dict):
        raise TypeError(f"{ticker}: scaler artifact is not a dictionary")
    for key in ["scaler", "medians"]:
        if key not in bundle:
            raise ValueError(f"{ticker}: scaler artifact lacks {key!r}")
    return config, bundle, weights_path


def apply_training_scaler(df: pd.DataFrame, bundle: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    available_features = list(artifact_get(bundle, "available_features", "availablefeatures"))
    scaled_columns = list(artifact_get(bundle, "scaled_feature_columns", "scaledfeaturecolumns"))
    if len(available_features) != len(scaled_columns):
        raise ValueError("Saved raw feature and scaled feature lists differ in length")

    missing = [feature for feature in available_features if feature not in df.columns]
    if missing:
        raise ValueError(f"Feature matrix is missing training features: {missing}")

    medians = bundle["medians"]
    values = pd.DataFrame(index=df.index)
    for feature in available_features:
        median = medians.get(feature, 0.0) if hasattr(medians, "get") else 0.0
        values[feature] = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(median)
    if values.isna().any().any():
        remaining = values.columns[values.isna().any()].tolist()
        raise ValueError(f"NaNs remain after saved-median imputation: {remaining}")

    transformed = bundle["scaler"].transform(values[available_features])
    result = df.copy()
    for index, name in enumerate(scaled_columns):
        result[name] = transformed[:, index]
    return result, scaled_columns


def select_daily_candidates(day_df: pd.DataFrame, max_candidates: int) -> list[dict[str, Any]]:
    day = day_df[(day_df["EODProxyPrice"] > 0) & (day_df["optionvolume"] > 0) & (day_df["DTE"] >= 0)].copy()
    if day.empty:
        return []
    day["selection_dollar_volume"] = pd.to_numeric(day["OptionDollarVolume"], errors="coerce").fillna(0.0)
    day["selection_atm_distance"] = pd.to_numeric(day["LogMoneyness"], errors="coerce").abs().fillna(np.inf)
    day["selection_dte_bucket"] = day["DTEBucket"].astype(str).fillna("Unknown")

    representatives = []
    for _, group in day.groupby(["selection_dte_bucket", "contracttype"], dropna=False):
        representatives.append(group.sort_values(["selection_dollar_volume", "selection_atm_distance"], ascending=[False, True]).head(CONTRACTS_PER_EXPIRY_TYPE_BUCKET))
    selected = pd.concat(representatives, ignore_index=True) if representatives else pd.DataFrame(columns=day.columns)
    selected = selected.drop_duplicates(subset=["optionsymbol"]).sort_values(["selection_dollar_volume", "selection_atm_distance"], ascending=[False, True]).head(max_candidates)
    selected_symbols = set(selected["optionsymbol"])
    remaining = day[~day["optionsymbol"].isin(selected_symbols)].sort_values(["selection_dollar_volume", "selection_atm_distance"], ascending=[False, True])
    if len(selected) < max_candidates:
        selected = pd.concat([selected, remaining.head(max_candidates - len(selected))], ignore_index=True)
    return selected.drop_duplicates(subset=["optionsymbol"]).head(max_candidates).to_dict(orient="records")


def build_model(state_size: int, action_size: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(state_size,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(action_size, activation="linear"),
    ])
    return model


def build_flat_entry_state(
    day_df: pd.DataFrame,
    scaled_columns: list[str],
    max_candidates: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if day_df.empty:
        raise ValueError("Cannot build an EOD state from an empty day")
    global_row = day_df.iloc[0]
    global_values = np.asarray([float(global_row.get(column, 0.0)) for column in scaled_columns], dtype=np.float32)
    candidates = select_daily_candidates(day_df, max_candidates)
    candidate_values: list[float] = []
    for index in range(max_candidates):
        if index < len(candidates):
            candidate_values.extend([float(candidates[index].get(column, 0.0)) for column in scaled_columns])
            candidate_values.append(1.0)
        else:
            candidate_values.extend([0.0] * len(scaled_columns))
            candidate_values.append(0.0)
    portfolio_values = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state = np.concatenate([global_values, np.asarray(candidate_values, dtype=np.float32), portfolio_values]).astype(np.float32)
    return state, candidates


def finite_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if np.isfinite(output) else None


def evaluate_ticker(
    df: pd.DataFrame,
    models_root: Path,
    ticker: str,
    decision_date: pd.Timestamp,
) -> dict[str, Any]:
    ticker = ticker.upper()

    config, bundle, weights_path = load_artifacts(models_root, ticker)

    ticker_df = df.loc[
        df["underlyingticker"].eq(ticker)
    ].copy()

    if ticker_df.empty:
        raise ValueError(f"{ticker}: no feature rows")

    scaled_df, scaled_columns = apply_training_scaler(
        ticker_df,
        bundle,
    )

    # Use only the requested completed EOD decision date.
    day_df = scaled_df.loc[
        scaled_df["quotedate"].eq(decision_date)
    ].copy()

    # The short-DTE model must only see contracts in its training universe.
    day_df = day_df.loc[
        day_df["DTE"].between(MIN_ENTRY_DTE, MAX_ENTRY_DTE)
    ].copy()

    # A normal no-trade condition, not an inference failure.
    if day_df.empty:
        return {
            "generated_at_ist": datetime.now(IST).isoformat(),
            "decision_date": decision_date.date().isoformat(),
            "ticker": ticker,
            "decision": "HOLD",
            "direction": None,
            "selected_candidate": None,
            "status": (
                f"NO_ELIGIBLE_SHORT_DTE_CANDIDATES_"
                f"{MIN_ENTRY_DTE}_TO_{MAX_ENTRY_DTE}"
            ),
        }

    state_size = int(
        artifact_get(config, "state_size", "statesize")
    )

    action_size = int(
        artifact_get(config, "action_size", "actionsize")
    )

    max_candidates = int(
        config.get(
            "max_candidates",
            config.get(
                "maxcandidates",
                DEFAULT_MAX_CANDIDATES,
            ),
        )
    )

    expected_action_size = max_candidates + 1

    if action_size != expected_action_size:
        raise ValueError(
            f"{ticker}: action_size={action_size}; "
            f"later mechanical-exit model requires "
            f"{expected_action_size}"
        )

    state, candidates = build_flat_entry_state(
        day_df,
        scaled_columns,
        max_candidates,
    )

    if len(state) != state_size:
        raise ValueError(
            f"{ticker}: constructed state has {len(state)} values; "
            f"saved model expects {state_size}"
        )

    model = build_model(state_size, action_size)
    model.load_weights(weights_path)

    q_values = model.predict(
        state.reshape(1, -1),
        verbose=0,
    )[0]

    valid_actions = [0]

    for index, candidate_row in enumerate(candidates):
        action = index + 1

        dte = pd.to_numeric(
            candidate_row.get("DTE"),
            errors="coerce",
        )

        price = pd.to_numeric(
            candidate_row.get("EODProxyPrice"),
            errors="coerce",
        )

        volume = pd.to_numeric(
            candidate_row.get("optionvolume"),
            errors="coerce",
        )

        if (
            action < action_size
            and pd.notna(dte)
            and MIN_ENTRY_DTE <= dte <= MAX_ENTRY_DTE
            and pd.notna(price)
            and price > 0
            and pd.notna(volume)
            and volume > 0
        ):
            valid_actions.append(action)

    masked_q_values = np.full(
        action_size,
        -np.inf,
        dtype=np.float32,
    )

    for action in valid_actions:
        masked_q_values[action] = q_values[action]

    selected_action = int(
        np.argmax(masked_q_values)
    )

    result: dict[str, Any] = {
        "generated_at_ist": datetime.now(IST).isoformat(),
        "decision_date": decision_date.date().isoformat(),
        "ticker": ticker,
        "model_weights": str(weights_path),
        "state_size": state_size,
        "action_size": action_size,
        "max_candidates": max_candidates,
        "candidate_count": len(candidates),
        "valid_actions": valid_actions,
        "selected_action": selected_action,
        "q_values": [
            finite_float(value)
            for value in q_values
        ],
        "masked_q_values": [
            finite_float(value)
            for value in masked_q_values
        ],
        "take_profit_return": finite_float(
            config.get(
                "take_profit_return",
                config.get("takeprofitreturn"),
            )
        ),
        "training_round_trip_cost": finite_float(
            config.get(
                "round_trip_cost",
                config.get("roundtripcost"),
            )
        ),
        "quantity": int(
            config.get(
                "max_position_contracts",
                config.get("maxpositioncontracts", 1),
            )
        ),
        "hard_flatten_deadline_ist": HARD_FLATTEN_DEADLINE_IST,
        "entry_dte_min": MIN_ENTRY_DTE,
        "entry_dte_max": MAX_ENTRY_DTE,
    }

    if selected_action == 0:
        return {
            **result,
            "decision": "HOLD",
            "direction": None,
            "selected_candidate": None,
            "status": "NO_ENTRY",
        }

    selected = candidates[selected_action - 1]

    contract_type = str(
        selected.get("contracttype", "")
    ).lower()

    if contract_type not in {"call", "put"}:
        raise ValueError(
            f"{ticker}: unexpected selected contract type "
            f"{contract_type!r}"
        )

    expiration = pd.to_datetime(
        selected.get("expirationdate"),
        errors="coerce",
    )

    candidate = {
        "candidate_rank": selected_action,
        "historical_option_symbol": str(
            selected.get("optionsymbol")
        ),
        "contract_type": contract_type,
        "direction": (
            "LONG_CALL"
            if contract_type == "call"
            else "LONG_PUT"
        ),
        "strike_price": finite_float(
            selected.get("strikeprice")
        ),
        "expiration_date": (
            expiration.date().isoformat()
            if pd.notna(expiration)
            else None
        ),
        "reference_dte": finite_float(
            selected.get("DTE")
        ),
        "reference_moneyness": finite_float(
            selected.get("Moneyness")
        ),
        "reference_log_moneyness": finite_float(
            selected.get("LogMoneyness")
        ),
        "eod_proxy_price": finite_float(
            selected.get("EODProxyPrice")
        ),
        "option_volume": finite_float(
            selected.get("optionvolume")
        ),
        "option_dollar_volume": finite_float(
            selected.get("OptionDollarVolume")
        ),
    }

    return {
        **result,
        "decision": "ENTER",
        "direction": candidate["direction"],
        "selected_candidate": candidate,
        "live_selector_constraints": {
            "ticker": ticker,
            "contract_type": contract_type,
            "reference_dte": candidate["reference_dte"],
            "reference_moneyness": candidate[
                "reference_moneyness"
            ],
            "min_entry_dte": MIN_ENTRY_DTE,
            "max_entry_dte": MAX_ENTRY_DTE,
            "max_round_trip_spread_cost": 0.05,
            "quantity": 1,
            "execution_environment": "ALPACA_PAPER",
            "requires_fresh_quote": True,
            "requires_buying_power_check": True,
            "requires_no_duplicate_position_check": True,
        },
        "status": (
            "REQUIRES_LIVE_CHAIN_AND_ACCOUNT_VALIDATION"
        ),
    }

def main() -> None:
    args = parse_args()
    df = load_feature_matrix(args.features)
    decision_date = pd.Timestamp(args.decision_date).normalize()
    available_dates = set(df["quotedate"].dropna().unique())
    if decision_date not in available_dates:
        latest = df["quotedate"].max()
        raise ValueError(f"Decision date {decision_date.date()} absent; latest available is {latest.date() if pd.notna(latest) else None}")

    models_root = Path(args.models_root)
    if not models_root.exists():
        raise FileNotFoundError(f"Models root not found: {models_root}")
    tickers = [ticker.upper() for ticker in args.ticker] if args.ticker else DEFAULT_TICKERS

    results = []
    for ticker in tickers:
        try:
            result = evaluate_ticker(df, models_root, ticker, decision_date)
            results.append(result)
            print(f"{ticker}: {result['decision']}{' ' + result['direction'] if result.get('direction') else ''}")
        except Exception as error:
            result = {
                "generated_at_ist": datetime.now(IST).isoformat(),
                "decision_date": decision_date.date().isoformat(),
                "ticker": ticker,
                "status": "ERROR",
                "error": str(error),
            }
            results.append(result)
            print(f"{ticker}: ERROR — {error}")

    output = {
        "generated_at_ist": datetime.now(IST).isoformat(),
        "decision_date": decision_date.date().isoformat(),
        "mode": "EOD_INFERENCE_ONLY",
        "portfolio_assumption": "FLAT",
        "hard_flatten_deadline_ist": HARD_FLATTEN_DEADLINE_IST,
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    archive = output_dir / f"eod_direction_intents_{decision_date.date()}_{timestamp}.json"
    latest = output_dir / "latest_direction_intents.json"
    for path in [archive, latest]:
        path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved: {latest}")


if __name__ == "__main__":
    main()

