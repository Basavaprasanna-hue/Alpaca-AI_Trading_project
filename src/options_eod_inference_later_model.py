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
