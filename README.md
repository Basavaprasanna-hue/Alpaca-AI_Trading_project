# OptionRelay — RL-First Live Options Validation Agent

OptionRelay is a semi-autonomous options research and Alpaca paper-trading
workflow. A ticker-specific end-of-day Deep Q-Network (DQN) produces
structured `HOLD` or `ENTER` intents from an options feature matrix.

For an `ENTER` signal, the workflow preserves the model’s intended option
type, direction, days to expiration (DTE), moneyness, quantity, and modeled
transaction-cost assumptions. A live Alpaca validation layer then resolves
currently active option contracts and applies quote-quality gates before
human-supervised paper-order authorization.

## Architecture

```text
Historical options feature matrix
        ↓
Ticker-specific EOD DQN inference
        ↓
Structured HOLD / ENTER intent
        ↓
Active Alpaca option-contract discovery
        ↓
DTE + moneyness profile matching
        ↓
Live option quote-quality gates
        ↓
Ranked eligible-contract shortlist
        ↓
Human-supervised Alpaca paper-order authorization
```

## Key capabilities

- Ticker-specific TensorFlow/Keras DQN inference
- Structured `HOLD` / `ENTER` trade intents
- Long-call and long-put reference contract profiles
- Active option-contract discovery through Alpaca
- Historical-to-live matching using contract type, DTE, and moneyness proximity
- Live bid/ask, displayed-size, quote-freshness, debit, and spread-cost gates
- Alpaca MCP integration in VS Code Copilot for account and contract research
- Alert-only long-option position monitoring
- Paper-account audit and trade-performance reporting

## Autonomy boundary

OptionRelay autonomously performs EOD DQN inference, structured intent
generation, active-contract resolution, DTE/moneyness profile matching, and
live quote-quality validation.

Final paper-order authorization remains human-supervised while idempotent
order-state reconciliation and policy-consistent exits are being validated.
The workflow safe-fails on ambiguous order or position state rather than
blindly resubmitting an order.

## Repository contents

```text
src/        Sanitized source code
examples/   Sanitized sample intents and live-contract shortlist outputs
docs/       Architecture, slides, one-page write-up, and demo materials
```

## Data and model artifacts

The public repository contains the submitted reference implementation,
sanitized source code, environment configuration, architecture documentation,
and representative outputs.

Full historical feature data, trained model weights, scaler artifacts, API
credentials, and private account records are intentionally excluded.

## Disclaimer

This repository is provided for educational and research purposes only. It is
a paper-trading prototype and is not investment advice, a solicitation, or a
recommendation to buy or sell any security or option.

Options involve substantial risk and may not be suitable for all investors.
Use of this software is entirely at the user's own risk. Do not use it with
live trading credentials without independent testing, risk controls, and
professional review.

## License and permitted review

Copyright © 2026 [Basavaprasanna Angadi]. All rights reserved.

This repository is publicly available solely to support evaluation of the
OptionRelay hackathon submission. You may view and inspect the repository for
personal, educational, and non-commercial evaluation purposes.

No permission is granted to use, copy, modify, distribute, sublicense, sell,
or otherwise commercially exploit the code, model logic, feature engineering,
documentation, data, or derivative works without prior written permission from
the copyright holder.
