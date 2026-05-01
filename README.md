# risk-threshold-engine

A lightweight, zero-dependency Python library that converts factor scores into a market risk regime, deployment decision, and recommended mitigation actions.

## Overview

Feed in five 0–100 risk scores (volatility, momentum, breadth, macro, drawdown), get back a structured result telling you whether to deploy capital at full pace, reduce lot sizes, pause, or rebalance.

```python
from risk_threshold_engine import RiskThresholdEngine, FactorScores

engine = RiskThresholdEngine()

scores = FactorScores(
    volatility_score=72.0,
    momentum_score=60.0,
    breadth_score=55.0,
    macro_score=65.0,
    drawdown_score=50.0,
)

result = engine.evaluate(scores)
print(result.summary())
# [2026-05-01T...] Risk Engine Result
#   Composite Score : 61.25 / 100
#   Regime          : ELEVATED
#   Decision        : REDUCED_DCA
#   ...
```

## Installation

```bash
pip install risk-threshold-engine
```

Or install from source:

```bash
git clone https://github.com/BrainScanStudios/risk-threshold-engine.git
cd risk-threshold-engine
pip install -e .
```

## Requirements

Python 3.10+. No external dependencies — only the standard library.

## Risk Regimes

| Composite Score | Regime   | Default Decision |
|-----------------|----------|------------------|
| 0 – 30          | LOW      | FULL_DCA         |
| 31 – 55         | MODERATE | REDUCED_DCA      |
| 56 – 75         | ELEVATED | REDUCED_DCA / PAUSE* |
| 76 – 100        | SEVERE   | REBALANCE        |

*ELEVATED switches to PAUSE when the portfolio's cash/bond buffer exceeds 30%.

## Default Factor Weights

| Factor            | Weight | What it captures                  |
|-------------------|--------|-----------------------------------|
| volatility_score  | 25%    | Realised vol / VIX signal         |
| momentum_score    | 20%    | Price momentum across assets      |
| breadth_score     | 15%    | Market breadth (advance/decline)  |
| macro_score       | 20%    | Rates, credit spreads, inflation  |
| drawdown_score    | 20%    | Drawdown vs high-water mark       |

All weights are configurable. Scores are 0–100 where **higher = more risk**.

## API

### `RiskThresholdEngine`

```python
RiskThresholdEngine(
    factor_weights: dict[str, float] | None = None,
    ticker_beta:    dict[str, float] | None = None,
    db_path:        str | None = None,
)
```

- **`factor_weights`** — Override the default weight map. Must sum to 1.0.
- **`ticker_beta`** — Map of ticker → beta coefficient. When provided, the engine generates `TRIM` and `OPTIONS_SIGNAL` actions for high-beta tickers held in the portfolio during elevated/severe regimes.
- **`db_path`** — Optional path to a SQLite file. When set, every `evaluate()` call persists results to a `risk_engine_results` table.

#### `evaluate(scores, portfolio=None) -> EngineResult`

Runs the full evaluation pipeline.

### `FactorScores`

```python
FactorScores(
    volatility_score: float,   # 0–100
    momentum_score:   float,   # 0–100
    breadth_score:    float,   # 0–100
    macro_score:      float,   # 0–100
    drawdown_score:   float,   # 0–100
)
```

All five scores are required and must be in [0, 100]. How you produce them is up to you — any factor scoring module works.

### `PortfolioState` (optional)

```python
PortfolioState(
    buffer_pct:   float,              # cash/short-term bond % of portfolio
    cash_usd:   float,
    positions:  dict[str, float],   # ticker → market value in USD
)
```

Passing a `PortfolioState` enables portfolio-aware decisions (buffer checks, beta-weighted trim candidates, options signals).

### `EngineResult`

| Field                | Type                  | Description                            |
|----------------------|-----------------------|----------------------------------------|
| `composite_score`    | `float`               | Weighted composite (0–100)             |
| `regime`             | `RiskRegime`          | LOW / MODERATE / ELEVATED / SEVERE     |
| `decision`           | `DeploymentDecision`  | FULL_DCA / REDUCED_DCA / PAUSE / REBALANCE |
| `actions`            | `list[MitigationAction]` | Ordered list of recommended actions |
| `weighted_breakdown` | `dict[str, float]`    | Per-factor contribution to composite   |
| `notes`              | `str`                 | Contextual caveats                     |
| `evaluated_at`       | `str`                 | ISO 8601 UTC timestamp                 |

```python
result.to_dict()    # fully serialisable dict
result.summary()    # human-readable string
```

## Custom Configuration

```python
engine = RiskThresholdEngine(
    factor_weights={
        "volatility_score": 0.40,
        "momentum_score":   0.20,
        "breadth_score":    0.10,
        "macro_score":      0.20,
        "drawdown_score":   0.10,
    },
    ticker_beta={
        "QQQ":  1.20,
        "SPY":  1.00,
        "TLT":  0.30,
    },
    db_path="risk_results.db",
)
```

## Development

Clone the repo and set up an isolated environment:

```bash
git clone https://github.com/BrainScanStudios/risk-threshold-engine.git
cd risk-threshold-engine

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

The `[dev]` extra installs `pytest`. The library itself has no runtime dependencies beyond the standard library.

```bash
pytest
pytest -v          # verbose
pytest -k regime   # run tests matching a keyword
```

### Project layout

```
src/
    risk_threshold_engine/
        __init__.py    public API surface
        engine.py      all core logic lives here
tests/
    test_engine.py     full test suite
```

The entire implementation is in `engine.py`. If you're adding a new action type or regime, the three methods to touch are `_build_actions`, `_decide`, and `_contextual_notes`. If you're changing the scoring model, adjust `DEFAULT_FACTOR_WEIGHTS` or pass custom weights at construction time.

### Submitting changes

1. Fork the repository and create a branch from `main`.
2. Add or update tests for any behaviour you change — `pytest` must pass before opening a PR.
3. Keep the zero-dependency constraint: no new imports outside the standard library.

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 Brain Scan Studios
