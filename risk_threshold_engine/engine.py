"""
Consumes factor scores from any scoring module and outputs a deployment
decision, regime label, and recommended rebalancing actions.

Usage:
    from risk_threshold_engine import RiskThresholdEngine, FactorScores

    engine = RiskThresholdEngine()
    result = engine.evaluate(factor_scores)
    print(result.summary())
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class RiskRegime(str, Enum):
    LOW       = "LOW"        # 0–30
    MODERATE  = "MODERATE"   # 31–55
    ELEVATED  = "ELEVATED"   # 56–75
    SEVERE    = "SEVERE"     # 76–100


class DeploymentDecision(str, Enum):
    FULL_DCA     = "FULL_DCA"       # Deploy at normal cadence / lot size
    REDUCED_DCA  = "REDUCED_DCA"    # Cut lot size
    PAUSE        = "PAUSE"          # Stop new equity buys
    REBALANCE    = "REBALANCE"      # Rotate existing positions; no new buys


# Thresholds (inclusive lower bound)
REGIME_THRESHOLDS: list[tuple[int, RiskRegime]] = [
    (76, RiskRegime.SEVERE),
    (56, RiskRegime.ELEVATED),
    (31, RiskRegime.MODERATE),
    (0,  RiskRegime.LOW),
]

# Factor weights – must sum to 1.0
DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "volatility_score": 0.25,
    "momentum_score":   0.20,
    "breadth_score":    0.15,
    "macro_score":      0.20,
    "drawdown_score":   0.20,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FactorScores:
    """
    Input scores from your factor scoring module.
    All values are 0–100 where higher = more risk.
    """
    volatility_score: float
    momentum_score:   float
    breadth_score:    float
    macro_score:      float
    drawdown_score:   float
    as_of:            str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate(self) -> None:
        for name, val in asdict(self).items():
            if name == "as_of":
                continue
            if not (0.0 <= val <= 100.0):
                raise ValueError(f"Factor '{name}' out of range [0,100]: {val}")


@dataclass
class PortfolioState:
    """
    Snapshot of current portfolio positions.
    buffer_pct: short-term bond / cash-equivalent as % of total portfolio (0–100).
    positions: ticker → market value in USD.
    """
    buffer_pct:    float
    cash_usd:    float
    positions:   dict[str, float] = field(default_factory=dict)
    total_value: float = 0.0

    def __post_init__(self):
        if not self.total_value and self.positions:
            self.total_value = sum(self.positions.values()) + self.cash_usd


@dataclass
class MitigationAction:
    action_type: str           # e.g. "TRIM", "ADD_BUFFER", "PAUSE_DCA"
    ticker:      Optional[str]
    rationale:   str
    magnitude:   Optional[str] = None   # e.g. "20%", "50% of lot"


@dataclass
class EngineResult:
    composite_score:    float
    regime:             RiskRegime
    decision:           DeploymentDecision
    actions:            list[MitigationAction]
    weighted_breakdown: dict[str, float]
    evaluated_at:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes:              str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regime"]   = self.regime.value
        d["decision"] = self.decision.value
        return d

    def summary(self) -> str:
        lines = [
            f"[{self.evaluated_at}] Risk Engine Result",
            f"  Composite Score : {self.composite_score:.1f} / 100",
            f"  Regime          : {self.regime.value}",
            f"  Decision        : {self.decision.value}",
            f"  Notes           : {self.notes}",
            "  Actions:",
        ]
        for a in self.actions:
            ticker_str = f" [{a.ticker}]" if a.ticker else ""
            mag_str    = f" ({a.magnitude})" if a.magnitude else ""
            lines.append(f"    * {a.action_type}{ticker_str}{mag_str} – {a.rationale}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RiskThresholdEngine:
    """
    Evaluates factor scores against configurable risk thresholds and emits
    a deployment decision with recommended mitigation actions.

        factor_scores -> RiskThresholdEngine.evaluate() -> EngineResult

    Parameters
    ----------
    factor_weights:
        Dict mapping factor name to weight. Must sum to 1.0.
        Defaults to DEFAULT_FACTOR_WEIGHTS.
    ticker_beta:
        Dict mapping ticker symbol to beta coefficient. Used to rank
        trim candidates in ELEVATED/SEVERE regimes. Pass an empty dict
        (default) to skip beta-weighted trim recommendations.
    db_path:
        Optional path to a SQLite file for result persistence.
    """

    def __init__(
        self,
        factor_weights: Optional[dict[str, float]] = None,
        ticker_beta:    Optional[dict[str, float]] = None,
        db_path:        Optional[str] = None,
    ):
        self.weights     = factor_weights or DEFAULT_FACTOR_WEIGHTS
        self.ticker_beta = ticker_beta or {}
        self._validate_weights()
        self.db_path = db_path

        if self.db_path:
            self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        scores:    FactorScores,
        portfolio: Optional[PortfolioState] = None,
    ) -> EngineResult:
        """
        Main entry point. Returns an EngineResult with regime, decision,
        and recommended actions.
        """
        scores.validate()

        composite, breakdown = self._compute_composite(scores)
        regime               = self._classify_regime(composite)
        decision             = self._decide(regime, portfolio)
        actions              = self._build_actions(regime, portfolio, scores)
        notes                = self._contextual_notes(regime, portfolio)

        result = EngineResult(
            composite_score    = round(composite, 2),
            regime             = regime,
            decision           = decision,
            actions            = actions,
            weighted_breakdown = breakdown,
            notes              = notes,
        )

        if self.db_path:
            self._persist(result)

        return result

    # ------------------------------------------------------------------
    # Internal: scoring
    # ------------------------------------------------------------------

    def _compute_composite(
        self, scores: FactorScores
    ) -> tuple[float, dict[str, float]]:
        score_map = {
            "volatility_score": scores.volatility_score,
            "momentum_score":   scores.momentum_score,
            "breadth_score":    scores.breadth_score,
            "macro_score":      scores.macro_score,
            "drawdown_score":   scores.drawdown_score,
        }
        breakdown: dict[str, float] = {}
        composite = 0.0
        for factor, weight in self.weights.items():
            contribution      = score_map[factor] * weight
            breakdown[factor] = round(contribution, 3)
            composite        += contribution
        return composite, breakdown

    def _classify_regime(self, score: float) -> RiskRegime:
        for threshold, regime in REGIME_THRESHOLDS:
            if score >= threshold:
                return regime
        return RiskRegime.LOW

    # ------------------------------------------------------------------
    # Internal: decision logic
    # ------------------------------------------------------------------

    def _decide(
        self,
        regime:    RiskRegime,
        portfolio: Optional[PortfolioState],
    ) -> DeploymentDecision:
        if regime == RiskRegime.SEVERE:
            return DeploymentDecision.REBALANCE

        if regime == RiskRegime.ELEVATED:
            if portfolio and portfolio.buffer_pct >= 30:
                return DeploymentDecision.PAUSE
            return DeploymentDecision.REDUCED_DCA

        if regime == RiskRegime.MODERATE:
            return DeploymentDecision.REDUCED_DCA

        return DeploymentDecision.FULL_DCA

    # ------------------------------------------------------------------
    # Internal: action generation
    # ------------------------------------------------------------------

    def _build_actions(
        self,
        regime:    RiskRegime,
        portfolio: Optional[PortfolioState],
        scores:    FactorScores,
    ) -> list[MitigationAction]:
        actions: list[MitigationAction] = []

        if regime == RiskRegime.LOW:
            actions.append(MitigationAction(
                action_type = "FULL_DCA",
                ticker      = None,
                rationale   = "Risk environment benign; deploy at full cadence.",
            ))
            return actions

        if regime in (RiskRegime.MODERATE, RiskRegime.ELEVATED):
            actions.append(MitigationAction(
                action_type = "REDUCE_LOT_SIZE",
                ticker      = None,
                magnitude   = "50%" if regime == RiskRegime.ELEVATED else "25%",
                rationale   = "Elevated risk warrants smaller new-buy lots.",
            ))

        # Beta-weighted trim candidates (only if ticker_beta was provided)
        if regime in (RiskRegime.ELEVATED, RiskRegime.SEVERE) and self.ticker_beta:
            for ticker, beta in sorted(self.ticker_beta.items(), key=lambda x: -x[1]):
                if beta >= 1.0 and portfolio and portfolio.positions.get(ticker, 0) > 0:
                    actions.append(MitigationAction(
                        action_type = "TRIM",
                        ticker      = ticker,
                        magnitude   = "10–20%",
                        rationale   = f"Beta {beta:.2f} — trim to reduce portfolio vol.",
                    ))

        # Cash / short-term bond buffer
        if portfolio and portfolio.buffer_pct < 20 and regime != RiskRegime.LOW:
            actions.append(MitigationAction(
                action_type = "ADD_BUFFER",
                ticker      = None,
                rationale   = (
                    f"Cash buffer low ({portfolio.buffer_pct:.1f}% of portfolio). "
                    "Route DCA surplus to short-term bonds or cash equivalents."
                ),
            ))

        # Volatility-driven options signal
        if scores.volatility_score >= 60 and portfolio:
            high_beta = [t for t, b in self.ticker_beta.items() if b >= 1.0]
            held_high_beta = [t for t in high_beta if portfolio.positions.get(t, 0) > 0]
            for ticker in held_high_beta:
                actions.append(MitigationAction(
                    action_type = "OPTIONS_SIGNAL",
                    ticker      = ticker,
                    rationale   = (
                        f"Elevated vol (score {scores.volatility_score:.0f}) — consider "
                        f"covered calls on {ticker} to harvest premium while hedging downside."
                    ),
                ))

        if regime == RiskRegime.SEVERE:
            actions.append(MitigationAction(
                action_type = "PAUSE_DCA",
                ticker      = None,
                rationale   = "Severe risk regime. Halt all new equity buys.",
            ))

        return actions

    def _contextual_notes(
        self,
        regime:    RiskRegime,
        portfolio: Optional[PortfolioState],
    ) -> str:
        notes = []
        if portfolio:
            if portfolio.buffer_pct > 40:
                notes.append(
                    f"Buffer is {portfolio.buffer_pct:.1f}% of portfolio — "
                    "consider whether excess cash is opportunity cost."
                )
            if portfolio.buffer_pct < 10 and regime != RiskRegime.LOW:
                notes.append("Cash buffer critically low relative to risk regime.")
        return " | ".join(notes)

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_engine_results (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluated_at    TEXT,
                    composite_score REAL,
                    regime          TEXT,
                    decision        TEXT,
                    breakdown_json  TEXT,
                    actions_json    TEXT,
                    notes           TEXT
                )
            """)

    def _persist(self, result: EngineResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO risk_engine_results
                    (evaluated_at, composite_score, regime, decision,
                     breakdown_json, actions_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.evaluated_at,
                result.composite_score,
                result.regime.value,
                result.decision.value,
                json.dumps(result.weighted_breakdown),
                json.dumps([asdict(a) for a in result.actions]),
                result.notes,
            ))

    # ------------------------------------------------------------------
    # Internal: validation
    # ------------------------------------------------------------------

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Factor weights must sum to 1.0, got {total:.4f}")


# ---------------------------------------------------------------------------
# Runnable example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = RiskThresholdEngine(
        ticker_beta={
            "ACME_GROWTH": 1.30,
            "ACME_VALUE":  0.80,
        },
        db_path="risk_engine_demo.db",
    )

    scores = FactorScores(
        volatility_score = 72.0,
        momentum_score   = 60.0,
        breadth_score    = 55.0,
        macro_score      = 65.0,
        drawdown_score   = 50.0,
    )

    portfolio = PortfolioState(
        buffer_pct  = 18.5,
        cash_usd  = 2_500.0,
        positions = {
            "ACME_GROWTH": 22_000.0,
            "ACME_VALUE":  15_000.0,
        },
    )

    result = engine.evaluate(scores, portfolio)
    print(result.summary())
    print("\nFull dict:")
    print(json.dumps(result.to_dict(), indent=2))
