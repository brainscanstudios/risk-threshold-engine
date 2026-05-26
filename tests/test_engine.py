import json
import os
import tempfile

import pytest

from risk_threshold_engine import (
    DeploymentDecision,
    EngineResult,
    FactorScores,
    MitigationAction,
    PortfolioState,
    RiskRegime,
    RiskThresholdEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def low_scores(**overrides) -> FactorScores:
    defaults = dict(volatility_score=10.0, momentum_score=10.0, breadth_score=10.0, macro_score=10.0, drawdown_score=10.0)
    return FactorScores(**{**defaults, **overrides})


def moderate_scores(**overrides) -> FactorScores:
    defaults = dict(volatility_score=40.0, momentum_score=40.0, breadth_score=40.0, macro_score=40.0, drawdown_score=40.0)
    return FactorScores(**{**defaults, **overrides})


def elevated_scores(**overrides) -> FactorScores:
    defaults = dict(volatility_score=65.0, momentum_score=65.0, breadth_score=65.0, macro_score=65.0, drawdown_score=65.0)
    return FactorScores(**{**defaults, **overrides})


def severe_scores(**overrides) -> FactorScores:
    defaults = dict(volatility_score=85.0, momentum_score=85.0, breadth_score=85.0, macro_score=85.0, drawdown_score=85.0)
    return FactorScores(**{**defaults, **overrides})


def basic_portfolio(**overrides) -> PortfolioState:
    defaults = dict(buffer_pct=20.0, cash_usd=1_000.0, positions={"AAAA": 10_000.0, "BBBB": 5_000.0})
    return PortfolioState(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# FactorScores validation
# ---------------------------------------------------------------------------

class TestFactorScoresValidation:
    def test_valid_scores_pass(self):
        low_scores().validate()

    def test_boundary_zero_passes(self):
        FactorScores(
            volatility_score=0.0,
            momentum_score=0.0,
            breadth_score=0.0,
            macro_score=0.0,
            drawdown_score=0.0,
        ).validate()

    def test_boundary_hundred_passes(self):
        FactorScores(
            volatility_score=100.0,
            momentum_score=100.0,
            breadth_score=100.0,
            macro_score=100.0,
            drawdown_score=100.0,
        ).validate()

    def test_negative_score_raises(self):
        with pytest.raises(ValueError, match="volatility_score"):
            FactorScores(
                volatility_score=-1.0,
                momentum_score=50.0,
                breadth_score=50.0,
                macro_score=50.0,
                drawdown_score=50.0,
            ).validate()

    def test_over_hundred_raises(self):
        with pytest.raises(ValueError, match="macro_score"):
            FactorScores(
                volatility_score=50.0,
                momentum_score=50.0,
                breadth_score=50.0,
                macro_score=101.0,
                drawdown_score=50.0,
            ).validate()


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------

class TestWeightValidation:
    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            RiskThresholdEngine(factor_weights={"volatility_score": 0.5, "momentum_score": 0.3})

    def test_custom_valid_weights_accepted(self):
        engine = RiskThresholdEngine(
            factor_weights={
                "volatility_score": 0.20,
                "momentum_score":   0.20,
                "breadth_score":    0.20,
                "macro_score":      0.20,
                "drawdown_score":   0.20,
            }
        )
        assert engine.weights["volatility_score"] == 0.20


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

class TestRegimeClassification:
    def setup_method(self):
        self.engine = RiskThresholdEngine()

    def test_score_0_is_low(self):
        assert self.engine._classify_regime(0.0) == RiskRegime.LOW

    def test_score_30_is_low(self):
        assert self.engine._classify_regime(30.0) == RiskRegime.LOW

    def test_score_31_is_moderate(self):
        assert self.engine._classify_regime(31.0) == RiskRegime.MODERATE

    def test_score_55_is_moderate(self):
        assert self.engine._classify_regime(55.0) == RiskRegime.MODERATE

    def test_score_56_is_elevated(self):
        assert self.engine._classify_regime(56.0) == RiskRegime.ELEVATED

    def test_score_75_is_elevated(self):
        assert self.engine._classify_regime(75.0) == RiskRegime.ELEVATED

    def test_score_76_is_severe(self):
        assert self.engine._classify_regime(76.0) == RiskRegime.SEVERE

    def test_score_100_is_severe(self):
        assert self.engine._classify_regime(100.0) == RiskRegime.SEVERE


# ---------------------------------------------------------------------------
# Composite computation
# ---------------------------------------------------------------------------

class TestCompositeComputation:
    def test_uniform_scores_equal_weighted_average(self):
        engine = RiskThresholdEngine()
        scores = FactorScores(
            volatility_score=50.0,
            momentum_score=50.0,
            breadth_score=50.0,
            macro_score=50.0,
            drawdown_score=50.0,
        )
        composite, _ = engine._compute_composite(scores)
        assert abs(composite - 50.0) < 1e-6

    def test_breakdown_contributions_sum_to_composite(self):
        engine = RiskThresholdEngine()
        scores = moderate_scores()
        composite, breakdown = engine._compute_composite(scores)
        assert abs(sum(breakdown.values()) - composite) < 1e-4

    def test_custom_weights_change_composite(self):
        weights = {
            "volatility_score": 1.0,
            "momentum_score":   0.0,
            "breadth_score":    0.0,
            "macro_score":      0.0,
            "drawdown_score":   0.0,
        }
        engine = RiskThresholdEngine(factor_weights=weights)
        scores = FactorScores(
            volatility_score=80.0,
            momentum_score=10.0,
            breadth_score=10.0,
            macro_score=10.0,
            drawdown_score=10.0,
        )
        composite, _ = engine._compute_composite(scores)
        assert abs(composite - 80.0) < 1e-6


# ---------------------------------------------------------------------------
# Deployment decision
# ---------------------------------------------------------------------------

class TestDeploymentDecision:
    def setup_method(self):
        self.engine = RiskThresholdEngine()

    def test_low_regime_full_dca(self):
        result = self.engine.evaluate(low_scores())
        assert result.decision == DeploymentDecision.FULL_DCA

    def test_moderate_regime_reduced_dca(self):
        result = self.engine.evaluate(moderate_scores())
        assert result.decision == DeploymentDecision.REDUCED_DCA

    def test_elevated_regime_reduced_dca_when_buffer_low(self):
        portfolio = basic_portfolio(buffer_pct=10.0)
        result = self.engine.evaluate(elevated_scores(), portfolio)
        assert result.decision == DeploymentDecision.REDUCED_DCA

    def test_elevated_regime_pause_when_buffer_high(self):
        portfolio = basic_portfolio(buffer_pct=35.0)
        result = self.engine.evaluate(elevated_scores(), portfolio)
        assert result.decision == DeploymentDecision.PAUSE

    def test_severe_regime_rebalance(self):
        result = self.engine.evaluate(severe_scores())
        assert result.decision == DeploymentDecision.REBALANCE

    def test_elevated_without_portfolio_returns_reduced_dca(self):
        result = self.engine.evaluate(elevated_scores(), portfolio=None)
        assert result.decision == DeploymentDecision.REDUCED_DCA


# ---------------------------------------------------------------------------
# Action generation
# ---------------------------------------------------------------------------

class TestActionGeneration:
    def setup_method(self):
        self.engine = RiskThresholdEngine()

    def _action_types(self, result: EngineResult) -> list[str]:
        return [a.action_type for a in result.actions]

    def test_low_regime_only_full_dca_action(self):
        result = self.engine.evaluate(low_scores())
        assert self._action_types(result) == ["FULL_DCA"]

    def test_moderate_regime_includes_reduce_lot_size(self):
        result = self.engine.evaluate(moderate_scores())
        assert "REDUCE_LOT_SIZE" in self._action_types(result)

    def test_elevated_regime_reduce_lot_size_is_50_pct(self):
        result = self.engine.evaluate(elevated_scores())
        reduce = next(a for a in result.actions if a.action_type == "REDUCE_LOT_SIZE")
        assert reduce.magnitude == "50%"

    def test_moderate_regime_reduce_lot_size_is_25_pct(self):
        result = self.engine.evaluate(moderate_scores())
        reduce = next(a for a in result.actions if a.action_type == "REDUCE_LOT_SIZE")
        assert reduce.magnitude == "25%"

    def test_severe_regime_includes_pause_dca(self):
        result = self.engine.evaluate(severe_scores())
        assert "PAUSE_DCA" in self._action_types(result)

    def test_low_buffer_triggers_add_buffer(self):
        portfolio = basic_portfolio(buffer_pct=10.0)
        result = self.engine.evaluate(moderate_scores(), portfolio)
        assert "ADD_BUFFER" in self._action_types(result)

    def test_adequate_buffer_no_add_buffer_action(self):
        portfolio = basic_portfolio(buffer_pct=25.0)
        result = self.engine.evaluate(moderate_scores(), portfolio)
        assert "ADD_BUFFER" not in self._action_types(result)

    def test_no_trim_when_ticker_beta_empty(self):
        portfolio = basic_portfolio()
        result = self.engine.evaluate(elevated_scores(), portfolio)
        assert "TRIM" not in self._action_types(result)

    def test_trim_generated_for_high_beta_tickers(self):
        engine = RiskThresholdEngine(ticker_beta={"AAAA": 1.5, "BBBB": 0.6})
        portfolio = PortfolioState(
            buffer_pct=20.0,
            cash_usd=1_000.0,
            positions={"AAAA": 10_000.0, "BBBB": 5_000.0},
        )
        result = engine.evaluate(elevated_scores(), portfolio)
        trim_tickers = [a.ticker for a in result.actions if a.action_type == "TRIM"]
        assert "AAAA" in trim_tickers
        assert "BBBB" not in trim_tickers

    def test_trim_not_generated_for_zero_position(self):
        engine = RiskThresholdEngine(ticker_beta={"AAAA": 1.5})
        portfolio = PortfolioState(
            buffer_pct=20.0,
            cash_usd=1_000.0,
            positions={"BBBB": 5_000.0},
        )
        result = engine.evaluate(elevated_scores(), portfolio)
        assert "TRIM" not in self._action_types(result)

    def test_options_signal_for_high_vol_and_held_high_beta(self):
        engine = RiskThresholdEngine(ticker_beta={"AAAA": 1.5})
        portfolio = PortfolioState(
            buffer_pct=20.0,
            cash_usd=500.0,
            positions={"AAAA": 10_000.0},
        )
        scores = moderate_scores(volatility_score=65.0)
        result = engine.evaluate(scores, portfolio)
        assert "OPTIONS_SIGNAL" in self._action_types(result)

    def test_no_options_signal_below_vol_threshold(self):
        engine = RiskThresholdEngine(ticker_beta={"AAAA": 1.5})
        portfolio = basic_portfolio(positions={"AAAA": 10_000.0})
        scores = moderate_scores(volatility_score=50.0)
        result = engine.evaluate(scores, portfolio)
        assert "OPTIONS_SIGNAL" not in self._action_types(result)


# ---------------------------------------------------------------------------
# Contextual notes
# ---------------------------------------------------------------------------

class TestContextualNotes:
    def setup_method(self):
        self.engine = RiskThresholdEngine()

    def test_no_notes_without_portfolio(self):
        result = self.engine.evaluate(moderate_scores())
        assert result.notes == ""

    def test_high_buffer_note(self):
        portfolio = basic_portfolio(buffer_pct=45.0)
        result = self.engine.evaluate(moderate_scores(), portfolio)
        assert "opportunity cost" in result.notes

    def test_critically_low_buffer_note(self):
        portfolio = basic_portfolio(buffer_pct=5.0)
        result = self.engine.evaluate(moderate_scores(), portfolio)
        assert "critically low" in result.notes

    def test_no_critical_note_in_low_regime(self):
        portfolio = basic_portfolio(buffer_pct=5.0)
        result = self.engine.evaluate(low_scores(), portfolio)
        assert "critically low" not in result.notes


# ---------------------------------------------------------------------------
# End-to-end evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def setup_method(self):
        self.engine = RiskThresholdEngine()

    def test_returns_engine_result(self):
        result = self.engine.evaluate(low_scores())
        assert isinstance(result, EngineResult)

    def test_composite_score_within_range(self):
        result = self.engine.evaluate(moderate_scores())
        assert 0.0 <= result.composite_score <= 100.0

    def test_to_dict_serialisable(self):
        result = self.engine.evaluate(elevated_scores(), basic_portfolio())
        d = result.to_dict()
        json.dumps(d)  # must not raise

    def test_to_dict_regime_is_string(self):
        result = self.engine.evaluate(low_scores())
        assert isinstance(result.to_dict()["regime"], str)

    def test_summary_contains_regime(self):
        result = self.engine.evaluate(severe_scores())
        assert "SEVERE" in result.summary()

    def test_evaluate_without_portfolio(self):
        result = self.engine.evaluate(low_scores(), portfolio=None)
        assert result.regime == RiskRegime.LOW

    def test_portfolio_total_value_auto_computed(self):
        portfolio = PortfolioState(
            buffer_pct=20.0,
            cash_usd=1_000.0,
            positions={"AAAA": 9_000.0},
        )
        assert portfolio.total_value == 10_000.0


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_results_written_to_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            engine = RiskThresholdEngine(db_path=db_path)
            engine.evaluate(moderate_scores())
            engine.evaluate(severe_scores())

            import sqlite3
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute("SELECT * FROM risk_engine_results").fetchall()
            assert len(rows) == 2
        finally:
            os.unlink(db_path)

    def test_no_persistence_without_db_path(self):
        engine = RiskThresholdEngine()
        result = engine.evaluate(low_scores())
        assert isinstance(result, EngineResult)

    def test_db_row_contains_correct_regime(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            engine = RiskThresholdEngine(db_path=db_path)
            engine.evaluate(severe_scores())

            import sqlite3
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT regime FROM risk_engine_results"
                ).fetchone()
            assert row[0] == "SEVERE"
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Threshold parameter overrides
# ---------------------------------------------------------------------------

class TestThresholdOverrides:
    def test_custom_severe_threshold_changes_regime(self):
        engine = RiskThresholdEngine(severe_threshold=90.0)
        # Score of 80 is SEVERE with default (76), but ELEVATED with threshold=90
        assert engine._classify_regime(80.0) == RiskRegime.ELEVATED

    def test_custom_elevated_threshold_changes_regime(self):
        engine = RiskThresholdEngine(elevated_threshold=70.0)
        # Score of 60 is ELEVATED with default (56), but MODERATE with threshold=70
        assert engine._classify_regime(60.0) == RiskRegime.MODERATE

    def test_custom_moderate_threshold_changes_regime(self):
        engine = RiskThresholdEngine(moderate_threshold=35.0)
        # Score of 32 is MODERATE with default (31), but LOW with raised threshold=35
        assert engine._classify_regime(32.0) == RiskRegime.LOW

    def test_evaluate_uses_custom_thresholds(self):
        # Raise severe threshold so a normally-SEVERE score lands in ELEVATED
        engine = RiskThresholdEngine(severe_threshold=90.0)
        result = engine.evaluate(severe_scores())  # composite ~85
        assert result.regime == RiskRegime.ELEVATED

    def test_threshold_attributes_stored(self):
        engine = RiskThresholdEngine(
            severe_threshold=80.0,
            elevated_threshold=60.0,
            moderate_threshold=35.0,
            hy_weight=0.70,
            curve_weight=0.30,
            hy_floor=1.5,
            hy_range=10.0,
            curve_normal=2.0,
        )
        assert engine.severe_threshold   == 80.0
        assert engine.elevated_threshold == 60.0
        assert engine.moderate_threshold == 35.0
        assert engine.hy_weight          == 0.70
        assert engine.curve_weight       == 0.30
        assert engine.hy_floor           == 1.5
        assert engine.hy_range           == 10.0
        assert engine.curve_normal       == 2.0

    def test_unknown_kwargs_absorbed_gracefully(self):
        engine = RiskThresholdEngine(unknown_future_param=42)
        assert engine._classify_regime(50.0) == RiskRegime.MODERATE

    def test_default_thresholds_match_original_boundaries(self):
        engine = RiskThresholdEngine()
        assert engine.severe_threshold   == 76.0
        assert engine.elevated_threshold == 56.0
        assert engine.moderate_threshold == 31.0


# ---------------------------------------------------------------------------
# near_threshold_info()
# ---------------------------------------------------------------------------

class TestNearThresholdInfo:
    def setup_method(self):
        self.engine = RiskThresholdEngine()

    def test_below_moderate_points_to_moderate(self):
        info = self.engine.near_threshold_info(20.0)
        assert info["next_regime"]    == "MODERATE"
        assert info["next_boundary"]  == 31.0
        assert abs(info["pts_to_next_boundary"] - 11.0) < 0.1

    def test_in_moderate_points_to_elevated(self):
        info = self.engine.near_threshold_info(45.0)
        assert info["next_regime"]   == "ELEVATED"
        assert info["next_boundary"] == 56.0

    def test_in_elevated_points_to_severe(self):
        info = self.engine.near_threshold_info(65.0)
        assert info["next_regime"]   == "SEVERE"
        assert info["next_boundary"] == 76.0
        assert abs(info["pts_to_next_boundary"] - 11.0) < 0.1

    def test_at_or_above_severe_no_next_boundary(self):
        info = self.engine.near_threshold_info(80.0)
        assert info["next_boundary"]        is None
        assert info["next_regime"]          is None
        assert info["pts_to_next_boundary"] is None

    def test_composite_score_rounded(self):
        info = self.engine.near_threshold_info(45.678)
        assert info["composite_score"] == 45.7

    def test_hot_factors_included_when_scores_provided(self):
        scores = FactorScores(
            volatility_score=70.0,
            momentum_score=50.0,
            breadth_score=80.0,
            macro_score=40.0,
            drawdown_score=30.0,
        )
        info = self.engine.near_threshold_info(45.0, risk_factor_scores=scores)
        assert "hot_factors" in info
        assert "volatility_score" in info["hot_factors"]
        assert "breadth_score"    in info["hot_factors"]
        assert "momentum_score"   not in info["hot_factors"]

    def test_no_hot_factors_without_scores(self):
        info = self.engine.near_threshold_info(45.0)
        assert "hot_factors" not in info

    def test_uses_custom_thresholds(self):
        engine = RiskThresholdEngine(severe_threshold=90.0, elevated_threshold=70.0)
        info = engine.near_threshold_info(65.0)
        # With default thresholds 65 → next is SEVERE at 76; with custom → next is ELEVATED at 70
        assert info["next_regime"]   == "ELEVATED"
        assert info["next_boundary"] == 70.0
