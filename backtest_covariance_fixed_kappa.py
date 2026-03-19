import numpy as np
import pandas as pd
import warnings
import cvxpy as cp

# 1. Import the core engine blocks
from bl_backtest_engine import (
    BacktestConfig,
    prepare_data,
    run_single_strategy_backtest,
    build_equal_weight_benchmark,
    evaluate_strategies,
    export_results,
    format_metrics_table
)

warnings.filterwarnings("ignore")


# =============================================================================
# COVARIANCE ESTIMATION FUNCTIONS
# =============================================================================

def make_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(symmetric))
    if min_eig < epsilon:
        symmetric = symmetric + np.eye(symmetric.shape[0]) * (epsilon - min_eig)
    return symmetric


def sample_covariance(train_returns: pd.DataFrame) -> np.ndarray:
    return make_psd(train_returns.cov().values)


def shrinkage_covariance(train_returns: pd.DataFrame, shrinkage_strength: float = 0.35) -> np.ndarray:
    sample = sample_covariance(train_returns)
    n_assets = sample.shape[0]
    target_scale = np.trace(sample) / n_assets
    target = target_scale * np.eye(n_assets)

    sigma = cp.Variable((n_assets, n_assets), PSD=True)
    objective = cp.Minimize(
        cp.sum_squares(sigma - sample)
        + shrinkage_strength * cp.sum_squares(sigma - target)
    )
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS, verbose=False)

    if sigma.value is None:
        raise ValueError("cvxpy failed to estimate the shrinkage covariance matrix.")

    return make_psd(np.asarray(sigma.value))

def l2_regularized_covariance(train_returns: pd.DataFrame, l2_penalty: float = 0.1) -> np.ndarray:
    sample = sample_covariance(train_returns)
    n_assets = sample.shape[0]
    identity = np.eye(n_assets)
    sigma = cp.Variable((n_assets, n_assets), PSD=True)
    objective = cp.Minimize(
        cp.sum_squares(sigma - sample) + l2_penalty * cp.sum_squares(sigma @ identity)
    )
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS, verbose=False)

    if sigma.value is None:
        raise ValueError("cvxpy failed to estimate the L2-regularized covariance matrix.")

    sigma_l2 = np.asarray(sigma.value) + l2_penalty * identity
    return make_psd(sigma_l2)


# =============================================================================
# MAIN EXPERIMENT RUNNER (FIXED KAPPA = 0.25)
# =============================================================================

def run_fixed_kappa_covariance_experiment():
    config = BacktestConfig(
        assets=[
            'AAPL', 'AMZN', 'CAT', 'JNJ', 'JPM', 'KO', 'MSFT', 'NVDA', 'V', 'XOM',
            'EEM', 'EFA', 'IWM', 'QQQ', 'SPY', 'TLT', 'VNQ', 'XLE', 'XLK', 'XLV'
        ],
        start_date='2015-01-02',
        end_date='2025-12-30',
        window_months=60,
        result_dir='result_fixed_cov_backtest'
    )

    print("Fetching and preparing market and macro data...")
    daily_rets, monthly_rets, macro_df = prepare_data(config)

    # We set kappa globally for this experiment
    FIXED_KAPPA = 0.25
    strategy_results = {}

    # -------------------------------------------------------------------------
    # Strategy 1: Equal Weight Benchmark (1/N)
    # -------------------------------------------------------------------------
    print("Building Equal Weight Benchmark...")
    ew_res = build_equal_weight_benchmark(monthly_rets, config.window_months)
    strategy_results[ew_res["name"]] = ew_res

    # -------------------------------------------------------------------------
    # Strategy 2: Proposed 2 (Fixed Kappa) + Sample Covariance
    # -------------------------------------------------------------------------
    print(f"Running Strategy: Fixed Kappa {FIXED_KAPPA} (Sample Cov)...")

    def sample_cov_builder(daily_window: pd.DataFrame) -> np.ndarray:
        return np.asarray(sample_covariance(daily_window)) * 21  # Monthly scaling

    res_sample = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Fixed Kappa (Sample Cov)',
        build_sigma_fn=sample_cov_builder,
        omega_method='advanced',  # Advanced static Omega
        kappa=FIXED_KAPPA  # <--- Fixed Kappa injected here
    )
    strategy_results[res_sample["name"]] = res_sample

    # -------------------------------------------------------------------------
    # Strategy 3: Proposed 2 (Fixed Kappa) + L2 Regularized Covariance
    # -------------------------------------------------------------------------
    print(f"Running Strategy: Fixed Kappa {FIXED_KAPPA} (L2 Reg Cov)...")

    def l2_cov_builder(daily_window: pd.DataFrame) -> np.ndarray:
        return np.asarray(l2_regularized_covariance(daily_window)) * 21  # Monthly scaling

    res_l2 = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Fixed Kappa (L2 Reg Cov)',
        build_sigma_fn=l2_cov_builder,
        omega_method='advanced',
        kappa=FIXED_KAPPA
    )
    strategy_results[res_l2["name"]] = res_l2

    # -------------------------------------------------------------------------
    # Strategy 4: Proposed 2 (Fixed Kappa) + Shrinkage Covariance
    # -------------------------------------------------------------------------
    print(f"Running Strategy: Fixed Kappa {FIXED_KAPPA} (Shrinkage Cov)...")

    def shrink_cov_builder(daily_window: pd.DataFrame) -> np.ndarray:
        return np.asarray(shrinkage_covariance(daily_window)) * 21  # Monthly scaling

    res_shrink = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Fixed Kappa (Shrinkage Cov)',
        build_sigma_fn=shrink_cov_builder,
        omega_method='advanced',
        kappa=FIXED_KAPPA
    )
    strategy_results[res_shrink["name"]] = res_shrink


    # =========================================================================
    # DATA CLEANUP: Remove trailing zero-weight rows
    # =========================================================================
    for name, res in strategy_results.items():
        w_df = res["weights"]
        # Keep only rows where the sum of the absolute weights is greater than 0
        res["weights"] = w_df[w_df.abs().sum(axis=1) > 0]

    # -------------------------------------------------------------------------
    # Evaluate and Export
    # -------------------------------------------------------------------------
    print("\nEvaluating strategies...")
    report_df = evaluate_strategies(strategy_results, benchmark_name="Equal Weight (1/N)")

    fmt_df = format_metrics_table(report_df)
    print("\n" + "=" * 90)
    print("FIXED KAPPA COVARIANCE EVALUATION MATRIX")
    print("=" * 90)
    print(fmt_df.to_string())

    print(f"\nExporting results to {config.result_dir}/ ...")
    export_results(report_df, strategy_results, config.result_dir)
    print("Done!")

    # --- PRINT LAST WEIGHTS ---
    print("\n" + "=" * 90)
    print("FINAL PORTFOLIO WEIGHTS (LAST REBALANCE)")
    print("=" * 90)

    for strategy_name, result in strategy_results.items():
        # Retrieve the weights DataFrame for the current strategy
        weights_df = result["weights"]

        # Get the last row (the most recent rebalancing weights)
        last_weights = weights_df.iloc[-1]

        # Filter out assets with negligible weights (e.g., < 0.1%) for cleaner output
        active_weights = last_weights[last_weights > 0.001].sort_values(ascending=False)

        print(f"\nStrategy: {strategy_name}")
        print(f"Date: {weights_df.index[-1].date()}")
        print(active_weights.to_string())
    # ----------------------------------------------


if __name__ == "__main__":
    run_fixed_kappa_covariance_experiment()
