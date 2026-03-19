import numpy as np
import pandas as pd
import warnings
import cvxpy as cp

# 1. Import the core engine and all necessary building blocks
from bl_backtest_engine import (
    BacktestConfig,
    prepare_data,
    run_single_strategy_backtest,
    build_equal_weight_benchmark,
    evaluate_strategies,
    export_results,
    format_metrics_table,
    default_omega_builder,
    ridge_q_builder,
    default_sigma_builder,
    default_pi_builder,
    compute_bl_posterior,
    optimize_portfolio
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

def l2_regularized_covariance(train_returns: pd.DataFrame, l2_penalty: float = 0.10) -> np.ndarray:
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
# DYNAMIC OMEGA BUILDER (From Proposed 3)
# =============================================================================

class DynamicOmegaBuilder:
    def __init__(self, kappas_array, start_t):
        self.kappas_array = kappas_array
        self.t = start_t

    def __call__(self, omega_method, sigma_monthly, residual_variances, tau, kappa):
        k = self.kappas_array[self.t]
        omega = default_omega_builder("advanced", sigma_monthly, residual_variances, tau, kappa=k)
        self.t += 1
        return omega

# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_dynamic_covariance_experiment():
    config = BacktestConfig(
        assets=[
            'AAPL', 'AMZN', 'CAT', 'JNJ', 'JPM', 'KO', 'MSFT', 'NVDA', 'V', 'XOM',
            'EEM', 'EFA', 'IWM', 'QQQ', 'SPY', 'TLT', 'VNQ', 'XLE', 'XLK', 'XLV'
        ],
        start_date='2015-01-02',
        end_date='2025-12-30',
        window_months=60,
        result_dir='result_dynamic_cov_backtest'
    )

    print("Fetching and preparing market and macro data...")
    daily_rets, monthly_rets, macro_df = prepare_data(config)
    n_months = len(monthly_rets)

    # -------------------------------------------------------------------------
    # PRE-COMPUTE DYNAMIC KAPPAS (WFO Grid Search)
    # -------------------------------------------------------------------------
    print(" -> Pre-computing Grid Search for Dynamic BL Kappas...")
    kappa_grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    grid_returns_track = {k: np.zeros(n_months) for k in kappa_grid}
    dynamic_kappas = np.zeros(n_months)
    current_best_kappa = 0.25
    n_assets = len(config.assets)
    market_weights = np.ones(n_assets) / n_assets

    for t in range(config.window_months, n_months - 1):
        y_train_m = monthly_rets.iloc[t - config.window_months + 1: t + 1].values
        X_train_m = macro_df.iloc[t - config.window_months: t].values
        X_curr_m = macro_df.iloc[t].values

        daily_window = daily_rets.loc[monthly_rets.index[t - config.window_months + 1]: monthly_rets.index[t]]
        sigma_monthly = default_sigma_builder(daily_window)
        pi = default_pi_builder(sigma_monthly, market_weights, config.risk_aversion)

        q_views, residual_variances = ridge_q_builder(y_train_m, X_train_m, X_curr_m, config.ridge_alpha)

        for k in kappa_grid:
            omega = default_omega_builder("advanced", sigma_monthly, residual_variances, config.tau, kappa=k)
            post_ret, post_cov = compute_bl_posterior(sigma_monthly, pi, q_views, omega, config.tau)
            w_k = optimize_portfolio(post_ret, post_cov, drift_weights=None, risk_aversion=config.risk_aversion,
                                     tc_penalty=config.tc_rate)
            grid_returns_track[k][t + 1] = np.dot(w_k, monthly_rets.iloc[t + 1].values)

        if t % 12 == 0:
            lookback = 24
            best_k = kappa_grid[0]
            max_sharpe = -np.inf
            for k in kappa_grid:
                recent_rets = grid_returns_track[k][t - lookback: t]
                if len(recent_rets) < lookback:
                    continue
                vol = np.std(recent_rets, ddof=1) * np.sqrt(12)
                ann_ret = np.mean(recent_rets) * 12
                sharpe = ann_ret / vol if vol > 0 else -np.inf
                if sharpe > max_sharpe:
                    max_sharpe = sharpe
                    best_k = k
            current_best_kappa = best_k

        dynamic_kappas[t] = current_best_kappa

    strategy_results = {}

    # -------------------------------------------------------------------------
    # Strategy 1: Equal Weight Benchmark (1/N)
    # -------------------------------------------------------------------------
    print("Building Equal Weight Benchmark...")
    ew_res = build_equal_weight_benchmark(monthly_rets, config.window_months)
    strategy_results[ew_res["name"]] = ew_res

    # -------------------------------------------------------------------------
    # Strategy 2: Dynamic BL + Sample Covariance
    # -------------------------------------------------------------------------
    print("Running Strategy: Dynamic BL with Sample Covariance...")

    def sample_cov_builder(daily_window: pd.DataFrame) -> np.ndarray:
        return np.asarray(sample_covariance(daily_window)) * 21

    res_sample = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Dynamic BL (Sample Cov)',
        build_sigma_fn=sample_cov_builder,
        build_omega_fn=DynamicOmegaBuilder(dynamic_kappas, config.window_months),
        omega_method='advanced'
    )
    strategy_results[res_sample["name"]] = res_sample

    # -------------------------------------------------------------------------
    # Strategy 3: Dynamic BL + L2 Regularized Covariance
    # -------------------------------------------------------------------------
    print("Running Strategy: Dynamic BL with L2 Regularized Covariance...")

    def l2_cov_builder(daily_window: pd.DataFrame) -> np.ndarray:
        return np.asarray(l2_regularized_covariance(daily_window)) * 21

    res_l2 = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Dynamic BL (L2 Reg Cov)',
        build_sigma_fn=l2_cov_builder,
        build_omega_fn=DynamicOmegaBuilder(dynamic_kappas, config.window_months),
        omega_method='advanced'
    )
    strategy_results[res_l2["name"]] = res_l2

    # -------------------------------------------------------------------------
    # Strategy 4: Dynamic BL + Shrinkage Covariance
    # -------------------------------------------------------------------------
    print("Running Strategy: Dynamic BL with Shrinkage Covariance...")

    def shrink_cov_builder(daily_window: pd.DataFrame) -> np.ndarray:
        return np.asarray(shrinkage_covariance(daily_window)) * 21

    res_shrink = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Dynamic BL (Shrinkage Cov)',
        build_sigma_fn=shrink_cov_builder,
        build_omega_fn=DynamicOmegaBuilder(dynamic_kappas, config.window_months),
        omega_method='advanced'
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
    print("DYNAMIC BL COVARIANCE EVALUATION MATRIX")
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
        print(f"Date: {weights_df.index[-1].date}")
        print(active_weights.to_string())
    # ----------------------------------------------

if __name__ == "__main__":
    run_dynamic_covariance_experiment()
