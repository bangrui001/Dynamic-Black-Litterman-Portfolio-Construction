import numpy as np
import pandas as pd

# Import the core engine and its default building blocks
from bl_backtest_engine import (
    BacktestConfig, prepare_data, run_single_strategy_backtest,
    build_equal_weight_benchmark, evaluate_strategies, export_results,
    ridge_q_builder, default_sigma_builder, default_pi_builder,
    default_omega_builder, compute_bl_posterior, optimize_portfolio,
    format_metrics_table
)

def run_full_experiment():
    # 1. Configuration
    config = BacktestConfig(
        assets=[
            'AAPL', 'AMZN', 'CAT', 'JNJ', 'JPM', 'KO', 'MSFT', 'NVDA', 'V', 'XOM',
            'EEM', 'EFA', 'IWM', 'QQQ', 'SPY', 'TLT', 'VNQ', 'XLE', 'XLK', 'XLV'
        ],
        start_date='2015-01-02',
        end_date='2025-12-30',
        window_months=60
    )

    print("Fetching and preparing data...")
    daily_rets, monthly_rets, macro_df = prepare_data(config)

    strategy_results = {}

    # Add Equal Weight Baseline
    print("Building Equal Weight Benchmark...")
    ew_res = build_equal_weight_benchmark(monthly_rets, config.window_months)
    strategy_results[ew_res["name"]] = ew_res

    # =========================================================================
    # Strategy 1: Benchmark 1 (MVO)
    # =========================================================================
    print("Running Benchmark 1 (MVO)...")
    def mvo_q_builder(y_monthly_train, **kwargs):
        # MVO expected returns = historical mean
        q_views = np.mean(y_monthly_train, axis=0)
        return q_views, np.zeros(y_monthly_train.shape[1])

    def mvo_posterior_fn(sigma_monthly, q_views, **kwargs):
        # MVO skips the BL posterior update completely
        return q_views, sigma_monthly

    res_mvo = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Benchmark 1 (MVO)',
        build_q_views_fn=mvo_q_builder,
        bl_posterior_fn=mvo_posterior_fn
    )
    strategy_results[res_mvo["name"]] = res_mvo

    # =========================================================================
    # Strategy 2: Benchmark 2 (Standard BL)
    # =========================================================================
    print("Running Benchmark 2 (Standard BL)...")
    res_std_bl = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Benchmark 2 (Standard BL)',
        omega_method='subjective'
    )
    strategy_results[res_std_bl["name"]] = res_std_bl

    # =========================================================================
    # Strategy 3: Proposed 1 (Baseline Omega)
    # =========================================================================
    print("Running Proposed 1 (Baseline Omega)...")
    res_prop1 = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Proposed 1 (Baseline Omega)',
        omega_method='baseline'
    )
    strategy_results[res_prop1["name"]] = res_prop1

    # =========================================================================
    # Strategy 4: Proposed 2 (Advanced Omega)
    # =========================================================================
    print("Running Proposed 2 (Advanced Omega)...")
    res_prop2 = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Proposed 2 (Advanced Omega)',
        omega_method='advanced',
        kappa=0.25
    )
    strategy_results[res_prop2["name"]] = res_prop2

    # =========================================================================
    # Strategy 5: Proposed 3 (Dynamic BL) - Precompute Kappas & Run
    # =========================================================================
    print("Running Proposed 3 (Dynamic BL)...")
    kappa_grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    n_months = len(monthly_rets)
    grid_returns_track = {k: np.zeros(n_months) for k in kappa_grid}
    dynamic_kappas = np.zeros(n_months)
    current_best_kappa = 0.25
    n_assets = len(config.assets)
    market_weights = np.ones(n_assets) / n_assets

    print(" -> Pre-computing Grid Search for Dynamic BL...")
    for t in range(config.window_months, n_months - 1):
        y_train_m = monthly_rets.iloc[t - config.window_months + 1 : t + 1].values
        X_train_m = macro_df.iloc[t - config.window_months : t].values
        X_curr_m = macro_df.iloc[t].values

        daily_window = daily_rets.loc[monthly_rets.index[t - config.window_months + 1] : monthly_rets.index[t]]
        sigma_monthly = default_sigma_builder(daily_window)
        pi = default_pi_builder(sigma_monthly, market_weights, config.risk_aversion)

        q_views, residual_variances = ridge_q_builder(y_train_m, X_train_m, X_curr_m, config.ridge_alpha)

        for k in kappa_grid:
            omega = default_omega_builder("advanced", sigma_monthly, residual_variances, config.tau, kappa=k)
            post_ret, post_cov = compute_bl_posterior(sigma_monthly, pi, q_views, omega, config.tau)
            w_k = optimize_portfolio(post_ret, post_cov, drift_weights=None, risk_aversion=config.risk_aversion, tc_penalty=config.tc_rate)
            grid_returns_track[k][t+1] = np.dot(w_k, monthly_rets.iloc[t+1].values)

        # WFO Meta-Optimization Logic
        if t % 12 == 0:
            lookback = 24
            best_k = kappa_grid[0]
            max_sharpe = -np.inf
            for k in kappa_grid:
                recent_rets = grid_returns_track[k][t - lookback : t]
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

    # Stateful Callable injected into the engine
    class DynamicOmegaBuilder:
        def __init__(self, kappas_array, start_t):
            self.kappas_array = kappas_array
            self.t = start_t
            
        def __call__(self, omega_method, sigma_monthly, residual_variances, tau, kappa):
            k = self.kappas_array[self.t]
            omega = default_omega_builder("advanced", sigma_monthly, residual_variances, tau, kappa=k)
            self.t += 1
            return omega

    res_prop3 = run_single_strategy_backtest(
        config=config, monthly_rets=monthly_rets, daily_rets=daily_rets, macro_df=macro_df,
        strategy_name='Proposed 3 (Dynamic BL)',
        build_omega_fn=DynamicOmegaBuilder(dynamic_kappas, config.window_months),
        omega_method='advanced'
    )
    strategy_results[res_prop3["name"]] = res_prop3

    # =========================================================================
    # DATA CLEANUP: Remove trailing zero-weight rows
    # =========================================================================
    for name, res in strategy_results.items():
        w_df = res["weights"]
        # Keep only rows where the sum of the absolute weights is greater than 0
        res["weights"] = w_df[w_df.abs().sum(axis=1) > 0]


    # =========================================================================
    # Evaluate and Export
    # =========================================================================
    print("\nEvaluating strategies...")
    report_df = evaluate_strategies(strategy_results, benchmark_name="Equal Weight (1/N)")
    
    fmt_df = format_metrics_table(report_df)
    print("\n" + "="*90)
    print("EMPIRICAL RESEARCH EVALUATION MATRIX")
    print("="*90)
    print(fmt_df.to_string())

    print("\nExporting results...")
    export_results(report_df, strategy_results, config.result_dir)
    print(f"[Success] All results exported to ./{config.result_dir}/")

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
    run_full_experiment()
