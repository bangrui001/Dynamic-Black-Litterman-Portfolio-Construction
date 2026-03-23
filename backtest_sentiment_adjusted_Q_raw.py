import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from bl_backtest_engine import (
    BacktestConfig,
    fetch_market_data,
    run_single_strategy_backtest,
    build_equal_weight_benchmark,
    evaluate_strategies,
    export_results,
    default_sigma_builder,
    default_pi_builder,
    default_omega_builder,
    compute_bl_posterior,
    optimize_portfolio,
    format_metrics_table,
)

# =========================================================
# 1. CONFIG
# =========================================================
NEWS_PATH = "market_news_with_sentiment.csv"

ASSETS = [
    "AAPL", "AMZN", "CAT", "JNJ", "JPM", "KO", "MSFT", "NVDA", "V", "XOM",
    "EEM", "EFA", "IWM", "QQQ", "SPY", "TLT", "VNQ", "XLE", "XLK", "XLV"
]


# =========================================================
# 2. BUILD MONTHLY RAW SENTIMENT MATRIX
# =========================================================
def build_monthly_sentiment_features(
    news_path: str,
    tickers: list[str],
    monthly_index: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build RAW monthly sentiment features from a CSV with columns:
      timestamp, symbols, headline, summary, sentiment_label, sentiment_score

    RAW means:
    - use only sentiment label mapping {-1, 0, 1}
    - no confidence scaling
    - no cross-ticker weighting
    - no article-count shrinkage
    - no z-score normalization
    - no smoothing

    Returns
    -------
    sent_raw : DataFrame
        Monthly raw sentiment using simple label mapping {-1, 0, 1}.
    article_count : DataFrame
        Monthly article count per ticker.
    sent_feature_df : DataFrame
        Same as sent_raw, aligned to monthly_index.
    """
    news = pd.read_csv(news_path, low_memory=False)

    required_cols = {
        "timestamp", "symbols", "sentiment_label", "sentiment_score"
    }
    missing = required_cols - set(news.columns)
    if missing:
        raise ValueError(f"Missing required columns in sentiment CSV: {missing}")

    news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True, errors="coerce")
    news = news.dropna(subset=["timestamp"]).copy()
    news["timestamp"] = news["timestamp"].dt.tz_localize(None)

    def parse_symbols(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    news["symbols"] = news["symbols"].apply(parse_symbols)

    news_exploded = news.explode("symbols").rename(columns={"symbols": "ticker"})
    news_exploded = news_exploded[news_exploded["ticker"].isin(set(tickers))].copy()

    # RAW sentiment: direction only
    label_sign = {"positive": 1, "negative": -1, "neutral": 0}
    direction = news_exploded["sentiment_label"].map(label_sign).fillna(0.0)
    news_exploded["sent_strength"] = direction * news_exploded["sentiment_score"].fillna(0.0)

    news_exploded["month"] = (
        news_exploded["timestamp"].dt.to_period("M").dt.to_timestamp()
    )

    # Simple unweighted monthly average by ticker
    sent_raw = (
        news_exploded.groupby(["month", "ticker"])["sent_strength"]
        .sum()
        .unstack(level=1)
        .reindex(columns=tickers)
        .fillna(0.0)
    )

    article_count = (
        news_exploded.groupby(["month", "ticker"])
        .size()
        .unstack(level=1)
        .reindex(columns=tickers)
        .fillna(0.0)
    )

    # Align to monthly returns index
    sent_raw = sent_raw.reindex(monthly_index).fillna(0.0)
    article_count = article_count.reindex(monthly_index).fillna(0.0)

    return sent_raw, article_count, sent_raw.copy()


# =========================================================
# 3. SENTIMENT-ONLY Q BUILDER
# =========================================================
def sentiment_q_builder(
    y_monthly_train,
    X_monthly_train,
    X_current,
    ridge_alpha=1.0,
    beta_cap=0.10,
    q_cap=0.10,
    min_obs=12,
    current_date=None,
    assets=None,
    **kwargs
):
    """
    Build Q from RAW sentiment only.

    For asset i:
        r_{i,t+1} = a_i + b_i * raw_sentiment_{i,t} + eps

    Returns
    -------
    q_views : np.ndarray
        Asset-level expected return views for BL.
    residual_variances : np.ndarray
        Per-asset residual variance for Omega baseline method.
    """
    n_assets = y_monthly_train.shape[1]
    q_views = np.zeros(n_assets)
    residual_variances = np.zeros(n_assets)

    for i in range(n_assets):
        x_i = X_monthly_train[:, i].reshape(-1, 1)
        y_i = y_monthly_train[:, i]

        valid = np.isfinite(x_i[:, 0]) & np.isfinite(y_i)
        x_i = x_i[valid]
        y_i = y_i[valid]

        if len(y_i) < min_obs:
            q_views[i] = 0.0
            residual_variances[i] = np.var(y_i, ddof=1) if len(y_i) > 1 else 1e-4
            residual_variances[i] = max(residual_variances[i], 1e-6)
            continue

        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(x_i, y_i)

        beta_i = float(np.clip(model.coef_[0], -beta_cap, beta_cap))
        intercept_i = float(model.intercept_)
        x_curr_i = float(X_current[i])

        q_pred = intercept_i + beta_i * x_curr_i
        q_views[i] = float(np.clip(q_pred, -q_cap, q_cap))

        fitted = intercept_i + beta_i * x_i[:, 0]
        resid = y_i - fitted
        residual_variances[i] = max(np.var(resid, ddof=1), 1e-6) if len(resid) > 1 else 1e-6

    return q_views, residual_variances


# =========================================================
# 4. MAIN EXPERIMENT
# =========================================================
def run_full_experiment():
    config = BacktestConfig(
        assets=ASSETS,
        start_date="2015-01-01",
        end_date="2025-12-31",
        window_months=60,
        result_dir="result_sentiment_adjusted_Q_raw"
    )

    print("Fetching market data...")
    daily_rets, monthly_rets = fetch_market_data(
        config.assets,
        config.start_date,
        config.end_date,
    )

    print("Building RAW sentiment features from CSV...")
    sent_raw, article_count, sentiment_df = build_monthly_sentiment_features(
        news_path=NEWS_PATH,
        tickers=config.assets,
        monthly_index=monthly_rets.index,
    )

    common_idx = monthly_rets.index.intersection(sentiment_df.index)
    monthly_rets = monthly_rets.loc[common_idx].copy()
    sentiment_df = sentiment_df.loc[common_idx].copy()
    article_count = article_count.loc[common_idx].copy()

    strategy_results = {}

    # ============================================
    # Equal Weight
    # ============================================
    print("Building Equal Weight Benchmark...")
    ew_res = build_equal_weight_benchmark(monthly_rets, config.window_months)
    strategy_results[ew_res["name"]] = ew_res

    # ============================================
    # Benchmark 1: MVO
    # ============================================
    print("Running Benchmark 1 (MVO)...")

    def mvo_q_builder(y_monthly_train, **kwargs):
        q_views = np.mean(y_monthly_train, axis=0)
        return q_views, np.zeros(y_monthly_train.shape[1])

    def mvo_posterior_fn(sigma_monthly, q_views, **kwargs):
        return q_views, sigma_monthly

    res_mvo = run_single_strategy_backtest(
        config=config,
        monthly_rets=monthly_rets,
        daily_rets=daily_rets,
        macro_df=sentiment_df,  
        strategy_name="Benchmark 1 (MVO)",
        build_q_views_fn=mvo_q_builder,
        bl_posterior_fn=mvo_posterior_fn,
    )
    strategy_results[res_mvo["name"]] = res_mvo

    # ============================================
    # Sentiment BL: Baseline Omega
    # ============================================
    print("Running RAW Sentiment BL (Baseline Omega)...")
    res_sent_baseline = run_single_strategy_backtest(
        config=config,
        monthly_rets=monthly_rets,
        daily_rets=daily_rets,
        macro_df=sentiment_df,
        strategy_name="RAW Sentiment BL (Baseline Omega)",
        build_q_views_fn=sentiment_q_builder,
        omega_method="baseline",
    )
    strategy_results[res_sent_baseline["name"]] = res_sent_baseline

    # ============================================
    # Sentiment BL: Advanced Omega
    # ============================================
    print("Running RAW Sentiment BL (Advanced Omega)...")
    res_sent_adv = run_single_strategy_backtest(
        config=config,
        monthly_rets=monthly_rets,
        daily_rets=daily_rets,
        macro_df=sentiment_df,
        strategy_name="RAW Sentiment BL (Advanced Omega)",
        build_q_views_fn=sentiment_q_builder,
        omega_method="advanced",
        kappa=0.25,
    )
    strategy_results[res_sent_adv["name"]] = res_sent_adv

    # ============================================
    # Sentiment Dynamic BL
    # ============================================
    print("Running RAW Sentiment Dynamic BL...")
    kappa_grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    n_months = len(monthly_rets)
    grid_returns_track = {k: np.zeros(n_months) for k in kappa_grid}
    dynamic_kappas = np.zeros(n_months)
    current_best_kappa = 0.25

    n_assets = len(config.assets)
    market_weights = np.ones(n_assets) / n_assets

    print(" -> Pre-computing grid search for RAW Sentiment Dynamic BL...")
    for t in range(config.window_months, n_months - 1):
        y_train_m = monthly_rets.iloc[t - config.window_months + 1 : t + 1].values
        X_train_m = sentiment_df.iloc[t - config.window_months : t].values
        X_curr_m = sentiment_df.iloc[t].values

        daily_window = daily_rets.loc[
            monthly_rets.index[t - config.window_months + 1] : monthly_rets.index[t]
        ]

        sigma_monthly = default_sigma_builder(daily_window)
        pi = default_pi_builder(sigma_monthly, market_weights, config.risk_aversion)

        q_views, residual_variances = sentiment_q_builder(
            y_monthly_train=y_train_m,
            X_monthly_train=X_train_m,
            X_current=X_curr_m,
            ridge_alpha=config.ridge_alpha,
        )

        for k in kappa_grid:
            omega = default_omega_builder(
                "advanced",
                sigma_monthly,
                residual_variances,
                config.tau,
                kappa=k,
            )
            post_ret, post_cov = compute_bl_posterior(
                sigma_monthly,
                pi,
                q_views,
                omega,
                config.tau,
            )
            w_k = optimize_portfolio(
                post_ret,
                post_cov,
                drift_weights=None,
                risk_aversion=config.risk_aversion,
                tc_penalty=config.tc_rate,
            )
            grid_returns_track[k][t + 1] = np.dot(w_k, monthly_rets.iloc[t + 1].values)

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

    class DynamicOmegaBuilder:
        def __init__(self, kappas_array, start_t):
            self.kappas_array = kappas_array
            self.t = start_t

        def __call__(self, omega_method, sigma_monthly, residual_variances, tau, kappa):
            k = self.kappas_array[self.t]
            omega = default_omega_builder(
                "advanced",
                sigma_monthly,
                residual_variances,
                tau,
                kappa=k,
            )
            self.t += 1
            return omega

    res_sent_dynamic = run_single_strategy_backtest(
        config=config,
        monthly_rets=monthly_rets,
        daily_rets=daily_rets,
        macro_df=sentiment_df,
        strategy_name="RAW Sentiment Dynamic BL",
        build_q_views_fn=sentiment_q_builder,
        build_omega_fn=DynamicOmegaBuilder(dynamic_kappas, config.window_months),
        omega_method="advanced",
    )
    strategy_results[res_sent_dynamic["name"]] = res_sent_dynamic

    # ============================================
    # Cleanup
    # ============================================
    for name, res in strategy_results.items():
        w_df = res["weights"]
        res["weights"] = w_df[w_df.abs().sum(axis=1) > 0]

    # ============================================
    # Evaluate and export
    # ============================================
    print("\nEvaluating strategies...")
    report_df = evaluate_strategies(
        strategy_results,
        benchmark_name="Equal Weight (1/N)",
    )

    fmt_df = format_metrics_table(report_df)
    print("\n" + "=" * 90)
    print("EMPIRICAL RESEARCH EVALUATION MATRIX")
    print("=" * 90)
    print(fmt_df.to_string())

    print("\nExporting results...")
    export_results(report_df, strategy_results, config.result_dir)
    print(f"[Success] All results exported to ./{config.result_dir}/")

    print("\n" + "=" * 90)
    print("FINAL PORTFOLIO WEIGHTS (LAST REBALANCE)")
    print("=" * 90)

    for strategy_name, result in strategy_results.items():
        weights_df = result["weights"]
        last_weights = weights_df.iloc[-1]
        active_weights = last_weights[last_weights > 0.001].sort_values(ascending=False)

        print(f"\nStrategy: {strategy_name}")
        print(f"Date: {weights_df.index[-1].date()}")
        print(active_weights.to_string())


if __name__ == "__main__":
    run_full_experiment()