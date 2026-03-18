from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.linear_model import Ridge
from yahooquery import Ticker

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class BacktestConfig:
    assets: Sequence[str]
    start_date: str = "2015-01-02"
    end_date: str = "2025-12-30"
    window_months: int = 60
    risk_aversion: float = 2.5
    tau: float = 0.05
    tc_rate: float = 0.001
    ridge_alpha: float = 1.0
    result_dir: str = "result"


# =============================================================================
# DATA PIPELINE
# =============================================================================

def fetch_market_data(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch adjusted close data and return both daily and monthly returns.
    """
    tq = Ticker(list(tickers))
    df = tq.history(start=start_date, end=end_date)

    daily_prices = df["adjclose"].unstack(level=0)
    daily_prices.index = pd.to_datetime(daily_prices.index)
    daily_prices = daily_prices.sort_index()

    daily_returns = daily_prices.pct_change().dropna()
    monthly_prices = daily_prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    return daily_returns, monthly_returns


def fetch_macro_factors(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch macro data from FRED and create stationary monthly features.
    """
    series_dict = {
        "INDPRO": "Growth",
        "CPIAUCSL": "Inflation",
        "GS10": "Rates",
        "BAA10Y": "Credit",
    }

    raw_df = web.DataReader(list(series_dict.keys()), "fred", start_date, end_date)
    raw_df.rename(columns=series_dict, inplace=True)
    raw_df.index = pd.to_datetime(raw_df.index)

    monthly_macro = raw_df.resample("ME").last()

    features = pd.DataFrame(index=monthly_macro.index)
    features["Growth_Signal"] = monthly_macro["Growth"].pct_change()
    features["Inflation_Signal"] = monthly_macro["Inflation"].pct_change(periods=12)
    features["Rates_Signal"] = monthly_macro["Rates"].diff()
    features["Credit_Signal"] = monthly_macro["Credit"].diff()

    return features.dropna()


# =============================================================================
# DEFAULT BUILDING BLOCKS
# =============================================================================

def default_sigma_builder(daily_returns_window: pd.DataFrame) -> np.ndarray:
    """
    Default monthly covariance estimate from daily returns.
    """
    sigma_monthly = daily_returns_window.cov().values * 21
    return sigma_monthly


def default_pi_builder(
    sigma_monthly: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    """
    BL equilibrium prior: pi = delta * Sigma * w_mkt
    """
    return risk_aversion * (sigma_monthly @ market_weights)


def ridge_q_builder(
    y_monthly_train: np.ndarray,
    X_monthly_train: np.ndarray,
    X_current: np.ndarray,
    ridge_alpha: float = 1.0,
    current_date=None,
    assets=None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Default view builder using one Ridge regression per asset.
    Returns:
        q_views: shape (n_assets,)
        residual_variances: shape (n_assets,)
    """
    n_assets = y_monthly_train.shape[1]
    q_views = np.zeros(n_assets)
    residual_variances = np.zeros(n_assets)

    for i in range(n_assets):
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_monthly_train, y_monthly_train[:, i])

        q_views[i] = model.predict(X_current.reshape(1, -1))[0]

        residuals = y_monthly_train[:, i] - model.predict(X_monthly_train)
        residual_variances[i] = np.var(residuals, ddof=1)

    return q_views, residual_variances


def default_omega_builder(
    omega_method: str,
    sigma_monthly: np.ndarray,
    residual_variances: np.ndarray,
    tau: float,
    kappa: float = 0.25,
) -> np.ndarray:
    """
    Default omega construction.
    """
    if omega_method == "baseline":
        return np.diag(residual_variances)
    if omega_method == "advanced":
        return kappa * sigma_monthly
    if omega_method == "subjective":
        return tau * sigma_monthly

    raise ValueError(f"Unsupported omega_method: {omega_method}")


def compute_bl_posterior(
    sigma_monthly: np.ndarray,
    pi: np.ndarray,
    q_views: np.ndarray,
    omega: np.ndarray,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Absolute-view BL posterior with P = I.
    Returns:
        posterior mean, posterior covariance
    """
    tau_sigma = tau * sigma_monthly
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(omega)

    m = np.linalg.inv(tau_sigma_inv + omega_inv)
    posterior_returns = m @ (tau_sigma_inv @ pi + omega_inv @ q_views)
    posterior_cov = sigma_monthly + m

    return posterior_returns, posterior_cov


# =============================================================================
# PORTFOLIO OPTIMIZER
# =============================================================================

def optimize_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    drift_weights: Optional[np.ndarray],
    risk_aversion: float = 2.5,
    tc_penalty: float = 0.001,
) -> np.ndarray:
    """
    Long-only mean-variance optimization with L1 turnover penalty.
    """
    n_assets = len(expected_returns)
    w = cp.Variable(n_assets)

    ret = expected_returns.T @ w
    vol = cp.quad_form(w, cov_matrix)
    utility = ret - (risk_aversion / 2.0) * vol

    if drift_weights is not None:
        utility -= tc_penalty * cp.norm(w - drift_weights, 1)

    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Maximize(utility), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    if w.value is None:
        raise ValueError("Optimization failed: no solution returned by solver.")

    weights = np.asarray(w.value).flatten()
    weights = np.where(weights < 1e-6, 0, weights)

    total = weights.sum()
    if total <= 0:
        raise ValueError("Optimization failed: weights sum to non-positive value.")

    return weights / total


# =============================================================================
# METRICS / REPORTING
# =============================================================================

def calculate_portfolio_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    turnover_history: Optional[pd.Series] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    ann_return = np.mean(strategy_returns) * 12
    ann_vol = np.std(strategy_returns, ddof=1) * np.sqrt(12)

    metrics["Annualized Return"] = ann_return
    metrics["Annualized Volatility"] = ann_vol
    metrics["Sharpe Ratio"] = ann_return / ann_vol if ann_vol != 0 else np.nan

    if benchmark_returns is not None:
        active_returns = strategy_returns - benchmark_returns
        ann_active_return = np.mean(active_returns) * 12
        tracking_error = np.std(active_returns, ddof=1) * np.sqrt(12)
        metrics["Information Ratio"] = (
            ann_active_return / tracking_error if tracking_error != 0 else np.nan
        )
    else:
        metrics["Information Ratio"] = np.nan

    cum_returns = (1 + strategy_returns).cumprod()
    rolling_max = cum_returns.cummax()
    metrics["Maximum Drawdown"] = np.min((cum_returns / rolling_max) - 1)
    metrics["Value at Risk (95%)"] = np.percentile(strategy_returns, 5)

    if turnover_history is not None:
        metrics["Annualized Turnover"] = (np.mean(turnover_history) * 12) / 2
    else:
        metrics["Annualized Turnover"] = np.nan

    return metrics


def format_metrics_table(report_df: pd.DataFrame) -> pd.DataFrame:
    fmt_df = report_df.copy()

    pct_rows = [
        "Annualized Return",
        "Maximum Drawdown",
        "Annualized Volatility",
        "Value at Risk (95%)",
        "Annualized Turnover",
    ]
    ratio_rows = ["Sharpe Ratio", "Information Ratio"]

    for row in pct_rows:
        fmt_df.loc[row] = fmt_df.loc[row].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
        )

    for row in ratio_rows:
        fmt_df.loc[row] = fmt_df.loc[row].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )

    return fmt_df


# =============================================================================
# PLOTS
# =============================================================================

def plot_comparison(returns_dict: Dict[str, pd.Series]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    for name, rets in returns_dict.items():
        cum_ret = 100 * (1 + rets).cumprod()
        ax1.plot(cum_ret.index, cum_ret, label=name, linewidth=2)

        drawdown = (cum_ret / cum_ret.cummax()) - 1
        ax2.plot(drawdown.index, drawdown, label=name, linewidth=1.5)

    ax1.set_title("Cumulative Portfolio Value (Starting at $100)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", frameon=True, fontsize=9)

    ax2.set_title("Underwater Plot (Drawdowns)")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", frameon=True, fontsize=9)

    plt.tight_layout()


def plot_asset_allocation(weights_dict: Dict[str, pd.DataFrame]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    n = len(weights_dict)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    colors_palette = plt.cm.tab20.colors
    strategy_names = list(weights_dict.keys())

    for idx, ax in enumerate(axes):
        if idx < len(strategy_names):
            name = strategy_names[idx]
            weights_df = weights_dict[name]
            ax.stackplot(
                weights_df.index,
                weights_df.T,
                labels=weights_df.columns,
                colors=colors_palette,
                alpha=0.85,
            )
            ax.set_title(name)
            ax.set_ylabel("Weight")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        else:
            ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(8, len(labels)),
        bbox_to_anchor=(0.5, 0.02),
        title="Assets",
        frameon=True,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)


# =============================================================================
# FLEXIBLE BACKTEST ENGINE
# =============================================================================

def run_single_strategy_backtest(
    config: BacktestConfig,
    monthly_rets: pd.DataFrame,
    daily_rets: pd.DataFrame,
    macro_df: pd.DataFrame,
    strategy_name: str,
    build_q_views_fn: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None,
    build_sigma_fn: Optional[Callable[..., np.ndarray]] = None,
    build_omega_fn: Optional[Callable[..., np.ndarray]] = None,
    bl_posterior_fn: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None,
    omega_method: str = "advanced",
    kappa: float = 0.25,
    initial_weights: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Run one strategy with pluggable components.

    You can replace:
    - build_q_views_fn
    - build_sigma_fn
    - build_omega_fn
    - bl_posterior_fn

    This is the key function you will call for your own optimization results.
    """

    build_q_views_fn = build_q_views_fn or ridge_q_builder
    build_sigma_fn = build_sigma_fn or default_sigma_builder
    build_omega_fn = build_omega_fn or default_omega_builder
    bl_posterior_fn = bl_posterior_fn or compute_bl_posterior

    common_idx = monthly_rets.index.intersection(macro_df.index)
    monthly_rets = monthly_rets.loc[common_idx].copy()
    macro_df = macro_df.loc[common_idx].copy()

    n_months, n_assets = monthly_rets.shape
    assets = list(monthly_rets.columns)

    if initial_weights is None:
        current_weights = np.ones(n_assets) / n_assets
    else:
        current_weights = np.asarray(initial_weights, dtype=float)
        current_weights = current_weights / current_weights.sum()

    net_returns_hist = np.zeros(n_months)
    turnover_hist = np.zeros(n_months)
    weights_hist = np.zeros((n_months, n_assets))

    market_weights = np.ones(n_assets) / n_assets

    for t in range(config.window_months, n_months - 1):
        y_train_m = monthly_rets.iloc[t - config.window_months + 1 : t + 1].values
        X_train_m = macro_df.iloc[t - config.window_months : t].values
        X_curr_m = macro_df.iloc[t].values

        start_period = monthly_rets.index[t - config.window_months + 1]
        end_period = monthly_rets.index[t]
        daily_window = daily_rets.loc[start_period:end_period]

        sigma_monthly = build_sigma_fn(daily_window)
        pi = default_pi_builder(sigma_monthly, market_weights, config.risk_aversion)

        q_views, residual_variances = build_q_views_fn(
            y_monthly_train=y_train_m,
            X_monthly_train=X_train_m,
            X_current=X_curr_m,
            ridge_alpha=config.ridge_alpha,
            current_date=monthly_rets.index[t],
            assets=assets,
        )

        omega = build_omega_fn(
            omega_method=omega_method,
            sigma_monthly=sigma_monthly,
            residual_variances=residual_variances,
            tau=config.tau,
            kappa=kappa,
        )

        post_returns, post_cov = bl_posterior_fn(
            sigma_monthly=sigma_monthly,
            pi=pi,
            q_views=q_views,
            omega=omega,
            tau=config.tau,
        )

        ret_t = monthly_rets.iloc[t].values
        drift_weights = current_weights * (1 + ret_t)
        drift_weights = drift_weights / drift_weights.sum()

        new_w = optimize_portfolio(
            expected_returns=post_returns,
            cov_matrix=post_cov,
            drift_weights=drift_weights,
            risk_aversion=config.risk_aversion,
            tc_penalty=config.tc_rate,
        )

        turnover = np.sum(np.abs(new_w - drift_weights))
        tc = turnover * config.tc_rate

        ret_t_plus_1 = monthly_rets.iloc[t + 1].values
        gross_ret = np.dot(current_weights, ret_t_plus_1)

        net_returns_hist[t + 1] = gross_ret - tc
        turnover_hist[t] = turnover
        weights_hist[t] = new_w
        current_weights = new_w

    eval_idx = monthly_rets.index[config.window_months:]
    strategy_returns = pd.Series(net_returns_hist[config.window_months:], index=eval_idx, name=strategy_name)
    strategy_turnover = pd.Series(turnover_hist[config.window_months:], index=eval_idx, name=strategy_name)
    strategy_weights = pd.DataFrame(weights_hist[config.window_months:], index=eval_idx, columns=assets)

    return {
        "name": strategy_name,
        "returns": strategy_returns,
        "turnover": strategy_turnover,
        "weights": strategy_weights,
    }


def build_equal_weight_benchmark(monthly_rets: pd.DataFrame, window_months: int) -> Dict[str, object]:
    eval_idx = monthly_rets.index[window_months:]
    n_assets = monthly_rets.shape[1]

    ew_returns = monthly_rets.loc[eval_idx].mean(axis=1)
    ew_weights = pd.DataFrame(1 / n_assets, index=eval_idx, columns=monthly_rets.columns)
    ew_turnover = pd.Series(0.0, index=eval_idx)

    return {
        "name": "Equal Weight (1/N)",
        "returns": ew_returns,
        "turnover": ew_turnover,
        "weights": ew_weights,
    }


def evaluate_strategies(
    strategy_results: Dict[str, Dict[str, object]],
    benchmark_name: Optional[str] = None,
) -> pd.DataFrame:
    if benchmark_name is None:
        benchmark_name = list(strategy_results.keys())[0]

    benchmark_returns = strategy_results[benchmark_name]["returns"]

    report = {}
    for name, result in strategy_results.items():
        report[name] = calculate_portfolio_metrics(
            strategy_returns=result["returns"],
            benchmark_returns=benchmark_returns,
            turnover_history=result["turnover"],
        )

    ordered_rows = [
        "Annualized Return",
        "Sharpe Ratio",
        "Information Ratio",
        "Maximum Drawdown",
        "Annualized Volatility",
        "Value at Risk (95%)",
        "Annualized Turnover",
    ]

    report_df = pd.DataFrame(report).reindex(ordered_rows)
    return report_df


def export_results(
    report_df: pd.DataFrame,
    strategy_results: Dict[str, Dict[str, object]],
    result_dir: str,
) -> None:
    os.makedirs(result_dir, exist_ok=True)

    excel_path = os.path.join(result_dir, "evaluation_metrics.xlsx")
    report_df.to_excel(excel_path, index=True)

    returns_dict = {name: res["returns"] for name, res in strategy_results.items()}
    weights_dict = {name: res["weights"] for name, res in strategy_results.items()}

    plot_comparison(returns_dict)
    plt.savefig(os.path.join(result_dir, "return_comparison.pdf"), bbox_inches="tight", dpi=300)

    plot_asset_allocation(weights_dict)
    plt.savefig(os.path.join(result_dir, "weights_comparison.pdf"), bbox_inches="tight", dpi=300)


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def prepare_data(config: BacktestConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_rets, monthly_rets = fetch_market_data(config.assets, config.start_date, config.end_date)
    macro_df = fetch_macro_factors(config.start_date, config.end_date)

    common_idx = monthly_rets.index.intersection(macro_df.index)
    monthly_rets = monthly_rets.loc[common_idx]
    macro_df = macro_df.loc[common_idx]

    return daily_rets, monthly_rets, macro_df