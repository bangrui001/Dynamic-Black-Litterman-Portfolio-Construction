# Dynamic-Black-Litterman-Portfolio-Construction

## 📖 Overview

This repository implements a highly modular, end-to-end quantitative backtesting engine for multi-asset portfolio construction. It extends the traditional Black-Litterman (BL) model by dynamically integrating macroeconomic signals via Machine Learning (Ridge Regression) and employing advanced risk-management techniques, including Walk-Forward Optimization (WFO) for view confidence ($\Omega$) and robust covariance estimation (Shrinkage & L2 Regularization).

The goal is to address the notorious "error-maximizing" nature of standard Mean-Variance Optimization (MVO) by producing stable, theoretically sound, and transaction-cost-aware portfolio weights.

## ⚙️ Architecture & Data Pipeline

The backtest engine (`bl_backtest_engine.py`) is designed with a fully decoupled architecture, allowing researchers to plug in custom callables for signal generation, covariance estimation, and portfolio optimization.

### 🌐 Asset Universe

The portfolio consists of 20 diverse assets specifically selected to represent a global investment opportunity set:

| Category | Tickers | Description |
| --- | --- | --- |
| **Mega-Cap Equities** | AAPL, AMZN, MSFT, NVDA, V | Growth leaders in technology and payment processing. |
| **Value & Defensive** | CAT, JNJ, JPM, KO, XOM | Exposure to industrials, healthcare, financials, and energy. |
| **Broad Market ETFs** | SPY, QQQ, IWM | Proxies for the S&P 500, Nasdaq-100, and Russell 2000. |
| **International** | EFA, EEM | Developed (Ex-US) and Emerging Markets exposure. |
| **Fixed Income** | TLT | Long-term U.S. Treasury bonds for duration and hedging. |
| **Alternatives** | VNQ, XLE, XLK, XLV | Real Estate (REITs) and specific Sector SPDRs. |

---

### 📈 Data & Signals

* **Market Data**: Daily and monthly adjusted close prices are fetched via `yahooquery`, covering the period from 2015 to 2025.
* **Macroeconomic Factors**: Sourced directly from the Federal Reserve Economic Data (FRED) using `pandas_datareader`. Features are stationarized to prevent spurious regression:
* **Growth Signal**: 1-month % change in Industrial Production (**INDPRO**).
* **Inflation Signal**: 12-month % change (YoY) in Consumer Price Index (**CPIAUCSL**).
* **Rates Signal**: 1-month absolute change in 10-Year Treasury Yield (**GS10**).
* **Credit Signal**: 1-month absolute change in the Baa Corporate Bond Yield Spread (**BAA10Y**).



**Predictive Alignment**: The model implements a "point-in-time" approach where macro features at month $t$ are used to predict asset returns for month $t+1$. This structural lag is critical to eliminating look-ahead bias throughout the backtest.


## 🧮 Mathematical Framework

### 1. View Generation (Ridge Regression)

Instead of relying on subjective human inputs, absolute views are generated quantitatively. For each asset $i$, we fit a Ridge Regression model using lagged macroeconomic features $X$ to predict the next month's return $y_i$:

$$\hat{\beta}_i = \arg\min_{\beta} \|y_i - X\beta\|^2_2 + \alpha \|\beta\|^2_2$$

The quantitative view vector $Q$ for period $t$ is then formed by forecasting with the current macro state $X_t$:

$$Q_{t, i} = X_t \hat{\beta}_i$$

### 2. The Black-Litterman Posterior

The BL model blends the market equilibrium prior ($\Pi$) with our quantitative views ($Q$). Since we predict absolute returns for all assets, our pick matrix $P$ is the identity matrix $I$.

* **Equilibrium Prior**: $\Pi = \delta \Sigma w_{mkt}$
* **Posterior Expected Returns**:

$$\mu_{BL} = [(\tau \Sigma)^{-1} + \Omega^{-1}]^{-1} [(\tau \Sigma)^{-1} \Pi + \Omega^{-1} Q]$$


* **Posterior Covariance**:

$$\Sigma_{BL} = \Sigma + [(\tau \Sigma)^{-1} + \Omega^{-1}]^{-1}$$



Where $\Sigma$ is the historical covariance, $\delta$ is the risk aversion parameter, $\tau$ is the weight-on-views scalar, and $\Omega$ is the uncertainty matrix of our views.

### 3. Dynamic Confidence Modeling (The $\Omega$ Matrix)

A critical innovation in this project is the treatment of $\Omega$. The engine supports three methodologies:

* **Baseline**: A diagonal matrix using the out-of-sample residual variance from the Ridge regressions.
* **Advanced (Fixed Kappa)**: Assumes view uncertainty is proportional to the market covariance: $\Omega = \kappa \Sigma$.
* **Dynamic WFO**: A Walk-Forward Optimization approach that simulates past performance over a rolling 24-month window to dynamically select the optimal $\kappa_t$ from a predefined grid that maximizes the historical Sharpe ratio.

### 4. Robust Covariance Estimation

High-dimensional covariance matrices are prone to estimation errors, leading to drastic weight fluctuations. The engine incorporates `cvxpy`-based robust estimators:

* **Ledoit-Wolf Shrinkage**: Shrinks the sample covariance $S$ toward a highly structured target matrix (scaled identity $F$) with intensity $s$ (shrinkage strength):

$$\Sigma_{shrink} = (1 - s)S + s F$$

> **Hyperparameter Sensitivity**: The engine includes a dedicated grid search module to empirically test the impact of the **Shrinkage Strength** parameter:
> * $s \in \{0.25, 0.50, 0.75, 1.00\}$

* **L2 Regularized Covariance**: Adds a penalty to the diagonal to ensure strict positive semi-definiteness and reduce condition numbers:

$$\Sigma_{L2} = S + \lambda I$$



### 5. Turnover-Aware Portfolio Optimization

The final portfolio weights are solved using a long-only Mean-Variance Optimizer powered by `CVXPY`. To prevent excessive rebalancing driven by shifting expected returns, an $L_1$-norm penalty is applied to model transaction costs:

$$\max_{w} \ w^T \mu_{BL} - \frac{\gamma}{2} w^T \Sigma_{BL} w - tc \|w - w_{drift}\|_1$$

Subject to:


$$\sum_{i=1}^{N} w_i = 1, \quad w_i \ge 0$$

Where $w_{drift}$ represents the current portfolio weights drifted by the previous period's asset returns.


---

## 🚀 Experimental Setup & Usage

### 1. Environment Installation

Before running any experiments, install the required dependencies using the provided `requirements.txt` to ensure all mathematical solvers and data pipelines function correctly.

```bash
pip install -r requirements.txt

```

**Core Dependencies:**

* **Data Acquisition**: `numpy`, `pandas`, `pandas_datareader`, and `yahooquery` for fetching market prices and FRED macro signals.
* **Optimization**: `cvxpy` and `scipy` for solving the long-only Mean-Variance problem with $L_1$ penalties.
* **Machine Learning**: `scikit-learn` for the Ridge Regression-based view generation.
* **Visualization & Export**: `matplotlib` for generating underwater plots and `openpyxl` for data handling.

---

### 2. Baseline Configuration (Fixed Parameters)

The core experiment uses a consistent set of default values defined in the `BacktestConfig` class to ensure a fair comparison across all strategies:

| Parameter | Default Value | Description |
| --- | --- | --- |
| **Risk Aversion ($\delta$)** | `2.5` | Represents the market's average risk-reward trade-off. |
| **Tau ($\tau$)** | `0.05` | Scalar weight for the relative certainty of views. |
| **Window Months** | `60` | 5-year rolling lookback for training Ridge models. |
| **Transaction Cost** | `0.001` | 0.1% penalty per unit of turnover. |
| **Ridge Alpha** | `1.0` | L2 regularization strength for view generation. |
| **Kappa ($\kappa$)** | `0.25` | Default multiplier for the $\Omega$ matrix. |

---

### 3. Execution Scripts (Experimental Variables)

Each script isolates different moving parts of the framework to evaluate model robustness:


---

#### **A. Strategy Evaluation & Benchmarking**


**`run_experiment_br.py'**: This script executes a comparative backtest across six distinct portfolio formulations, segmented into a baseline control group and our proposed Black-Litterman (BL) enhancements.

**1. Control Group (Standard Baselines)**
* **$1/N$ (Equal Weight):** The naive, zero-information allocation benchmark.
* **Standard MVO:** Unconstrained Mean-Variance Optimization. Assumes zero estimation error in the Ridge regression view vector $\mathbf{q}$ (effectively strictly enforcing $\mathbf{\Omega} \to \mathbf{0}$).
* **Standard BL:** The standard Black-Litterman framework utilizing standard heuristic calibrations for the scalar $\tau$ and the uncertainty matrix $\mathbf{\Omega}$.

**2. Experimental Group (Proposed Enhancements)**
* **Proposed 1 (Baseline $\mathbf{\Omega}$):** A data-driven formulation where the diagonal elements of $\mathbf{\Omega}$ are strictly defined by the residual variance ($\sigma^2_{\epsilon}$) of the Ridge estimation, quantifying specific forecast noise.
* **Proposed 2 (Fixed $\kappa$):** Introduces a static hyperparameter ($\kappa = 0.25$) to explicitly calibrate the signal-to-noise ratio between the macroeconomic views and the market equilibrium prior.
* **Proposed 3 (Dynamic WFO $\mathbf{\Omega}$):** An adaptive confidence model. Utilizes Walk-Forward Optimization (WFO) to periodically solve $\arg\max_{\kappa} \text{Sharpe Ratio}$ over a rolling 24-month lookback, allowing the confidence scalar to dynamically adjust to shifting volatility regimes.

**Experimental Control: Sample Covariance**
To maintain rigorous causal inference, the prior risk model ($\mathbf{\Sigma}$) is strictly locked to the standard sample covariance estimator. By fixing the risk model and the signal generation ($\mathbf{q}$), we isolate the marginal performance contribution of our $\mathbf{\Omega}$ specifications. 


```bash
python run_experiment_br.py

```


---

#### **B. Covariance Robustness & Grid Search**

This section isolates the impact of different covariance estimators to address estimation noise. We introduce L2 Regularization and Shrinkage estimators to mitigate the ill-conditioning and estimation error inherent in the sample $\mathbf{\Sigma}$

There are two distinct tests:

* **Experiment 1 (Covariance Comparison)**: Directly compares Sample Covariance, L2 Regularized Covariance (penalty = 0.10), and Shrinkage Covariance under a fixed $\kappa = 0.25$ and shrinkage strength $s = 0.35$. You can run the dedicated standalone script for this:
```bash
python run_covariance_fixed_kappa.py

```


* **Experiment 2 (Shrinkage Grid Search)**: Iterates over various shrinkage strengths ($s \in \{0.25, 0.50, 0.75, 1.00\}$) to evaluate the optimal trade-off between the sample structure and the target matrix.
```bash
python Shrinkage_grid_search_fixed_kappa.py

```

*(Note: Both experiments are also housed within the `Shrinkage_grid_search_fixed_kappa.py` file. If you prefer to run Experiment 1 from there instead of the standalone script, simply comment/uncomment the corresponding functions in the `__main__` execution block at the bottom of the script).*



#### **C. The Integrated Robust Model**

**`run_covariance_dynamic_bl.py`**: The ultimate robustness test. It combines the **Walk-Forward Dynamic $\Omega$** (optimizing $\kappa$ from a grid of `{0.1, 0.25, 0.5, 0.75, 1.0}`) with the robust covariance estimators (Sample vs. L2 vs. Shrinkage with constant shrinkage strength $s = 0.35$) to evaluate the most advanced iteration of the model.

```bash
python run_covariance_dynamic_bl.py

```

> [!IMPORTANT]
> All results, including the Evaluation Matrix (`.xlsx`) and PDF charts, are automatically exported to specific folders generated during execution (e.g., `/result/`, `/result_shrinkage_grid/`, or `/result_dynamic_cov_backtest/`).

---


## 📈 Performance Evaluation

Upon execution, the engine automatically generates a comprehensive evaluation matrix saved to the `result/` directory, including:

* Annualized Return & Volatility
* Sharpe Ratio & Information Ratio
* Maximum Drawdown & 95% Value at Risk (VaR)
* Annualized Turnover
* Automated plotting for Cumulative Returns (Log Scale), Underwater Drawdown charts, and Asset Allocation Stackplots.


---

## 🏁 Conclusion & Key Findings

This project demonstrates the empirical advantages of integrating macroeconomic signals into a structured **Black-Litterman** framework. By moving away from static optimization and incorporating robust risk-management techniques, the following conclusions can be drawn:

* **Superior Stability over MVO**: The Black-Litterman model effectively anchors the portfolio to market equilibrium, preventing the "extreme" and concentrated asset weights typical of standard Mean-Variance Optimization (MVO).
* **Adaptability via Walk-Forward Optimization (WFO)**: The dynamic tuning of the confidence parameter $\kappa$ allows the model to adapt to shifting market regimes. By monitoring historical Sharpe ratios in a 24-month rolling window, the model automatically determines when to trust quantitative views over the market prior.
* **Impact of Robust Covariance**: Implementing **Ledoit-Wolf Shrinkage** and **L2 Regularization** significantly mitigates "estimation noise" in high-dimensional settings. Grid search results indicate that higher shrinkage strengths ($s \ge 0.50$) often yield more stable weights and lower realized volatility during high-stress periods.
* **Significance of Macro Signals**: The use of lagged features (Growth, Inflation, Rates, and Credit) provides a statistically valid foundation for generating alpha views without introducing look-ahead bias. Ridge regression effectively handles the multicollinearity often present in macroeconomic data.
* **Operational Realism**: By incorporating an $L_1$ turnover penalty and a transaction cost rate of **0.1%**, the backtest reflects realistic net performance. The framework proves that active management can be profitable even after accounting for frequent monthly rebalancing costs.

---
