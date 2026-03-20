# Dynamic-Black-Litterman-Portfolio-Construction

## 📖 Overview

This repository implements a highly modular, end-to-end quantitative backtesting engine for multi-asset portfolio construction. It extends the traditional Black-Litterman (BL) model by dynamically integrating macroeconomic signals via Machine Learning (Ridge Regression) and employing advanced risk-management techniques, including Walk-Forward Optimization (WFO) for view confidence ($\Omega$) and robust covariance estimation (Shrinkage & L2 Regularization).

The goal is to address the notorious "error-maximizing" nature of standard Mean-Variance Optimization (MVO) by producing stable, theoretically sound, and transaction-cost-aware portfolio weights.

## 🎯 Core Research Problem 

How can we construct stable and robust portfolios when the investment universe is highly correlated, leading to estimation error and instability in classical mean-variance optimization?


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

### 📊 Correlation Structure of the Asset Universe

To better understand the dependence structure of the portfolio universe, we compute return correlations for ETFs, individual stocks, and the combined asset universe, and visualize the cross-sectional relationships using correlation heatmaps. These assets define the investment universe for portfolio construction.

To mitigate look-ahead bias, the correlation analysis is conducted using historical data from **2005–2019**, while the portfolio models are evaluated over **2020–2025** using a rolling training framework.

The correlation results reveal several important characteristics of the universe:

- **Strong equity clustering:** Most equity ETFs and large-cap stocks exhibit high positive correlations, indicating substantial overlap in market exposure.  
- **Sector concentration:** Technology-related assets such as QQQ, XLK, AAPL, MSFT, and NVDA form a particularly tight cluster, suggesting strong common factor exposure.  
- **Limited diversification within equities:** Although the universe spans multiple sectors and regions, many assets remain highly correlated, reducing the true diversification benefit of naive allocation.  
- **TLT as a diversifier:** Long-duration U.S. Treasuries show low or negative correlation with most equity assets, making them one of the few meaningful hedging instruments in the universe.  
- **Implication for portfolio optimization:** The high-correlation structure makes the sample covariance matrix more likely to be ill-conditioned, which can destabilize classical Mean-Variance Optimization (MVO) and amplify estimation error.  

This correlation analysis motivates the central research problem of the project: constructing stable and robust portfolios in a highly correlated asset universe, where noisy expected returns and unstable covariance estimation can lead to extreme and unreliable portfolio weights.


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

## 🧪 3. Execution Scripts (Experimental Variables)

Each script isolates specific components of the framework to evaluate model robustness and the impact of estimation error mitigation.

---

#### **A. Strategy Evaluation & Benchmarking**
`backtest_omega_methods_sample_cov.py`
This script performs a comparative backtest across six portfolio formulations to isolate the marginal impact of different uncertainty ($\mathbf{\Omega}$) specifications.

**1. Control Group (Standard Baselines)**
* **$1/N$ (Equal Weight):** A naive, zero-information allocation benchmark.
* **Standard MVO:** Unconstrained Mean-Variance Optimization. Assumes zero estimation error in the Ridge views ($\mathbf{\Omega} \to \mathbf{0}$).
* **Standard BL:** The classical Black-Litterman framework using heuristic calibrations for $\tau$ and $\mathbf{\Omega}$.

**2. Experimental Group (Proposed Enhancements)**
* **Proposed 1 (Baseline $\mathbf{\Omega}$):** A data-driven formulation where the diagonal of $\mathbf{\Omega}$ is defined by the residual variance ($\sigma^2_{\epsilon}$) of the Ridge estimation.
* **Proposed 2 (Fixed $\kappa$):** Introduces a static hyperparameter ($\kappa = 0.25$) to calibrate the signal-to-noise ratio.
* **Proposed 3 (Dynamic WFO $\mathbf{\Omega}$):** An adaptive model using Walk-Forward Optimization (WFO) to solve $\arg\max_{\kappa} \text{Sharpe Ratio}$ over a rolling 24-month lookback window, allowing the confidence scalar $\kappa$ to dynamically adjust to changing market conditions.

> **Experimental Control:** The risk model ($\mathbf{\Sigma}$) is fixed to the **Sample Covariance** estimator to ensure causal interpretability of $\mathbf{\Omega}$ specifications.

```bash
python backtest_omega_methods_sample_cov.py
```
*Results are exported to the `results_omega_methods_sample_cov_backtest` folder.*

---

#### **B. Covariance Robustness & Grid Search**
This section evaluates the impact of covariance estimation methods in addressing estimation noise within the risk matrix $\mathbf{\Sigma}$.

**Experiment 1: Covariance Comparison**
Compares the following estimators under a fixed $\kappa = 0.25$:
* **Sample Covariance**
* **L2-Regularized Covariance** (penalty = 0.10)
* **Shrinkage Covariance** (shrinkage strength $s = 0.35$)

```bash
python backtest_covariance_fixed_kappa.py
```
*Results are exported to the `results_fixed_kappa_cov_backtest` folder.*

**Experiment 2: Shrinkage Grid Search**
Evaluates the sensitivity of the portfolio to varying shrinkage intensities:
$$s \in \{0.25, 0.50, 0.75, 1.00\}$$

```bash
python backtest_shrinkage_grid_search_fixed_kappa.py
```
*Results are exported to the `results_fixed_kappa_shrinkage_cov_grid` folder.*

---

#### **C. The Integrated Robust Model**
`backtest_covariance_dynamic_kappa.py`
This represents the most advanced configuration, evaluating the joint impact of dynamic confidence calibration and covariance stabilization.

* **Dynamic $\mathbf{\Omega}$ (via WFO):** $\kappa \in \{0.1, 0.25, 0.5, 0.75, 1.0\}$.
* **Robust Covariance Estimators:** Comparative analysis of Sample, L2-regularized, and Shrinkage ($s = 0.35$) methods.

```bash
python backtest_covariance_dynamic_kappa.py
```
*Results are exported to the `results_dynamic_kappa_cov_backtest` folder.*

---

> [!IMPORTANT]
> All outputs—including **evaluation metrics (.xlsx)** and **visualization charts (PDF)**—are automatically saved to the corresponding result directories generated during execution.

---



## 📈 Performance Evaluation

Upon execution, the engine automatically generates a comprehensive evaluation matrix along with a suite of performance visualizations. The reported metrics include:

- **Annualized Return & Volatility**
- **Sharpe Ratio & Information Ratio**
- **Maximum Drawdown & 95% Value at Risk (VaR)**
- **Annualized Turnover**

The pipeline also produces:

- **Cumulative return curves (log scale)**
- **Underwater (drawdown) plots**
- **Asset allocation stack plots**


---

## 🏁 Conclusion & Key Findings

### 1. Research Contribution: Solving the "Error Maximizer" Problem
The primary contribution of this research is the development of a structurally robust portfolio optimization framework designed to address the **Estimation Error** inherent in traditional Mean-Variance Optimization (MVO). 

In quantitative finance, MVO is frequently characterized as an **"error maximizer"**. Because the MVO objective function treats input parameters—expected returns ($\mu$) and covariance ($\Sigma$)—as absolute truths, the optimizer systematically over-allocates capital to assets with the largest positive estimation errors in returns and the largest negative errors in risk. This vulnerability renders standard MVO highly unstable, leading to extreme risk profiles and uninvestable portfolios. 

By integrating **Macro Ridge Regression**, a **Dynamic Black-Litterman (BL)** confidence framework, and **regularized risk models**, this project establishes a methodology to filter statistical noise while preserving actionable macroeconomic alpha.

---

### 2. Quantitative Performance Analysis
The following table synthesizes the performance across all major strategies, comparing traditional benchmarks against our proposed enhancements.

#### **Comprehensive Evaluation Matrix**
| Metric | Equal Weight (1/N) | Benchmark 1 (MVO) | Benchmark 2 (Standard BL) | Dynamic BL (Sample) | Dynamic BL (L2 Reg) | Dynamic BL (Shrinkage) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Annualized Return** | 17.10% | **40.49%** | 26.51% | 20.94% | 16.17% | 22.36% |
| **Sharpe Ratio** | 1.145 | 1.161 | 1.096 | **1.182** | 1.085 | 1.171 |
| **Information Ratio** | N/A | 0.894 | 0.610 | 0.458 | -0.455 | **0.551** |
| **Maximum Drawdown** | **-20.29%** | -45.25% | -31.65% | -26.28% | -20.39% | -26.70% |
| **Annualized Volatility** | 14.94% | 34.88% | 24.18% | 17.73% | **14.90%** | 19.09% |
| **Value at Risk (95%)** | -5.270% | -12.73% | -10.59% | -8.32% | **-5.268%** | -8.55% |
| **Annualized Turnover** | 0.00% | 71.00% | 399.22% | 95.48% | **20.35%** | 96.22% |

**Dynamic BL:** A Black-Litterman model with dynamically calibrated view uncertainty, where $\kappa$ is optimized via Walk-Forward Optimization to adapt the signal-to-noise balance across evolving market regimes. 

The reported results are constructed from outputs in the `results_omega_methods_sample_cov_backtest` and `results_dynamic_kappa_cov_backtest` directories.

### 3. Does This Experiment Solve the MVO Problem?

#### **Addressing Signal Noise (MVO vs. Omega Methods)**
Standard MVO fails by treating noisy forecasts as deterministic. While the **Benchmark 1 (MVO)** in our test achieved a high annualized return of **40.49%**, it came at the cost of extreme volatility (**34.88%**) and a catastrophic maximum drawdown of **-45.25%**.
* **Stability:** The Omega-based methods (Black-Litterman) transform optimization into a probabilistic Bayesian problem. By explicitly defining forecast uncertainty ($\Omega$), the model prevents the "blind bets" characteristic of MVO.
* **Risk-Adjusted Efficiency:** **Dynamic BL (Sample)** achieved a superior Sharpe Ratio of **1.182**, outperforming both the $1/N$ baseline (**1.145**) and the standard MVO (**1.161**) while significantly reducing tail risk (MDD of -26.28% vs -45.25%).

#### **Addressing Risk Instability (Covariance Conditioning)**
The second failure of MVO is its sensitivity to the inverse of a noisy covariance matrix ($\Sigma^{-1}$). 
* **Instability of Standard BL:** Surprisingly, the **Benchmark 2 (Standard BL)** exhibited an annualized turnover of **399.22%**, indicating that heuristic confidence levels can sometimes amplify instability if the risk model is not properly conditioned.
* **Shrinkage Resolution:** Our proposed **Shrinkage Covariance** model stabilized the risk matrix by pulling off-diagonal elements toward a structured target. This allowed the **Dynamic BL (Shrinkage)** strategy to maintain a strong return of **22.36%** with an Information Ratio of **0.551**, providing the best balance between alpha capture and tail risk management (-26.70% MDD).
* **L2 Regularization:** Proved to be an effective "stability anchor," reducing turnover to a minimal **20.35%**, though it effectively "washed out" the macro signal, returning only **16.17%**.

---

### 4. Balanced Evaluation: Successes and Remaining Limitations

**What the Experiment Successfully Addresses:**
* **Tail Risk Mitigation:** Successfully reduced the extreme drawdowns of MVO (-45%) to much more manageable levels (-26%) while maintaining high absolute returns.
* **Alpha Generation:** Proved that **Macro Ridge Regression** views, when combined with Bayesian priors, can generate significant out-of-sample alpha (**+5.26%** annually over 1/N for the Shrinkage model).
* **Noise Filtering:** Successfully demonstrated that conditioning the risk matrix (Shrinkage) is essential for preventing the optimizer from exploiting spurious historical correlations.

**Realistic Caveats:**
* **Turnover Friction:** While lower than some benchmarks, an annualized turnover of **~96%** remains non-trivial. In live implementation, slippage and transaction costs would act as a persistent drag on the gross 5.26% alpha.
* **The "WFO Lag":** The Walk-Forward Optimization relies on a historical lookback to calibrate confidence, which may introduce a delay in adapting to sudden, unprecedented market shocks.

**Final Verdict:** The **Dynamic BL + Shrinkage Covariance** model is a superior architecture for professional applications. It successfully bridges the gap between macroeconomic theory and stable execution by replacing naive MVO inputs with Bayesian priors and conditioned risk estimates.

---
