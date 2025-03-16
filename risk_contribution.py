import numpy as np
import pandas as pd

class RiskContribution:
    def __init__(self, asset_exposure=None, asset_covariance=None):
        self.asset_exposure = asset_exposure
        self.asset_covariance = asset_covariance

    @property
    def asset_variance(self):
        # Calculate asset variance based on asset covariance
        return pd.DataFrame(np.diag(self.asset_covariance), index=self.asset_covariance.index,
                            columns=self.asset_exposure.columns)

    @property
    def asset_volatility(self):
        # Calculate asset volatility based on asset variance
        return self.asset_variance ** 0.5

    @property
    def x_sigma(self):
        # Calculate volatility attribution
        return self.asset_exposure * self.asset_volatility

    @property
    def portfolio_volatility(self):
        # Calculate portfolio volatility
        return np.sqrt(
            self.asset_exposure.T.dot(self.asset_covariance).dot(self.asset_exposure)
        ).values[0]

    @property
    def covariance_times_exposure(self):
        # Calculate covariance times exposure
        return self.asset_covariance.dot(self.asset_exposure)

    @property
    def asset_vol_times_portfolio_vol(self):
        # Calculate asset volatility times portfolio volatility
        return self.portfolio_volatility * self.asset_volatility

    @property
    def asset_portfolio_correlation(self):
        # Calculate asset-portfolio correlation
        return self.covariance_times_exposure / self.asset_vol_times_portfolio_vol

    @property
    def x_sigma_rho(self):
        # Calculate x_sigma times asset-portfolio correlation
        return self.x_sigma * self.asset_portfolio_correlation

    @property
    def marginal_contribution_to_risk(self):
        # Calculate marginal contribution to risk
        return self.asset_covariance.dot(self.asset_exposure) / self.portfolio_volatility

    @property
    def asset_beta(self):
        # Calculate asset beta
        return self.marginal_contribution_to_risk / self.portfolio_volatility

    @property
    def total_risk_contribution(self):
        # Calculate total risk contribution
        return self.asset_exposure * self.marginal_contribution_to_risk

    @property
    def percentage_risk_contribution(self):
        # Calculate percentage risk contribution
        return self.total_risk_contribution / self.portfolio_volatility

    @property
    def var_90(self):
        # Calculate 90% value at risk (VaR)
        return 1.282 * self.portfolio_volatility

    @property
    def var_95(self):
        # Calculate 95% VaR
        return 1.645 * self.portfolio_volatility

    @property
    def var_99(self):
        # Calculate 99% VaR
        return 2.326 * self.portfolio_volatility