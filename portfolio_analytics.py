import cvxpy as cp
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import yfinance as yf
import random
import warnings
import sys
import numpy as np
import yfinance as yf
from utils.cvx_group.cvx_pca import pca
from utils.cvx_group.cvx_factor import FactorModel
import pandas as pd
from utils.perf_stats import annualize_rets
from utils.build_factor_cov import BuildFactorCovariance
from itertools import combinations
import risk_contribution
class portfolio_analytics:
    def __init__(self):
        print ('Portfolio Analytics')
        self.maximum_number_of_display=5 # show top 5 and bottom 5

    def calculate_weighted_avg(self, dataframe_a, dataframe_b):
        dataframe_a, dataframe_b = self.align_two_dataframe(dataframe_a, dataframe_b)
        weighted_avg = pd.DataFrame(dataframe_a.values*dataframe_b.values)
        weighted_avg.index = dataframe_b.index
        weighted_avg.columns = dataframe_b.columns
        weighted_avg = weighted_avg.sum()
        return weighted_avg

    def align_two_dataframe(self, dataframe_a, dataframe_b):
        common_index = dataframe_a.index.intersection(dataframe_b.index)
        dataframe_a = dataframe_a.loc[common_index]
        dataframe_b = dataframe_b.loc[common_index]
        return dataframe_a, dataframe_b

    def process_classification(self, classification):
        """
        :param exposures_classification:
        :return:
        """
        classification = pd.get_dummies(classification)
        return classification

    def calculate_classification_exposures(self, asset_holdings, classification):
        # portfolio analytics
        # convert to dummy
        classification = self.process_classification(classification)
        asset_holdings, classification = self.align_two_dataframe(asset_holdings, classification)
        exposures = self.calculate_weighted_avg(asset_holdings, classification)
        return exposures

    def calculate_ex_post_risk(self, weight_time_series_daily):
        daily_ex_post_risk = weight_time_series_daily.std()
        annualized_ex_post_risk = daily_ex_post_risk*252**0.5
        return daily_ex_post_risk, annualized_ex_post_risk

    def risk_analytics(self, asset_exposure, asset_covariance):
        ra = risk_contribution.RiskContribution(asset_exposure,asset_covariance)
        asset_volatility = ra.asset_volatility
        portfolio_volatility = ra.portfolio_volatility
        asset_beta = ra.asset_beta
        percentage_risk_contribution = ra.percentage_risk_contribution
        summary = {'asset_volatility': asset_volatility,
                   'portfolio_volatility': portfolio_volatility,
                   'asset_beta': asset_beta,
                   'percentage_risk_contribution': percentage_risk_contribution
                   }
        return summary

    def final_output(self, asset_holdings, classification,asset_covariance):
        summary={'classification_exposures': self.calculate_classification_exposures(asset_holdings, classification),
                 'risk_analytics': self.risk_analytics(asset_holdings, asset_covariance)}
        return summary


