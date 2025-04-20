import sys
sys.path.append(r'C:\Users\Walter\Desktop\mimicry\utils/')
import numbers
import pandas as pd
import numpy as np
import cvxpy
# Example - Return maximization with Variance Constraint
import cvxpy as cp
import random
# import warnings
import warnings
import sys
import numbers
import cvxpy as cp
from timeit import default_timer as timer
import yfinance as yf
import random
import warnings
import cvxpy as cvx
import portfolio_construction
import risk_contribution
import portfolio_analytics


data = pd.read_csv(r'C:\Users\Walter\Desktop\portfolio_construction/etf_sector_info.csv', index_col=0)
portfolio_symbols = data['symbol'].to_list()
category = data[['symbol', 'category']].set_index('symbol')
pc = portfolio_construction.portfolio_construction(portfolio_symbols)
#
# max_return_with_risk_target(pc)
# max_return_with_risk_target_plus_beta_constraint_no_relax(pc)
# max_return_with_risk_target_plus_beta_constraint_relax(pc)
# max_return_with_risk_target_plus_beta_dynamic_constraint_relax(pc)
# portfolio_exposures(pc)
# max_return_with_risk_target_shares_output(pc)
def portfolio_analysis(asset_weights, asset_returns, lag=1):
    # Various portfolio output
    return


def minimum_risk_portfolio(pc):
    """
    Suitable for investors with low and/or minimum risk tolerance.
    :param asset_name:
    :return:
    """
    print ('Working on Min Risk Portfolio')
    # Backtest
    # We use Rolling 252 days to estimate variance covariance matrix
    number_of_days_to_estimate = 252*3
    price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv'
    prices = pd.read_csv(price_url, index_col=0)
    portfolio_weights_all_periods = pd.DataFrame()
    for i in range(len(prices)-number_of_days_to_estimate+1):
        prices_period = prices.iloc[i:i+number_of_days_to_estimate].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        asset_name = ['AGZ', 'AIRR', 'ANGL', 'AOA', 'ASHX', 'ATMP', 'BAB', 'BBAX', 'BBJP']
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
        _, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=None, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)

        # input dicts
        params={'01_portfolio_variance': {'portfolio_variance':True},
                '02_budget_constraint': {'budget_constraint': 1},
                '03_long_only_constraint': {'long_only_constraint': True},
                '04_type_of_optimization':{'type_of_optimization': 'minimize'}}

        pc.run_optimization(params)
        assets = common_assets
        weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
        weights.columns = [prices_period.index[-1]]
        # print (f'Minimum Risk Portfolio Holdings: {weights}')
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')
    return portfolio_weights_all_periods


def mean_variance_portfolio(pc):
    """
    Suitable for investors with low and/or minimum risk tolerance.
    :param asset_name:
    :return:
    """
    print ('Working on Mean-Variance Portfolio')
    # Backtest
    # We use Rolling 252 days to estimate variance covariance matrix
    number_of_days_to_estimate = 252*3
    price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv'
    prices = pd.read_csv(price_url, index_col=0)
    portfolio_weights_all_periods = pd.DataFrame()
    for i in range(len(prices)-number_of_days_to_estimate+1):
        prices_period = prices.iloc[i:i+number_of_days_to_estimate].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        # asset_name = ['AGZ', 'AIRR', 'ANGL', 'AOA', 'ASHX', 'ATMP', 'BAB', 'BBAX', 'BBJP']
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=[],
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
        _, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=None, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = prices.pct_change().mean().to_dict()

        params = {'01_mean_variance': {
            'mean_variance': {'expected_returns_dict': expected_returns_dict_dataframe,
                              'common_assets': (['HMOP', 'HYGH', 'VGT']),
                              'lambda_value': 0.5}},
            '02_budget_constraint': {'budget_constraint': 1},
            '03_long_only_constraint': {'long_only_constraint': True},
            '06_type_of_optimization': {'type_of_optimization': 'maximize'}}

        pc.run_optimization(params)
        # calculating ex-ante risk based on the risk model


        assets = common_assets
        weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
        weights.columns = [prices_period.index[-1]]

        # print (f'Minimum Risk Portfolio Holdings: {weights}')
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')

    return portfolio_weights_all_periods


def mean_variance_portfolio_with_limit_risk(pc):
    """
    Suitable for investors with low and/or minimum risk tolerance.
    :param asset_name:
    :return:
    """
    print ('Working on Mean-Variance Portfolio')
    # Backtest
    # We use Rolling 252 days to estimate variance covariance matrix
    number_of_days_to_estimate = 252*3
    price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv'
    prices = pd.read_csv(price_url, index_col=0)
    portfolio_weights_all_periods = pd.DataFrame()
    risk_target = 0.1 # 10%

    for i in range(len(prices)-number_of_days_to_estimate+1):
        prices_period = prices.iloc[i:i+number_of_days_to_estimate].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        # asset_name = ['AGZ', 'AIRR', 'ANGL', 'AOA', 'ASHX', 'ATMP', 'BAB', 'BBAX', 'BBJP']
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=[],
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
        _, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=None, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = prices.pct_change().mean().to_dict()


        # To find the limit risk, we will require to iterate through the mean-variance framework using lambda value
        # given different lambda, we will have a different variance figure
        tolerance_level = 10**-6
        max_iteration = 100 # maximum iteration for lambda_value
        i=0
        lambda_value=0.5 #initial guess

        while i<= max_iteration:
            params = {'01_mean_variance': {
                'mean_variance': {'expected_returns_dict': expected_returns_dict_dataframe,
                                  'common_assets': (['HMOP', 'HYGH', 'VGT']),
                                  'lambda_value': lambda_value}},
                '02_budget_constraint': {'budget_constraint': 1},
                '03_long_only_constraint': {'long_only_constraint': True},
                '06_type_of_optimization': {'type_of_optimization': 'maximize'}}
            pc.run_optimization(params)
            # calculating ex-ante risk based on the risk model
            assets = common_assets
            weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
            weights.columns = [prices_period.index[-1]]
            ex_ante_variance =  np.dot(np.dot(weights.T, var_cov_matrix), weights)[0][0]*252
            # Based on if this number is higher or lower than the risk parameter, we will do adjustment
            print (abs(risk_target**2-ex_ante_variance))
            if abs(risk_target**2-ex_ante_variance) <= tolerance_level:
                i=max_iteration+1
                break
            else:
                lambda_value_step = -(risk_target**2-ex_ante_variance)*lambda_value/ex_ante_variance
                lambda_value = lambda_value+lambda_value_step
                i=i+1
                print(i)
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')

    return portfolio_weights_all_periods

# Example 1 - Equal Weight

# Example 2 - Minimum Risk
# single period
# portfolio_weights_all_periods = minimum_risk_portfolio(pc)
# backtest

# Example X - Mean Variance
# portfolio_weights_all_periods= mean_variance_portfolio(pc)

# Example X - How to use Mean-Variance to Do a Limit Risk Constraint
portfolio_weights_all_periods= mean_variance_portfolio_with_limit_risk(pc)


# Equal Weight
# Nothing to examine

# Max-Sharpe

# Risk-Parity


# Inverse Volatility
def inverse_volatility(pc):
    """
    Suitable for investors with low and/or minimum risk tolerance.
    :param asset_name:
    :return:
    """
    print ('Working on Mean-Variance Portfolio')
    # Backtest
    # We use Rolling 252 days to estimate variance covariance matrix
    number_of_days_to_estimate = 252*3
    price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv'
    prices = pd.read_csv(price_url, index_col=0)
    portfolio_weights_all_periods = pd.DataFrame()
    for i in range(len(prices)-number_of_days_to_estimate+1):
        prices_period = prices.iloc[i:i+number_of_days_to_estimate].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        # asset_name = ['AGZ', 'AIRR', 'ANGL', 'AOA', 'ASHX', 'ATMP', 'BAB', 'BBAX', 'BBJP']
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=[],
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
        _, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=None, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = prices.pct_change().mean().to_dict()

        params = {'01_mean_variance': {
            'mean_variance': {'expected_returns_dict': expected_returns_dict_dataframe,
                              'common_assets': (['HMOP', 'HYGH', 'VGT']),
                              'lambda_value': 0.5}},
            '02_budget_constraint': {'budget_constraint': 1},
            '03_long_only_constraint': {'long_only_constraint': True},
            '06_type_of_optimization': {'type_of_optimization': 'maximize'}}

        pc.run_optimization(params)
        # calculating ex-ante risk based on the risk model


        assets = common_assets
        weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
        weights.columns = [prices_period.index[-1]]

        # print (f'Minimum Risk Portfolio Holdings: {weights}')
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')

    return portfolio_weights_all_periods


# Limit Risk Constraint
def max_return_with_risk_target(pc):
    """
    Suitable for typical investors with some risk-taking appetite and understand what expected returns and risks mean
    :return:
    """
    print ('Working on Max Returns with Risk Target')
    # return_forecast = data[['symbol', 'threeYearAverageReturn']].set_index('symbol')
    return_forecast = pd.DataFrame({'HMOP': 0.0,'HYGH': -0.03,'VGT': 0.05}.values(), {'HMOP': 0.0,'HYGH': -0.03,'VGT': 0.05}.keys())
    return_forecast.columns = ['values']
    var_cov_matrix = pc.use_statistical_risk_model(asset_name=[], read_price_from_file=True, price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
    return_forecast, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=return_forecast, var_cov_matrix=var_cov_matrix)

    pc.portfolio_holdings(common_assets)
    pc.relax_constraint_version='beta'
    params={'01_expected_returns': {'expected_returns': {'expected_returns_dict': {'HMOP': 0.0,'HYGH': -0.03,'VGT': 0.05},
                                                  'common_assets': (['HMOP', 'HYGH', 'VGT'])}},
     '02_budget_constraint': {'budget_constraint': 1},
     '03_long_only_constraint': {'long_only_constraint': True},
     '04_limit_risk_constraint': {'limit_risk_constraint': 0.3},
     '05_maximum_limit_holding_constraint': {'maximum_limit_holding_constraint': 0.05},
     '06_type_of_optimization': {'type_of_optimization': 'maximize'}}

    # pc.run_optimization(params)
    from collections import OrderedDict
    available_constraints = list(params.keys())
    for i in range(len(available_constraints)-1):
        tmp_constraints = available_constraints[:i+1]
        # update params_tmp
        params_tmp = OrderedDict()
        for t in tmp_constraints:
            params_tmp[t] = params[t]
        reformulated_optimization = OrderedDict()
        # set maximize or minimize first
        reformulated_optimization['adding_type_of_optimization'] = {'type_of_optimization': 'maximize'}
        # add constraints
        if 'budget_constraint' not in tmp_constraints:
            reformulated_optimization['adding_budget_constraint'] = {'budget_constraint':1}
        if 'long_only_constraint' not in tmp_constraints:
            reformulated_optimization['adding_long_only_constraint'] = {'long_only_constraint':True}
        for e in tmp_constraints:
            reformulated_optimization[e] = params[e]
        pc.run_optimization(reformulated_optimization)

    assets = common_assets
    weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
    print (f'Minimum Risk Portfolio Holdings: {weights}')
    return weights



#
# #-----------------------------------------------------------------------------------------------------------------------------------
# # Results Check - Portfolio Analytics to check constraint
# print (weights)
# weights_total = weights.sum()[0]
# asset_holding = weights.max()[0]
# long_only = (weights>=0).sum()[0]
# risk = np.dot(np.dot(weights.T, var_cov_matrix), weights)[0][0]**0.5
# industry_exposures = (weights.join(asset_classification)['industry'] == 'Industry_0').sum()/len(weights)
#
# # example 2
# # Classical (Markowitz) portfolio optimization solves the optimization problem
# mean_variance = expected_returns(expected_returns_dataframe)-0.5*cp.quad_form(portfolio_holdings, var_cov_matrix)
# objective = cp.Maximize(mean_variance)
# constraints = [maximum_individual_asset_holding_constraints,maximum_industry_investment_constraints,long_only_portfolio,
#                minimum_holding_of_one_asset,
#                fully_invested]
#
# portfolio_optimizations(objective, constraints)
# weights = pd.DataFrame(portfolio_holdings.value, index=assets, columns =['Weights'])