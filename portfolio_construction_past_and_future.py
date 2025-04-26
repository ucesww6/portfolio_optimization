import sys

from scipy.signal import wiener

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
        iteration=0
        lambda_value=0.5 #initial guess
        optimization_variance_figure = []
        lambda_values_list = [lambda_value]

        while iteration<= max_iteration:
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
            optimization_variance_figure = optimization_variance_figure+[ex_ante_variance]
            # Based on if this number is higher or lower than the risk parameter, we will do adjustment
            # The idea of adjustment is similar to Gradient descent, where we calculate the change of risk vs. lambda
            # Since lambda is not a part of the risk calculation (or variance in mean-variance framework), we use the
            # first principle: change of risk/change of lambda.
            print (abs(risk_target**2-ex_ante_variance))
            if abs(risk_target**2-ex_ante_variance) <= tolerance_level:
                # No need to do anything. it is satisfied
                break
            else:
                if iteration ==0:
                    # if first iteration, we change lambda only tiny to find the first derivative
                    lambda_value_step = lambda_value/10 # 10%, just repeat
                    # lambda_value_step = -(risk_target ** 2 - ex_ante_variance) * lambda_value / ex_ante_variance
                    current_ex_ante_variance = optimization_variance_figure[iteration]
                    if current_ex_ante_variance > risk_target**2:
                        lambda_value = lambda_value + lambda_value_step
                    else:
                        lambda_value = lambda_value - lambda_value_step
                    iteration = iteration + 1
                    print(iteration)
                else:
                    # so if iteration >=1, it means that we can use gradient descent
                    current_ex_ante_variance=optimization_variance_figure[iteration]
                    previous_ex_ante_variance = optimization_variance_figure[iteration-1]
                    gradient = (current_ex_ante_variance-previous_ex_ante_variance)/lambda_value_step
                    gap = risk_target**2-current_ex_ante_variance
                    lambda_value_step = gap/gradient
                    lambda_value = lambda_value + lambda_value_step
                    lambda_values_list = lambda_values_list+[lambda_value]
                    iteration = iteration + 1
                    print(lambda_values_list)
                    print(optimization_variance_figure)

        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')

    return portfolio_weights_all_periods



def mean_variance_portfolio_different_expectation_scales_with_limit_risk(pc):
    """
    This is an example to show that the solution can be very different if you have a very different solution
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
        iteration=0
        lambda_value=0.5 #initial guess

        while iteration<= max_iteration:
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
                iteration=max_iteration+1
                break
            else:
                lambda_value_step = -(risk_target**2-ex_ante_variance)*lambda_value/ex_ante_variance
                lambda_value = lambda_value+lambda_value_step
                iteration=iteration+1
                print(iteration)
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')

    return portfolio_weights_all_periods


# Limit Risk Constraint
def max_return_with_risk_target(pc):
    """
    Suitable for typical investors with some risk-taking appetite and understand what expected returns and risks mean
    :return:
    """
    print ('Working on Max Returns with Risk Target')
    number_of_days_to_estimate = 252*3
    price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv'
    prices = pd.read_csv(price_url, index_col=0)
    portfolio_weights_all_periods = pd.DataFrame()
    risk_target = 0.1 # 10%
    expected_returns_dict_dataframe = prices.pct_change().mean()
    # var_cov_matrix = pc.use_statistical_risk_model(asset_name=[], read_price_from_file=True, price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
    # return_forecast, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=expected_returns_dict_dataframe, var_cov_matrix=var_cov_matrix)

    for i in range(len(prices)-number_of_days_to_estimate+1):
        prices_period = prices.iloc[i:i+number_of_days_to_estimate].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=[],
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
        _, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=None, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = prices.pct_change().mean().to_dict()

        params = {'01_expected_returns': {'expected_returns': {'expected_returns_dict': expected_returns_dict_dataframe,
                                                               'common_assets': (['HMOP', 'HYGH', 'VGT'])}},
                  '02_budget_constraint': {'budget_constraint': 1},
                  '03_long_only_constraint': {'long_only_constraint': True},
                  '04_limit_risk_constraint': {'limit_risk_constraint': risk_target},
                  '05_maximum_limit_holding_constraint': {'maximum_limit_holding_constraint': 1},
                  '06_type_of_optimization': {'type_of_optimization': 'maximize'}}

        pc.run_optimization(params)
        # calculating ex-ante risk based on the risk model
        # pc.run_optimization(params)
        assets = common_assets
        weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
        weights.columns = [prices_period.index[-1]]
        # print (f'Minimum Risk Portfolio Holdings: {weights}')
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')
    return portfolio_weights_all_periods



# Limit Risk Constraint
def max_return_with_risk_target_and_additional_constraints(pc):
    """
    Suitable for typical investors with some risk-taking appetite and understand what expected returns and risks mean.
    We add more constraints to make the example realistic
    :return:
    """
    print ('Working on Max Returns with Risk Target')
    number_of_days_to_estimate = 252*3
    price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv'
    prices = pd.read_csv(price_url, index_col=0)
    portfolio_weights_all_periods = pd.DataFrame()
    risk_target = 0.1 # 10%

    for i in range(len(prices)-number_of_days_to_estimate+1):
        prices_period = prices.iloc[i:i+number_of_days_to_estimate].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=[],
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
        _, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=None, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = prices.pct_change().mean().to_dict()

        params = {'01_expected_returns': {'expected_returns': {'expected_returns_dict': expected_returns_dict_dataframe,
                                                               'common_assets': (['HMOP', 'HYGH', 'VGT'])}},
                  '02_budget_constraint': {'budget_constraint': 1},
                  '03_long_only_constraint': {'long_only_constraint': True},
                  '04_limit_risk_constraint': {'limit_risk_constraint': risk_target},
                  '05_maximum_limit_holding_constraint': {'maximum_limit_holding_constraint': 0.05},
                  '06_type_of_optimization': {'type_of_optimization': 'maximize'}}

        pc.run_optimization(params)
        # calculating ex-ante risk based on the risk model
        # pc.run_optimization(params)
        assets = common_assets
        weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
        weights.columns = [prices_period.index[-1]]
        # print (f'Minimum Risk Portfolio Holdings: {weights}')
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')
    return portfolio_weights_all_periods


# Limit Risk Constraint
def max_return_with_max_sharpe(pc):
    """
    Max Sharpe, focusing on Long Only Portfolio. It is possible to also use iteration to determine the value.
    We use certain transformation and certain assumption in this approach in order to reduce the problem to a standard
    convex optimization. The problem is a standard
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
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=[],
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
        _, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=None, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = prices.pct_change().mean().to_dict()

        # input dicts
        params={'01_portfolio_variance': {'portfolio_variance':True},
                '02_maximum_weighted_average_alphas': {'maximum_limit_weighted_average':
                                                           {'attribute': list(expected_returns_dict_dataframe.values()),
                                                            'maximum_avg_weight': 1.0}},
                '03_minimum_weighted_average_alphas': {
                    'minimum_limit_weighted_average': {'attribute':  list(expected_returns_dict_dataframe.values()),
                                                       'minimum_avg_weight': 1.0}},
                '04_long_only_constraint': {'long_only_constraint': True},
                '04_type_of_optimization':{'type_of_optimization': 'minimize'}}

        pc.run_optimization(params)
        assets = common_assets
        weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
        weights.columns = [prices_period.index[-1]]
        # rescale to 1
        weights = weights/weights.sum()
        # print (f'Minimum Risk Portfolio Holdings: {weights}')
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')
    return portfolio_weights_all_periods



# Inverse Volatility
def inverse_volatility(pc):
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
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=[],
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
        _, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=None, var_cov_matrix=var_cov_matrix)
        # diagonal
        weights_diagonal = np.diag(var_cov_matrix)
        weights = weights_diagonal/weights_diagonal.sum()
        weights = pd.DataFrame(weights)
        weights.index = var_cov_matrix.index
        weights.columns = [prices_period.index[-1]]
        # print (f'Minimum Risk Portfolio Holdings: {weights}')
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')
    return


# Inverse Volatility
def equal_weight(pc):
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
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=[],
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
        _, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=None, var_cov_matrix=var_cov_matrix)
        # diagonal
        weights_diagonal = np.diag(var_cov_matrix)
        weights = weights_diagonal/weights_diagonal.sum()
        weights = pd.DataFrame(weights)
        weights.index = var_cov_matrix.index
        weights.columns = [prices_period.index[-1]]
        weights[prices_period.index[-1]] = 1/len(weights)
        # print (f'Minimum Risk Portfolio Holdings: {weights}')
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')
    return

# Example 1 - Equal Weight
# portfolio_weights_all_periods = equal_weight(pc)

# Example 2 - Minimum Risk
# single period
# portfolio_weights_all_periods = minimum_risk_portfolio(pc)
# backtest

# Example X - Mean Variance
# portfolio_weights_all_periods= mean_variance_portfolio(pc)

# Example X - How to use Mean-Variance to Do a Limit Risk Constraint
portfolio_weights_all_periods= mean_variance_portfolio_with_limit_risk(pc)

# max_returns_with_risk_target
# portfolio_weights_all_periods= max_return_with_risk_target(pc)


# Equal Weight
# Nothing to examine

# Max-Sharpe
# portfolio_weights_all_periods= max_return_with_max_sharpe(pc)

# Risk-Parity

# Inverse Volatility
# portfolio_weights_all_periods = inverse_volatility(pc)



# Example X - How to use Mean-Variance to Do a Limit Risk Constraint
# portfolio_weights_all_periods= max_return_with_risk_target(pc)

