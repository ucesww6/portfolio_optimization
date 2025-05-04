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
save_files_folder = r'C:\Users\Walter\Desktop\Review/'
asset_weights = pd.read_csv(r'C:\Users\Walter\Desktop\Review/mean_variance_portfolio.csv', index_col=0).T
price = pd.read_csv(r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv', index_col=0)
asset_returns = price.pct_change()
asset_returns.index = [x.split(' ')[0] for x in asset_returns.index]

etf_types_filter = ['Technology', 'Japan Stock', 'Miscellaneous Region',
       'Large Growth', 'Large Blend','Mid-Cap Blend',
       'China Region', 'Financial', 'Consumer Cyclical', 'Mid-Cap Growth',
       'Commodities Broad Basket',
       'Small Blend', 'Consumer Defensive',
       'Health','Natural Resources',
       'Trading--Leveraged Equity',
       'Large Value','Real Estate', 'Communications',
       'Foreign Large Blend', 'Foreign Large Value',
       'Energy Limited Partnership', 'Small Growth',
       'India Equity', 'Mid-Cap Value', 'Foreign Large Growth',
       'Small Value', 'Infrastructure',
       'Diversified Emerging Mkts', 'Foreign Small/Mid Blend',
       'Europe Stock', 'Industrials',
       'World Allocation', 'Foreign Small/Mid Value', 'Utilities',
       'Allocation--50% to 70% Equity', 'Diversified Pacific/Asia',
       'Equity Energy', 'Pacific/Asia ex-Japan Stk',
       'Equity Precious Metals']
minimum_aum = 100*10e5 # 100 million
expensive_ratio_filter=1/100 # 1%
etf_info = pd.read_csv(r'C:\Users\Walter\Desktop\portfolio_construction/etf_sector_info.csv', index_col=0)

def asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter):
    # etf_info
    expensive_ratio = etf_info['Annual Report Expense Ratio (net)'].str.replace('%', '').astype(float) / 100
    etf_info = etf_info[etf_info['totalAssets']>=minimum_aum]
    etf_info['Annual Report Expense Ratio (net)'] = expensive_ratio
    etf_info = etf_info[etf_info['Annual Report Expense Ratio (net)'] <= expensive_ratio_filter]
    etf_info = etf_info[etf_info['category'].isin(etf_types_filter)]
    final_assets=list(set(etf_info['symbol'].values))
    return final_assets


def portfolio_analysis(asset_weights, asset_returns, backtest_delay=1):
    # Ex-ante risk

    # Returns/performance
    common_assets = asset_weights.columns.intersection(asset_returns.columns)
    common_dates = asset_weights.index.intersection(asset_returns.index)
    asset_weights = asset_weights[common_assets].loc[common_dates]
    asset_returns = asset_returns[common_assets].loc[common_dates]


    portfolio_performance = ((asset_weights.shift(1+backtest_delay))*asset_returns).sum(axis=1)
    cumulative_returns = 1+portfolio_performance.cumsum()
    annualized_returns= portfolio_performance.mean()*252


    # Max-drawdown
    # Sharpe Ratio - 1year
    rolling_risk = portfolio_performance.rolling(window=252).std()*252**0.5
    sharpe_ratio = annualized_returns/rolling_risk

    # Holdings
    # Time (may not useful)
    return


def minimum_risk_portfolio(pc):
    """
    Suitable for investors with low and/or minimum risk tolerance.
    :param asset_name:
    :return:
    """
    print ('Working on Min Risk Portfolio')
    # Backtest
    # We use Rolling 252*3 number of days to estimate variance covariance matrix
    number_of_days_to_estimate = 252*3
    price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv'
    prices = pd.read_csv(price_url, index_col=0)
    final_assets = asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter)
    prices = prices[final_assets]
    prices.index = [x.split(' ')[0] for x in prices.index]
    portfolio_weights_all_periods = pd.DataFrame()
    simple_signal = prices.pct_change().rolling(window=252).mean()*252

    # Working on Covariance
    for i in range(number_of_days_to_estimate, len(prices)):
        prices_period = prices.iloc[i-number_of_days_to_estimate:i].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        simple_signal_period = simple_signal.iloc[i].dropna()
        attribute = simple_signal_period
        _, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix)
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

    portfolio_weights_all_periods.to_csv(rf'{save_files_folder}/minimum_risk_portfolio.csv')
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
    final_assets = asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter)
    prices = prices[final_assets]
    prices.index = [x.split(' ')[0] for x in prices.index]
    portfolio_weights_all_periods = pd.DataFrame()
    simple_signal = prices.pct_change().rolling(window=252).mean()*252

    # Working on Covariance
    for i in range(number_of_days_to_estimate, len(prices)):
        prices_period = prices.iloc[i-number_of_days_to_estimate:i].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        simple_signal_period = simple_signal.iloc[i].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = (
            pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix))

        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

        params = {'01_mean_variance': {
            'mean_variance': {'expected_returns_dict': expected_returns_dict_dataframe,
                              'common_assets': (list(common_assets)),
                              'lambda_value': 0.5}},
            '02_budget_constraint': {'budget_constraint': 1},
            '03_long_only_constraint': {'long_only_constraint': True},
            '06_type_of_optimization': {'type_of_optimization': 'maximize'}}

        pc.run_optimization(params)
        assets = common_assets
        weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
        weights.columns = [prices_period.index[-1]]
        # print (f'Minimum Risk Portfolio Holdings: {weights}')
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')

    portfolio_weights_all_periods.to_csv(rf'{save_files_folder}/mean_variance_portfolio.csv')
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
    final_assets = asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter)
    prices = prices[final_assets]
    prices.index = [x.split(' ')[0] for x in prices.index]
    portfolio_weights_all_periods = pd.DataFrame()
    simple_signal = prices.pct_change().rolling(window=252).mean()*252

    # Working on Covariance
    # 2020-03-06
    for i in range(number_of_days_to_estimate, len(prices)):
        # i=792
        # i=799
        prices_period = prices.iloc[i-number_of_days_to_estimate:i].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        simple_signal_period = simple_signal.iloc[i].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

        # To find the limit risk, we will require to iterate through the mean-variance framework using lambda value
        # given different lambda, we will have a different variance figure
        tolerance_level = 10**-6
        max_iteration = 100 # maximum iteration for lambda_value
        iteration=0
        # estimate lambda_value
        equal_weight_portfolio_returns = simple_signal_period.mean()
        equal_weights = pd.DataFrame([1/len(simple_signal)]*len(simple_signal_period))
        equal_weights.index = simple_signal_period.index
        equal_weight_portfolio_risk = np.dot(np.dot(equal_weights[0].T, var_cov_matrix), equal_weights[0])**0.5

        lambda_value=abs(equal_weight_portfolio_returns)/equal_weight_portfolio_risk #initial guess
        initial_lambda_value = lambda_value.copy()
        max_lambda_step = initial_lambda_value

        optimization_variance_figure = []
        lambda_values_list = [lambda_value]
        risk_target=0.1

        while iteration<= max_iteration:
            params = {'01_mean_variance': {
                'mean_variance': {'expected_returns_dict': expected_returns_dict_dataframe,
                                  'common_assets': (list(common_assets)),
                                  'lambda_value': lambda_value}},
                '02_budget_constraint': {'budget_constraint': 1},
                '03_long_only_constraint': {'long_only_constraint': True},
                '06_type_of_optimization': {'type_of_optimization': 'maximize'}}
            pc.run_optimization(params)
            # calculating ex-ante risk based on the risk model
            assets = common_assets
            weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
            weights.columns = [prices_period.index[-1]]
            ex_ante_variance =  np.dot(np.dot(weights.T, var_cov_matrix), weights)[0][0]
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
                    # Check the numerical stability
                    # Given the problem, if lambda_value_step >0 and current_ex_ante_variance>previous_ex_ante_variance
                    # Then it is an issue. We use tolerance
                    if abs(current_ex_ante_variance-previous_ex_ante_variance) < tolerance_level:
                        # The Gap is too small and it leads to issues and it does not lead to any changes to the optimization
                        lambda_value_step = max_lambda_step
                    else:
                        gap = risk_target**2-current_ex_ante_variance
                        lambda_value_step = gap/gradient
                        # Issue with this approach is that lambda_value_step can be very large and that would lead to numerical issues
                        # Therefore we introduce a limit and it is equal to the initial value of the lambda
                        if abs(lambda_value_step) > abs(max_lambda_step) :
                            # If value is greater
                            lambda_value_step = np.sign(lambda_value_step)*max_lambda_step
                    lambda_value = lambda_value + lambda_value_step
                    lambda_values_list = lambda_values_list+[lambda_value]
                    iteration = iteration + 1
                    print(lambda_values_list)
                    print(optimization_variance_figure)
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')

    portfolio_weights_all_periods.to_csv(rf'{save_files_folder}/mean_variance_portfolio_with_limit_risk.csv')
    return portfolio_weights_all_periods



def mean_variance_portfolio_different_expectation_scales_with_limit_risk(pc):
    """
     Suitable for investors with low and/or minimum risk tolerance.
     :param asset_name:
     :return:
     """
    print('Working on Mean-Variance Portfolio')
    # Backtest
    # We use Rolling 252 days to estimate variance covariance matrix
    number_of_days_to_estimate = 252 * 3
    price_url = r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv'
    prices = pd.read_csv(price_url, index_col=0)
    final_assets = asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter)
    prices = prices[final_assets]
    prices.index = [x.split(' ')[0] for x in prices.index]
    portfolio_weights_all_periods = pd.DataFrame()
    simple_signal = prices.pct_change().rolling(window=252).mean() * 252
    simple_signal = (simple_signal-simple_signal.mean(axis=1))/simple_signal.std() # same items, different scales

    # Working on Covariance
    # 2020-03-06
    for i in range(number_of_days_to_estimate, len(prices)):
        # i=792
        # i=799
        prices_period = prices.iloc[i - number_of_days_to_estimate:i].ffill()
        print('Working on %s Date' % (prices_period.index[-1]))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        simple_signal_period = simple_signal.iloc[i].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=attribute,
                                                                                             var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

        # To find the limit risk, we will require to iterate through the mean-variance framework using lambda value
        # given different lambda, we will have a different variance figure
        tolerance_level = 10 ** -6
        max_iteration = 100  # maximum iteration for lambda_value
        iteration = 0
        # estimate lambda_value
        equal_weight_portfolio_returns = simple_signal_period.mean()
        equal_weights = pd.DataFrame([1 / len(simple_signal)] * len(simple_signal_period))
        equal_weights.index = simple_signal_period.index
        equal_weight_portfolio_risk = np.dot(np.dot(equal_weights[0].T, var_cov_matrix), equal_weights[0]) ** 0.5

        lambda_value = abs(equal_weight_portfolio_returns) / equal_weight_portfolio_risk  # initial guess
        initial_lambda_value = lambda_value.copy()
        max_lambda_step = initial_lambda_value

        optimization_variance_figure = []
        lambda_values_list = [lambda_value]
        risk_target = 0.1

        while iteration <= max_iteration:
            params = {'01_mean_variance': {
                'mean_variance': {'expected_returns_dict': expected_returns_dict_dataframe,
                                  'common_assets': (list(common_assets)),
                                  'lambda_value': lambda_value}},
                '02_budget_constraint': {'budget_constraint': 1},
                '03_long_only_constraint': {'long_only_constraint': True},
                '06_type_of_optimization': {'type_of_optimization': 'maximize'}}
            pc.run_optimization(params)
            # calculating ex-ante risk based on the risk model
            assets = common_assets
            weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
            weights.columns = [prices_period.index[-1]]
            ex_ante_variance = np.dot(np.dot(weights.T, var_cov_matrix), weights)[0][0]
            optimization_variance_figure = optimization_variance_figure + [ex_ante_variance]
            # Based on if this number is higher or lower than the risk parameter, we will do adjustment
            # The idea of adjustment is similar to Gradient descent, where we calculate the change of risk vs. lambda
            # Since lambda is not a part of the risk calculation (or variance in mean-variance framework), we use the
            # first principle: change of risk/change of lambda.
            print(abs(risk_target ** 2 - ex_ante_variance))
            if abs(risk_target ** 2 - ex_ante_variance) <= tolerance_level:
                # No need to do anything. it is satisfied
                break
            else:
                if iteration == 0:
                    # if first iteration, we change lambda only tiny to find the first derivative
                    lambda_value_step = lambda_value / 10  # 10%, just repeat
                    # lambda_value_step = -(risk_target ** 2 - ex_ante_variance) * lambda_value / ex_ante_variance
                    current_ex_ante_variance = optimization_variance_figure[iteration]
                    if current_ex_ante_variance > risk_target ** 2:
                        lambda_value = lambda_value + lambda_value_step
                    else:
                        lambda_value = lambda_value - lambda_value_step
                    iteration = iteration + 1
                    print(iteration)
                else:
                    # so if iteration >=1, it means that we can use gradient descent
                    current_ex_ante_variance = optimization_variance_figure[iteration]
                    previous_ex_ante_variance = optimization_variance_figure[iteration - 1]
                    gradient = (current_ex_ante_variance - previous_ex_ante_variance) / lambda_value_step
                    # Check the numerical stability
                    # Given the problem, if lambda_value_step >0 and current_ex_ante_variance>previous_ex_ante_variance
                    # Then it is an issue. We use tolerance
                    if abs(current_ex_ante_variance - previous_ex_ante_variance) < tolerance_level:
                        # The Gap is too small and it leads to issues and it does not lead to any changes to the optimization
                        lambda_value_step = max_lambda_step
                    else:
                        gap = risk_target ** 2 - current_ex_ante_variance
                        lambda_value_step = gap / gradient
                        # Issue with this approach is that lambda_value_step can be very large and that would lead to numerical issues
                        # Therefore we introduce a limit and it is equal to the initial value of the lambda
                        if abs(lambda_value_step) > abs(max_lambda_step):
                            # If value is greater
                            lambda_value_step = np.sign(lambda_value_step) * max_lambda_step
                    lambda_value = lambda_value + lambda_value_step
                    lambda_values_list = lambda_values_list + [lambda_value]
                    iteration = iteration + 1
                    print(lambda_values_list)
                    print(optimization_variance_figure)
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')

    portfolio_weights_all_periods.to_csv(rf'{save_files_folder}/mean_variance_portfolio_with_limit_risk.csv')
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
    final_assets = asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter)
    prices = prices[final_assets]
    prices.index = [x.split(' ')[0] for x in prices.index]
    portfolio_weights_all_periods = pd.DataFrame()
    simple_signal = prices.pct_change().rolling(window=252).mean()*252

    # Working on Covariance
    for i in range(number_of_days_to_estimate, len(prices)):
        prices_period = prices.iloc[i-number_of_days_to_estimate:i].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        simple_signal_period = simple_signal.iloc[i].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

        params = {'01_expected_returns': {'expected_returns': {'expected_returns_dict': expected_returns_dict_dataframe,
                                                               'common_assets':(list(common_assets))
                                                               }
                                          },
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

    portfolio_weights_all_periods.to_csv(rf'{save_files_folder}/max_return_with_risk_target.csv')
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
    final_assets = asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter)
    prices = prices[final_assets]
    prices.index = [x.split(' ')[0] for x in prices.index]
    portfolio_weights_all_periods = pd.DataFrame()
    simple_signal = prices.pct_change().rolling(window=252).mean()*252

    # Working on Covariance
    for i in range(number_of_days_to_estimate, len(prices)):
        prices_period = prices.iloc[i-number_of_days_to_estimate:i].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        simple_signal_period = simple_signal.iloc[i].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

        params = {'01_expected_returns': {'expected_returns': {'expected_returns_dict': expected_returns_dict_dataframe,
                                                               'common_assets': (list(common_assets))
                                                               }
                                          },
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

    portfolio_weights_all_periods.to_csv(rf'{save_files_folder}/max_return_with_risk_target_and_additional_constraints.csv')
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
    final_assets = asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter)
    prices = prices[final_assets]
    prices.index = [x.split(' ')[0] for x in prices.index]
    portfolio_weights_all_periods = pd.DataFrame()
    simple_signal = prices.pct_change().rolling(window=252).mean()*252

    # Working on Covariance
    for i in range(number_of_days_to_estimate, len(prices)):
        prices_period = prices.iloc[i-number_of_days_to_estimate:i].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        simple_signal_period = simple_signal.iloc[i].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

        # input dicts
        params={'01_portfolio_variance': {'portfolio_variance':True},
                '02_maximum_weighted_average_alphas': {'maximum_limit_weighted_average':
                                                           {'attribute': list(expected_returns_dict_dataframe.values()),
                                                            'maximum_avg_weight': 1.0}
                                                       },
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
        portfolio_weights_all_periods.to_csv(
            rf'{save_files_folder}/max_return_with_max_sharpe.csv')

    return portfolio_weights_all_periods



# Inverse Volatility
def inverse_volatility(pc):
    print ('Working on Min Risk Portfolio')
    # Backtest
    # We use Rolling 252 days to estimate variance covariance matrix
    number_of_days_to_estimate = 252*3
    price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv'
    prices = pd.read_csv(price_url, index_col=0)
    final_assets = asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter)
    prices = prices[final_assets]
    prices.index = [x.split(' ')[0] for x in prices.index]
    portfolio_weights_all_periods = pd.DataFrame()
    simple_signal = prices.pct_change().rolling(window=252).mean()*252

    # Working on Covariance
    for i in range(number_of_days_to_estimate, len(prices)):
        prices_period = prices.iloc[i-number_of_days_to_estimate:i].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        simple_signal_period = simple_signal.iloc[i].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

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
    final_assets = asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter)
    prices = prices[final_assets]
    prices.index = [x.split(' ')[0] for x in prices.index]
    portfolio_weights_all_periods = pd.DataFrame()
    simple_signal = prices.pct_change().rolling(window=252).mean()*252

    # Working on Covariance
    for i in range(number_of_days_to_estimate, len(prices)):
        prices_period = prices.iloc[i-number_of_days_to_estimate:i].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        simple_signal_period = simple_signal.iloc[i].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

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

