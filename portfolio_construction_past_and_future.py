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
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r'C:\Users\Walter\Desktop\portfolio_construction/etf_sector_info.csv', index_col=0)
portfolio_symbols = data['symbol'].to_list()
category = data[['symbol', 'category']].set_index('symbol')
pc = portfolio_construction.portfolio_construction(portfolio_symbols)
save_files_folder = r'C:\Users\Walter\Desktop\Review/'
# asset_weights = pd.read_csv(r'C:\Users\Walter\Desktop\Review/max_return_with_risk_target.csv', index_col=0).T
price = pd.read_csv(r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv', index_col=0)
price.index =[x.split(' ')[0] for x in price.index]
asset_returns = price.pct_change()

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


def portfolio_analysis(asset_weights, price, asset_returns, backtest_delay=1, name_of_strategy=None, folder_to_save=None, file_name_to_save=None):
    min_asset_weight = 1e-6
    # Ex-ante risk
    # variance covariance matrix
    performance_to_save = pd.DataFrame()
    portfolio_ex_ante_risk = []
    if 'AIRR' in asset_weights.index:
        asset_weights = asset_weights.T # transpose()

    for each_date in asset_weights.index:
        print ('working on %s date'%each_date)
        current_date_pos = list(price.index).index(each_date)
        beginning_pos = current_date_pos+1-252*3
        prices_period = price.iloc[beginning_pos:current_date_pos+1]
        portfolio_period = asset_weights.loc[each_date].to_frame().fillna(0.0)
        asset_name = portfolio_period.index
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        ex_ante_risk_per_period = np.dot(np.dot(portfolio_period.T, var_cov_matrix), portfolio_period) ** 0.5
        portfolio_ex_ante_risk = portfolio_ex_ante_risk + [ex_ante_risk_per_period[0][0]]
    portfolio_ex_ante_risk = pd.DataFrame(portfolio_ex_ante_risk, index=asset_weights.index, columns =[name_of_strategy + ' Ex-Ante Risk'])

    # Returns/performance
    common_assets = asset_weights.columns.intersection(asset_returns.columns)
    common_dates = asset_weights.index.intersection(asset_returns.index)
    asset_weights = asset_weights[common_assets].loc[common_dates]
    asset_returns = asset_returns[common_assets].loc[common_dates]
    portfolio_performance = ((asset_weights.shift(1+backtest_delay))*asset_returns).sum(axis=1)
    cumulative_returns = 1+portfolio_performance.cumsum()
    annualized_returns= portfolio_performance.mean()*252
    annualized_risk = portfolio_performance.std()*252**0.5
    overall_sharpe_ratio = annualized_returns/annualized_risk

    performance_summary = pd.DataFrame([annualized_returns, annualized_risk, overall_sharpe_ratio])
    performance_summary.index = ['Annualized Returns', 'Annualized Risk', 'Sharpe Ratio', ]
    performance_summary.columns = [name_of_strategy + ' Performance']

    cumulative_returns = cumulative_returns.to_frame()
    cumulative_returns.columns = [name_of_strategy + ' Cumulative Returns']

    # Sharpe Ratio - 1year
    portfolio_performance = portfolio_performance.to_frame()
    portfolio_performance.columns = [name_of_strategy+ ' Rolling Returns']
    rolling_returns = portfolio_performance.rolling(window=252).mean()*252
    rolling_ex_post_risk = portfolio_performance.rolling(window=252).std()*252**0.5
    rolling_ex_post_risk.columns = [name_of_strategy + ' Rolling Risk']
    rolling_sharpe_ratio = np.divide(rolling_returns, rolling_ex_post_risk)
    rolling_sharpe_ratio.columns = [name_of_strategy + ' Rolling Sharpe Ratio']

    # Holdings
    holdings = pd.DataFrame()
    for each_date in asset_weights.index:
        temporary_holdings = asset_weights.loc[each_date]
        temporary_holdings = temporary_holdings[temporary_holdings>=min_asset_weight].to_frame()
        holdings = holdings.join(temporary_holdings, how='outer')
    holdings = holdings.T
    holdings = holdings.fillna(0.0)
    return {'ex_ante_risk': portfolio_ex_ante_risk,
            'annualized_returns': annualized_returns,
            'annualized_risk': annualized_risk,
            'overall_sharpe_ratio': overall_sharpe_ratio,
            'cumulative_returns':cumulative_returns,
            'rolling_ex_post_risk': rolling_ex_post_risk,
            'rolling_sharpe_ratio':rolling_sharpe_ratio,
            'holdings': holdings}


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
        date_to_extract = prices_period.index[-1]
        simple_signal_period = simple_signal.loc[date_to_extract].dropna()
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
        date_to_extract = prices_period.index[-1]
        simple_signal_period = simple_signal.loc[date_to_extract].dropna()
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


def original_mean_variance_portfolio_with_limit_risk(pc):
    """
    Suitable for investors with low and/or minimum risk tolerance.
    :param asset_name:
    :return:
    """
    print ('Working on Mean-Variance Portfolio With Limit Risk Constraint Solving by Secant Method ')
    # Backtest
    # We use Rolling 252 days to estimate variance covariance matrix
    number_of_days_to_estimate = 252*3
    price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv'
    prices = pd.read_csv(price_url, index_col=0)
    final_assets = asset_filter(etf_info, etf_types_filter, minimum_aum, expensive_ratio_filter)
    prices = prices[final_assets]
    prices.index = [x.split(' ')[0] for x in prices.index]
    simple_signal = prices.pct_change().rolling(window=252).mean()*252
    # Working on Covariance
    # 2020-03-25
    # Working on 2022-05-06 Date
    # The problem is solved by using Secant Method
    # The mean-variance portfolio has one additional constraint:  quadratic constraint where we require to have
    # a limit risk value less or equal to certain number
    # We solve this problem by using secant method where the ONLY additional unknown parameter to us is lambda
    # We iterate over lambda value until it converges
    # https://en.wikipedia.org/wiki/Secant_method
    # x_n = x_n-1 - f(x_n-1)/secant_estimation_of_first_derivative (estimated using two points)
    # The limit risk constraint effectively becomes
    # limit (variance) <= 0.1**2 or limit risk <= 0.1
    # We set G(x) = limit(risk) - 0.1 <=0
    # We assume that we always can fulfill the risk budgeting where we have G(x) = limit(risk) - 0.1 = 0
    # We can then use secant method (root finding) method to find the solution
    # We cannot directly use newton's method because lambda is not a part of the G(X) therefore we cannot derive
    # the first derivative directly
    # However, we could use secant method to estimate the first derivative and solve the equation (root-finding).
    # Secant method converges at a faster rate (super-linear) vs. bisection (linear).
    # The issue of the problem is that it may not converge occasionally
    # We will calculate converge ratio and converge rate
    portfolio_weights_all_periods = pd.DataFrame()
    iteration_per_period = []
    lambda_and_risk_per_period = {}
    final_lambda_value = []
    for i in range(number_of_days_to_estimate, len(prices)):
    # for i in range(number_of_days_to_estimate, 900):
        # 2020-03-26
        # i=792
        # i=799
        # i=900
        # i=812
        # i=813
        prices_period = prices.iloc[i-number_of_days_to_estimate:i].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        print ('Wokring on %i '%(i))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        date_to_extract = prices_period.index[-1]
        simple_signal_period = simple_signal.loc[date_to_extract].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

        # To find the limit risk, we will require to iterate through the mean-variance framework using lambda value
        # given different lambda, we will have a different variance figure
        tolerance_level = 10**-6
        max_iteration = 100 # maximum iteration for lambda_value
        iteration=0
        risk_target=0.1
        # estimate lambda_value

        equal_weight_portfolio_returns = simple_signal_period.mean()
        equal_weights = pd.DataFrame([1/len(simple_signal)]*len(simple_signal_period))
        equal_weights.index = simple_signal_period.index
        equal_weight_portfolio_risk = np.dot(np.dot(equal_weights[0].T, var_cov_matrix), equal_weights[0])**0.5
        # initial guess of lambda_value
        # Initial guess of lambda_value is important because it will set-up where the next step it will go
        # To determine a reasonable lambda value we use the following assumption:
        # No-off-diagonal (correlation between asset is 0)
        # Assuming that assets all have the same risk
        # Assuming that we can achieve the risk-target, 0.1
        # We then back-out the return (or expected returns of the portfolio)
        # Ratio between the 2 gives us the initial estimate
        # effectively we assume that the objective function is 0
        lambda_value=abs(equal_weight_portfolio_returns)/equal_weight_portfolio_risk #initial guess
        initial_lambda_value = lambda_value
        max_lambda_step = initial_lambda_value
        optimization_variance_figure = []
        lambda_values_list = [initial_lambda_value]

        while iteration <= max_iteration-1:
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
            # The idea of adjustment is similar to secant_estimation_of_first_derivative descent, where we calculate the change of risk vs. lambda
            # Since lambda is not a part of the risk calculation (or variance in mean-variance framework), we use the
            # first principle: change of risk/change of lambda.
            print('Gap is %s' % (ex_ante_variance - risk_target ** 2))

            if iteration == max_iteration-1:
                # Update the last iteration, risk figure
                break

            if abs(ex_ante_variance - risk_target ** 2) <= tolerance_level:
                # No need to do anything. it is satisfied
                break
            else:
                if iteration ==0:
                    # if first iteration, we change lambda only tiny to find the first derivative
                    change_in_lambda_value = lambda_value/10
                    # change_in_lambda_value = -(risk_target ** 2 - ex_ante_variance) * lambda_value / ex_ante_variance
                    current_ex_ante_variance = optimization_variance_figure[iteration]
                    if current_ex_ante_variance - risk_target**2 > 0: # G(x) > 0
                        lambda_value = lambda_value + change_in_lambda_value
                    else:
                        lambda_value = lambda_value - change_in_lambda_value
                    iteration = iteration + 1
                else:
                    # so if iteration >=1, it means that we can use secant_estimation_of_first_derivative descent
                    current_ex_ante_variance = optimization_variance_figure[iteration]
                    previous_ex_ante_variance = optimization_variance_figure[iteration-1]
                    secant_estimation_of_first_derivative = (current_ex_ante_variance-risk_target**2 - previous_ex_ante_variance + risk_target**2)/change_in_lambda_value # negative
                    # Check the numerical stability
                    # Given the problem, if change_in_lambda_value >0 and current_ex_ante_variance>previous_ex_ante_variance
                    # Then it is an issue. We use tolerance
                    if abs(current_ex_ante_variance-previous_ex_ante_variance) < tolerance_level:
                        # The Gap is too small and it leads to issues and it does not lead to any changes to the optimization
                        change_in_lambda_value = max_lambda_step*iteration
                    else:
                        g_x = current_ex_ante_variance - risk_target**2
                        change_in_lambda_value = (-1*g_x)/secant_estimation_of_first_derivative
                        # Another issue with this approach is that change_in_lambda_value can be very large and that
                        # would lead to explosion of first derivative (ill-defined). Especially we may run into
                        # problem of a negative lambda value. This can happen if g_x is less than zero and
                        # secant_estimation is very tiny, leading to a large positive lambda value step
                        # Then the lambda_value can then be negative which is an issue. Therefore we need to introduce
                        # a level of settings to avoid this problem
                        if lambda_value + change_in_lambda_value < 0:
                            change_in_lambda_value = np.sign(change_in_lambda_value)*lambda_value*0.5 # by half so that we reset the problem
                    lambda_value = lambda_value + change_in_lambda_value
                    iteration = iteration + 1
                lambda_values_list = lambda_values_list + [lambda_value]
        lambda_values_list = pd.DataFrame(lambda_values_list, columns = ['lambda'])
        optimization_risk_figure = pd.DataFrame(optimization_variance_figure, columns = ['Risk'])**0.5
        lambda_and_risk = lambda_values_list.join(optimization_risk_figure)
        lambda_and_risk_per_period[simple_signal_period.name] = lambda_and_risk
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')
        iteration_per_period = iteration_per_period + [iteration+1]
        final_lambda_value = final_lambda_value + [lambda_value]
        print('Converge After %s Steps ' % (iteration + 1))

    portfolio_weights_all_periods.to_csv(rf'{save_files_folder}/mean_variance_portfolio_with_limit_risk.csv')
    iteration_per_period = pd.DataFrame(iteration_per_period, index=portfolio_weights_all_periods.columns, columns=['Iteration Per Period'])
    final_lambda_value = pd.DataFrame(final_lambda_value, columns = ['Final lambda Value'], index=portfolio_weights_all_periods.columns)
    iteration_per_period.to_csv(rf'{save_files_folder}/iteration_per_period.csv')
    final_lambda_value.to_csv(rf'{save_files_folder}/final_lambda_value.csv')
    import pickle
    with open('secant_estimation_of_first_derivative_analysis.pkl', 'wb') as f:
        pickle.dump(lambda_and_risk_per_period, f)

    return portfolio_weights_all_periods, iteration_per_period


def original_mean_variance_portfolio_with_limit_risk_different_scale(pc):
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
    simple_signal = prices.pct_change().rolling(window=252).mean()*252*10
    # simple_signal = simple_signal.sub(simple_signal.mean(axis=1), axis=0)
    # simple_signal = simple_signal.div(simple_signal.std(axis=1), axis=0)

    # Working on Covariance
    # 2020-03-25
    # Working on 2022-05-06 Date
    iteration_per_period = []
    lambda_and_risk_per_period = {}
    final_lambda_value = []
    for i in range(number_of_days_to_estimate, len(prices)):
        # i=792
        # i=799
        # i=900
        # i=812
        prices_period = prices.iloc[i-number_of_days_to_estimate:i].ffill()
        print ('Working on %s Date'%(prices_period.index[-1]))
        asset_name = list(prices_period.columns)
        var_cov_matrix = pc.use_statistical_risk_model(asset_name=asset_name,
                                                       read_price_from_file=False,
                                                       use_price_input=True,
                                                       price_input=prices_period,
                                                       price_url=None)
        date_to_extract = prices_period.index[-1]
        simple_signal_period = simple_signal.loc[date_to_extract].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix)
        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

        # To find the limit risk, we will require to iterate through the mean-variance framework using lambda value
        # given different lambda, we will have a different variance figure
        tolerance_level = 10**-6
        max_iteration = 100 # maximum iteration for lambda_value
        iteration=0
        risk_target=0.1
        # estimate lambda_value

        equal_weight_portfolio_returns = simple_signal_period.mean()
        equal_weights = pd.DataFrame([1/len(simple_signal)]*len(simple_signal_period))
        equal_weights.index = simple_signal_period.index
        equal_weight_portfolio_risk = np.dot(np.dot(equal_weights[0].T, var_cov_matrix), equal_weights[0])**0.5

        lambda_value=abs(equal_weight_portfolio_returns)/equal_weight_portfolio_risk #initial guess
        initial_lambda_value = lambda_value
        max_lambda_step = initial_lambda_value

        optimization_variance_figure = []
        lambda_values_list = [initial_lambda_value]

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
            # The idea of adjustment is similar to secant_estimation_of_first_derivative descent, where we calculate the change of risk vs. lambda
            # Since lambda is not a part of the risk calculation (or variance in mean-variance framework), we use the
            # first principle: change of risk/change of lambda.
            if abs(risk_target**2-ex_ante_variance) <= tolerance_level:
                # No need to do anything. it is satisfied
                break
            else:
                if iteration ==0:
                    # if first iteration, we change lambda only tiny to find the first derivative
                    change_in_lambda_value = lambda_value/10
                    # change_in_lambda_value = -(risk_target ** 2 - ex_ante_variance) * lambda_value / ex_ante_variance
                    current_ex_ante_variance = optimization_variance_figure[iteration]
                    if current_ex_ante_variance > risk_target**2:
                        lambda_value = lambda_value + change_in_lambda_value
                    else:
                        lambda_value = lambda_value - change_in_lambda_value
                    iteration = iteration + 1
                else:
                    # so if iteration >=1, it means that we can use secant_estimation_of_first_derivative descent
                    current_ex_ante_variance=optimization_variance_figure[iteration]
                    previous_ex_ante_variance = optimization_variance_figure[iteration-1]
                    secant_estimation_of_first_derivative = (current_ex_ante_variance-previous_ex_ante_variance)/change_in_lambda_value
                    # Check the numerical stability
                    # Given the problem, if change_in_lambda_value >0 and current_ex_ante_variance>previous_ex_ante_variance
                    # Then it is an issue. We use tolerance
                    if abs(current_ex_ante_variance-previous_ex_ante_variance) < tolerance_level:
                        # The Gap is too small and it leads to issues and it does not lead to any changes to the optimization
                        change_in_lambda_value = max_lambda_step*iteration
                    else:
                        gap = risk_target**2-current_ex_ante_variance
                        change_in_lambda_value = gap/secant_estimation_of_first_derivative
                        # Issue with this approach is that change_in_lambda_value can be very large and that would lead to numerical issues
                        # Therefore we introduce a limit and it is equal to the initial value of the lambda
                        if abs(change_in_lambda_value) > abs(max_lambda_step*iteration) :
                            # If value is greater
                            change_in_lambda_value = np.sign(change_in_lambda_value)*max_lambda_step
                    lambda_value = lambda_value + change_in_lambda_value
                    iteration = iteration + 1
                lambda_values_list = lambda_values_list + [lambda_value]
        lambda_values_list = pd.DataFrame(lambda_values_list, columns = ['lambda'])
        optimization_risk_figure = pd.DataFrame(optimization_variance_figure, columns = ['Risk'])**0.5
        lambda_and_risk = lambda_values_list.join(optimization_risk_figure)
        lambda_and_risk_per_period[simple_signal_period.name] = lambda_and_risk
        portfolio_weights_all_periods = portfolio_weights_all_periods.join(weights, how='outer')
        iteration_per_period = iteration_per_period + [iteration+1]
        final_lambda_value = final_lambda_value + [lambda_value]

    portfolio_weights_all_periods.to_csv(rf'{save_files_folder}/mean_variance_portfolio_with_limit_risk_scaled_returns.csv')
    iteration_per_period = pd.DataFrame(iteration_per_period, index=portfolio_weights_all_periods.columns, columns=['Iteration Per Period'])
    final_lambda_value = pd.DataFrame(final_lambda_value, columns = ['Final lambda Value'], index=portfolio_weights_all_periods.columns)
    iteration_per_period.to_csv(rf'{save_files_folder}/iteration_per_period_scaled_returns.csv')
    final_lambda_value.to_csv(rf'{save_files_folder}/final_lambda_value_scaled_returns.csv')
    import pickle
    with open('secant_estimation_of_first_derivative_analysis_scaled_returns.pkl', 'wb') as f:
        pickle.dump(lambda_and_risk_per_period, f)

    return portfolio_weights_all_periods, iteration_per_period


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
    risk_target=0.1

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
        date_to_extract = prices_period.index[-1]
        simple_signal_period = simple_signal.loc[date_to_extract].dropna()
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
        date_to_extract = prices_period.index[-1]
        simple_signal_period = simple_signal.loc[date_to_extract].dropna()
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
        date_to_extract = prices_period.index[-1]
        simple_signal_period = simple_signal.loc[date_to_extract].dropna()
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
        date_to_extract = prices_period.index[-1]
        simple_signal_period = simple_signal.loc[date_to_extract].dropna()
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
        date_to_extract = prices_period.index[-1]
        simple_signal_period = simple_signal.loc[date_to_extract].dropna()
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



def mean_variance_portfolio_with_output_being_equal_weight(pc):
    """
    A special case where you want to use mean-variance optimization to obtain an equal weight portfolio
    An important point is that if we have calculated the variance covariance matrix and you want to maintain as much
    information as you can.

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
    simple_signal = prices.copy()
    for each_asset in simple_signal.columns:
        simple_signal[each_asset] = 1/len(simple_signal.columns)

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
        date_to_extract = prices_period.index[-1]
        simple_signal_period = simple_signal.loc[date_to_extract].dropna()
        attribute = simple_signal_period
        simple_signal_period, var_cov_matrix, common_assets = (
            pc.attribute_and_cov_alignment(attribute=attribute, var_cov_matrix=var_cov_matrix))


        pc.portfolio_holdings(common_assets)
        expected_returns_dict_dataframe = simple_signal_period.to_dict()

        # Constrained Regression
        # It is an optimization problem where the new matrix is as closely as possible to the old one
        # It is an optimization problem with some constraints. It is a constrained optimization problem
        # and we use portfolio_variance or quad_form with constraints to solve for values
        column_vector = var_cov_matrix.values.flatten()  # row-wise flattening
        column_vector = column_vector.reshape(-1, 1)
        # optional: make it an explicit column vector
        # large_x is the independent variables
        n = len(var_cov_matrix)
        large_x_a = np.eye(n)
        large_x_a = large_x_a.flatten()  # row-wise flattening
        large_x_a = large_x_a.reshape(-1, 1)
        large_x_a = pd.DataFrame(large_x_a)
        large_x_a.columns = ['xa']
        large_x_b = 1-large_x_a
        large_x_b.columns = ['xb']
        large_x = large_x_a.join(large_x_b)
        y = column_vector

        y = np.array(y).flatten()
        X = np.array(large_x)
        beta = cp.Variable(X.shape[1])  # This is safe now
        objective = cp.Minimize(cp.sum_squares(X @ beta - y))

        # Constraints: -b2 <= b1 <= b2, b1 > 0
        constraints = [
            beta[0] - 1*beta[1] <= 0,  # b1 - b2 <= 0
            -beta[0] - 1*beta[1] <= 0,  # -b1 - b2 <= 0
            beta[0] >= 0  # b1 >= 0
        ]

        # Problem setup and solve
        problem = cp.Problem(objective, constraints)
        problem.solve()


def admm_max_diverficiation_ratio():
    import numpy as np
    import cvxpy as cp
    import pandas as pd

    # Problem data
    np.random.seed(42)
    n = 10
    mu = np.random.uniform(0.05, 0.15, size=n)
    Sigma = np.random.randn(n, n)
    Sigma = Sigma.T @ Sigma + 1e-3 * np.eye(n)  # Make positive definite

    # ADMM parameters
    rho = 1.0
    max_iter = 200
    tol = 1e-5

    # Initialize variables
    w = np.ones(n) / n
    h = np.copy(w)
    u = np.zeros(n)

    # History tracking
    history = []

    for k in range(max_iter):
        # w-update
        w_var = cp.Variable(n)
        obj = cp.Minimize(-mu @ w_var + (rho / 2) * cp.sum_squares(w_var - h + u))
        constraints = [cp.sum(w_var) == 1, w_var >= 0]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        w_new = w_var.value

        # h-update (closed form)
        A = Sigma + (rho / 2) * np.eye(n)
        b = (rho / 2) * (w_new + u)
        h_new = np.linalg.solve(A, b)

        # u-update
        u += w_new - h_new

        # Stats
        port_return = mu @ w_new
        port_risk = np.sqrt(w_new.T @ Sigma @ w_new)
        sharpe = port_return / port_risk
        residual = np.linalg.norm(w_new - h_new)

        history.append((k, sharpe, port_return, port_risk, residual))

        if residual < tol:
            break

    # Convert history to DataFrame
    df = pd.DataFrame(history, columns=["iteration", "sharpe", "return", "risk", "primal_residual"])
    print(df.tail())



# Example 1 - Equal Weight
# portfolio_weights_all_periods = equal_weight(pc)

# Example 2 - Minimum Risk
# single period
# minimum_risk_portfolio(pc)
# backtest

# Example X - Mean Variance
# mean_variance_portfolio(pc)

# Example X - How to use Mean-Variance to Do a Limit Risk Constraint
# original_mean_variance_portfolio_with_limit_risk(pc)

# Example X - How to use Mean-Variance to Do a Limit Risk Constraint, but expectation is in different scale
# portfolio_weights_all_periods = original_mean_variance_portfolio_with_limit_risk_different_scale(pc)

# max_returns_with_risk_target
# max_return_with_risk_target(pc)

# Equal Weight
# mean_variance_portfolio_with_output_being_equal_weight(pc)
# Nothing to examine

# Max-Sharpe
# portfolio_weights_all_periods= max_return_with_max_sharpe(pc)

# Risk-Parity

# Inverse Volatility
# portfolio_weights_all_periods = inverse_volatility(pc)
mean_variance_portfolio_with_output_being_inverse_volatility(pc)
# Example X - How to use Mean-Variance to Do a Limit Risk Constraint
# portfolio_weights_all_periods= max_return_with_risk_target(pc)

# Performance Analysis
# secant_estimation_of_first_derivative analysis
# import pickle
# obj = pd.read_pickle(r'C:\Users\Walter\Desktop\Review/secant_estimation_of_first_derivative_analysis.pkl')
#
# file_folder = r'C:\Users\Walter\Desktop\Review/'
# ex_ante_risk_list = pd.DataFrame()
# cumulative_returns_list = pd.DataFrame()
# rolling_ex_post_risk = pd.DataFrame()
# rolling_sharpe_ratio_list = pd.DataFrame()
# holdings_list = {}
# for file_names in ['max_return_with_risk_target', 'mean_variance_portfolio', 'mean_variance_portfolio_with_limit_risk', 'minimum_risk_portfolio']:
#     print ('Working on %s' %file_names)
#     file_to_read = pd.read_csv(rf'{file_folder}/{file_names}.csv', index_col=0)
#     asset_weights = file_to_read
#     performance_each_strategy = portfolio_analysis(asset_weights, price, asset_returns, backtest_delay=1, name_of_strategy=file_names)
#     ex_ante_risk_list = ex_ante_risk_list.join(performance_each_strategy['ex_ante_risk'], how='outer')
#     cumulative_returns_list = cumulative_returns_list.join(performance_each_strategy['cumulative_returns'], how='outer')
#     rolling_ex_post_risk = rolling_ex_post_risk.join(performance_each_strategy['rolling_ex_post_risk'], how='outer')
#     rolling_sharpe_ratio_list = rolling_sharpe_ratio_list.join(performance_each_strategy['rolling_sharpe_ratio'], how='outer')
#
# ex_ante_risk_list.to_excel(rf'{file_folder}/ex_ante_risk_list.xlsx')
# cumulative_returns_list.to_excel(rf'{file_folder}/cumulative_returns_list.xlsx')
# rolling_ex_post_risk.to_excel(rf'{file_folder}/rolling_ex_post_risk.xlsx')
# rolling_sharpe_ratio_list.to_excel(rf'{file_folder}/rolling_sharpe_ratio_list.xlsx')