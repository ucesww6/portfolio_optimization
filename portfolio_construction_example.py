import numbers
import pandas as pd
import numpy as np
import cvxpy
import cvxpy as cp
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import yfinance as yf
import random
import warnings
import cvxpy as cvx
import pandas as pd
import portfolio_construction
import risk_contribution
import portfolio_analytics

def performance_and_risk_analysis(portfolio_holdings, covariance_matrix, attributes):
    # It will return some analytics
    rc =risk_contribution.RiskContribution(portfolio_holdings, covariance_matrix)
    info = {}
    info['asset_vol'] = rc.asset_volatility
    info['portfolio_vol'] = rc.portfolio_volatility
    info['asset_beta'] = rc.asset_beta

    # asset exposures
    joined_dataframe = portfolio_holdings.join(attributes, how='inner').fillna(0.0)
    portfolio_holdings = joined_dataframe['Weights'].to_frame()
    portfolio_holdings = pd.concat([portfolio_holdings]*len(attributes.columns), axis=1)
    portfolio_holdings.columns = attributes.columns
    attributes = joined_dataframe[attributes.columns]
    portfolio_exposures =np.multiply(portfolio_holdings, attributes).sum()
    info['portfolio_exposures']=portfolio_exposures

    return info


def minimum_risk_portfolio(pc):
    """
    Suitable for investors with low and/or minimum risk tolerance.
    :param asset_name:
    :return:
    """
    print ('Working on Min Risk Portfolio')
    var_cov_matrix = pc.use_statistical_risk_model(asset_name=[], read_price_from_file=True, price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
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
    print (f'Minimum Risk Portfolio Holdings: {weights}')
    return weights

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

def max_return_with_risk_target_shares_output(pc):
    """
    Suitable for typical investors with some risk-taking appetite and understand what expected returns and risks mean
    :return:
    """
    print ('Working on Max Returns with Risk Target')
    return_forecast = data[['symbol', 'threeYearAverageReturn']].set_index('symbol')
    var_cov_matrix = pc.use_statistical_risk_model(asset_name=[], read_price_from_file=True, price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
    return_forecast, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=return_forecast, var_cov_matrix=var_cov_matrix)
    pc.portfolio_holdings(common_assets)
    # return forecast
    expected_returns_dict = return_forecast['threeYearAverageReturn'].to_dict()
    risk_target = 0.3
    maximum_limit_holding_constraint = 0.05

    # input dicts
    params={'01_expected_returns': {'expected_returns':{'expected_returns_dict': expected_returns_dict, 'common_assets':common_assets}},
            '02_budget_constraint': {'budget_constraint': 1},
            '03_long_only_constraint': {'long_only_constraint': True},
            '04_limit_risk_constraint':{'limit_risk_constraint': risk_target},
            '05_maximum_limit_holding_constraint':{'maximum_limit_holding_constraint':maximum_limit_holding_constraint},
            '06_type_of_optimization':{'type_of_optimization': 'maximize'},
            }
    pc.run_optimization(params)
    weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=pc.asset_name, columns=['Weight'])
    share_price = data[['symbol','previousClose']].set_index('symbol')
    total_budget_in_dollars=50000
    shares,shares_in_dollar_values=pc.post_optimization_processing(share_price, total_budget_in_dollars)
    print (f'Minimum Risk Portfolio Holdings in Percentage: {weights}')
    print (f'Minimum Risk Portfolio Holdings in Shares: {shares}')
    print (f'Minimum Risk Portfolio Holdings in Dollars: {shares_in_dollar_values}')
    return weights


def max_return_with_risk_target_plus_beta_constraint_no_relax(pc):
    print (f'Max Alpha with Risk Limit Portfolio Holdings and beta Holdings')
    return_forecast = data[['symbol', 'threeYearAverageReturn']].set_index('symbol')
    var_cov_matrix = pc.use_statistical_risk_model(asset_name=[], read_price_from_file=True, price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
    return_forecast, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=return_forecast, var_cov_matrix=var_cov_matrix)
    pc.portfolio_holdings(common_assets)
    # return forecast
    expected_returns_dict = return_forecast['threeYearAverageReturn'].to_dict()
    risk_target = 0.25
    maximum_limit_holding_constraint = 0.05
    beta = data[['symbol', 'beta3Year']].set_index('symbol')
    beta, _, _ = pc.attribute_and_cov_alignment(attribute=beta, var_cov_matrix=var_cov_matrix)
    params={'01_expected_returns': {'expected_returns':{'expected_returns_dict': expected_returns_dict, 'common_assets':common_assets}},
            '02_budget_constraint': {'budget_constraint': 1},
            '03_long_only_constraint': {'long_only_constraint': True},
            '04_limit_risk_constraint':{'limit_risk_constraint': risk_target},
            '05_maximum_limit_holding_constraint':{'maximum_limit_holding_constraint':maximum_limit_holding_constraint},
            '06_type_of_optimization':{'type_of_optimization': 'minimize'},
            '07_maximum_beta_values':{'maximum_limit_weighted_average': {'attribute': beta['beta3Year'].to_list(), 'maximum_avg_weight': 1.0}}}

    pc.run_optimization(params)
    assets = common_assets
    weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
    # constraints
    print (weights)

    return weights


def max_return_with_risk_target_plus_beta_constraint_relax(pc):
    print (f'Max Alpha with Risk Limit Portfolio Holdings and beta Holdings, but Relax Constraints')
    return_forecast = data[['symbol', 'threeYearAverageReturn']].set_index('symbol')
    var_cov_matrix = pc.use_statistical_risk_model(asset_name=[], read_price_from_file=True, price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
    return_forecast, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=return_forecast, var_cov_matrix=var_cov_matrix)
    pc.portfolio_holdings(common_assets)
    # return forecast
    expected_returns_dict = return_forecast['threeYearAverageReturn'].to_dict()
    risk_target = 0.25
    maximum_limit_holding_constraint = 0.05
    beta = data[['symbol', 'beta3Year']].set_index('symbol')
    beta, _, _ = pc.attribute_and_cov_alignment(attribute=beta, var_cov_matrix=var_cov_matrix)
    min_agz = 0.1
    min_airr = 0.1

    params={'01_expected_returns': {'expected_returns':{'expected_returns_dict': expected_returns_dict, 'common_assets':common_assets}},
            '02_budget_constraint': {'budget_constraint': 1},
            '03_long_only_constraint': {'long_only_constraint': True},
            '04_limit_risk_constraint':{'limit_risk_constraint': risk_target},
            '05_maximum_limit_holding_constraint':{'maximum_limit_holding_constraint':maximum_limit_holding_constraint},
            '06_type_of_optimization':{'type_of_optimization': 'minimize'},
            '07_maximum_beta_values':{'maximum_limit_weighted_average': {'attribute': beta['beta3Year'].to_list(), 'maximum_avg_weight': 1.0}},
            '08_min_agz_holdings':{'minimum_limit_multiple_asset_aggregate_holding_constraint': {'asset_name_list': ['AGZ'],'common_assets':common_assets,'minimum_aggregate_holdings': min_agz}},
            '09_min_airr_holdings':{'minimum_limit_multiple_asset_aggregate_holding_constraint': {'asset_name_list': ['AIRR'],'common_assets':common_assets,'minimum_aggregate_holdings': min_airr}}}

    pc.run_optimization(params)
    assets = common_assets
    weights = pd.DataFrame(pc.portfolio_holdings_opt.value, index=assets, columns=['Weights'])
    # constraints
    print (weights)
    return weights


def max_return_with_risk_target_plus_beta_dynamic_constraint_relax(pc):
    print (f'Max Alpha with Risk Limit Portfolio Holdings and beta Holdings, but Relax Constraints')
    return_forecast = data[['symbol', 'threeYearAverageReturn']].set_index('symbol')
    var_cov_matrix = pc.use_statistical_risk_model(asset_name=[], read_price_from_file=True, price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
    return_forecast, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=return_forecast, var_cov_matrix=var_cov_matrix)
    pc.portfolio_holdings(common_assets)
    # return forecast
    expected_returns_dict = return_forecast['threeYearAverageReturn'].to_dict()
    risk_target = 0.10
    maximum_limit_holding_constraint = 0.10
    beta = data[['symbol', 'beta3Year']].set_index('symbol')
    beta, _, _ = pc.attribute_and_cov_alignment(attribute=beta, var_cov_matrix=var_cov_matrix)
    min_agz = 0.2
    min_airr = 0.35

    params={'01_expected_returns': {'expected_returns':{'expected_returns_dict': expected_returns_dict, 'common_assets':common_assets}},
            '02_budget_constraint': {'budget_constraint': 1},
            '03_long_only_constraint': {'long_only_constraint': True},
            '04_limit_risk_constraint':{'limit_risk_constraint': risk_target},
            '05_maximum_limit_holding_constraint':{'maximum_limit_holding_constraint':maximum_limit_holding_constraint},
            '07_maximum_beta_values':{'maximum_limit_weighted_average': {'attribute': beta['beta3Year'].to_list(), 'maximum_avg_weight': 1.0}},
            '08_min_agz_holdings':{'minimum_limit_multiple_asset_aggregate_holding_constraint': {'asset_name_list': ['AGZ'],'common_assets':common_assets,'minimum_aggregate_holdings': min_agz}},
            '09_min_airr_holdings':{'minimum_limit_multiple_asset_aggregate_holding_constraint': {'asset_name_list': ['AIRR'],'common_assets':common_assets,'minimum_aggregate_holdings': min_airr}},
            '06_type_of_optimization':{'type_of_optimization': 'maximize'},
            }

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
    # constraints
    print (weights)
    return weights


def portfolio_exposures(pc):
    # portfolio analytics
    weights = max_return_with_risk_target(pc)
    pa = portfolio_analytics.portfolio_analytics()
    classifications = data[['symbol', 'category']].set_index('symbol')
    classifications = classifications[~classifications.index.duplicated(keep='first')]
    var_cov_matrix = pc.use_statistical_risk_model(asset_name=[], read_price_from_file=True, price_url=r'C:\Users\Walter\Desktop\portfolio_construction/etf_prices.csv')
    weights, var_cov_matrix, common_assets = pc.attribute_and_cov_alignment(attribute=weights, var_cov_matrix=var_cov_matrix)
    summary = pa.final_output(asset_holdings=weights, classification=classifications,asset_covariance=var_cov_matrix)
    print(summary)
    return summary


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    data = pd.read_csv(r'C:\Users\Walter\Desktop\portfolio_construction/etf_sector_info.csv', index_col=0)
    portfolio_symbols = data['symbol'].to_list()
    category = data[['symbol', 'category']].set_index('symbol')
    portfolio_div = data['trailingAnnualDividendYield'].to_frame()
    pc = portfolio_construction.portfolio_construction(portfolio_symbols)
    # minimum_risk_portfolio(pc)
    max_return_with_risk_target(pc)
    # max_return_with_risk_target_plus_beta_constraint_no_relax(pc)
    # max_return_with_risk_target_plus_beta_constraint_relax(pc)
    # max_return_with_risk_target_plus_beta_dynamic_constraint_relax(pc)
    # portfolio_exposures(pc)
    # max_return_with_risk_target_shares_output(pc)
