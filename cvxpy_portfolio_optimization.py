import numbers

import pandas as pd
import numpy as np
import cvxpy
# Example - Return maximization with Variance Constraint
import cvxpy as cp
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import yfinance as yf
import random
# import warnings

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logger= logging.getLogger(__name__)
logging.basicConfig(level=logger.INFO)


def expected_returns(expected_returns_dataframe, portfolio_holdings ):
    mu = np.array(expected_returns_dataframe).reshape(1,-1)
    ret = mu@portfolio_holdings
    return ret

def maximum_limit_holding_constraint(portfolio_holdings, maximum_individual_holdings):
    constraints = portfolio_holdings <=maximum_individual_holdings
    return constraints

def minimum_limit_holding_constraint(portfolio_holdings, minimum_individual_holdings):
    import numbers
    constraints = portfolio_holdings>=minimum_individual_holdings
    return constraints

def limit_risk_constraint(portfolio_holdings, var_cov_matrix, risk_target):
    risk = cp.quad_form(portfolio_holdings, var_cov_matrix)
    constraints = risk <= risk_target ** 2  # variance constraint
    return constraints

def minimum_limit_single_asset_holding_constraint(portfolio_holdings, asset_name, minimum_individual_holdings, all_available_assets):
    # constraint written if users need to constraint individual asset weight
    # Users may decide to constraint Microsoft or Apple to be, say at least 20% the total portfolio and optimize the rest
    # We could use this constraint
    new_array = [0]*len(all_available_assets)
    new_array[all_available_assets.index(asset_name)]=1.0
    constraints = new_array @ portfolio_holdings >= minimum_individual_holdings
    return constraints

def maximum_limit_single_asset_holding_constraint(portfolio_holdings, asset_name, maximum_individual_holdings, all_available_assets):
    # constraint written if users need to constraint individual asset weight
    # Users may decide to constraint Microsoft or Apple to be, say at least 20% the total portfolio and optimize the rest
    # We could use this constraint
    new_array = [0]*len(all_available_assets)
    new_array[all_available_assets.index(asset_name)]=1.0
    constraints = new_array @ portfolio_holdings <= maximum_individual_holdings
    return constraints

def budget_constraint(portfolio_holdings, total_budget_values):
    constraints = cp.sum(portfolio_holdings) == total_budget_values
    return constraints

def long_only_constraint(portfolio_holdings):
    constraints = portfolio_holdings>=0
    return constraints

def maximum_limit_weighted_average(portfolio_holdings, attribute, maximum_avg_weight, asset_classification,  all_available_assets):

    new_array = [0] * len(all_available_assets)
    asset_sel = [name for name, age in asset_classification.items() if age == attribute]
    for asset_name in asset_sel:
        new_array[all_available_assets.index(asset_name)] = 1.0
    # weighted avg
    weighted_avg = new_array@portfolio_holdings <= maximum_avg_weight
    return weighted_avg

def minimum_limit_weighted_average(portfolio_holdings, attribute, minimum_avg_weight, asset_classification,  all_available_assets):
    # weighted avg

    new_array = [0] * len(all_available_assets)
    asset_sel = [name for name, age in asset_classification.items() if age == attribute]
    for asset_name in asset_sel:
        new_array[all_available_assets.index(asset_name)] = 1.0

    weighted_avg = new_array@portfolio_holdings >= minimum_avg_weight
    return weighted_avg

def maximum_number_of_assets(assets_weights):
    # This is done post-processing as the problem is not convex and it causes issues.
    # However it is not necessary to have a perfect solution to get values. We can post-process the value
    return

def portfolio_optimizations(objective, constraints):
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status not in ['infeasible', 'unbounded']:
        return prob.solve()
    else:
        # change objective function and change it to mean-variance
        # We would need to know which constraint is the risk constraint
        return

def compute_portfolio_weights(  params ):

    portfolio_holdings = cp.Variable((len(params['asset_names']), 1))

    objective = cp.Maximize(expected_returns(params['expected_returns'], portfolio_holdings))

    # to include a list of constraints from user inputs
    constraints = []

    if params["maximum_limit_holding_constraint"] is True :
        # constrain at the asset level
        maximum_individual_asset_holding_constraints = maximum_limit_holding_constraint(portfolio_holdings, params[
            'maximum_individual_asset_holding'])
        constraints.append(maximum_individual_asset_holding_constraints)

    if params["long_only_constraint"] is True:
        long_only_portfolio = long_only_constraint(portfolio_holdings)
        constraints.append(long_only_portfolio)

    if params["budget_constraint"] is True:
        fully_invested = budget_constraint(portfolio_holdings, params['total_budget'])
        constraints.append(fully_invested)

    if params["limit_risk_constraint"] is True:
        # define as low, medium and high risk for risk target
        target_risk_of_the_portfolio = limit_risk_constraint(portfolio_holdings, params["var_cov_matrix"],
                                                             params['risk_target'])
        constraints.append(target_risk_of_the_portfolio)

    if params["minimum_limit_single_asset_holding_constraint"] is True:
        # must have some assets with minimum holding
        for asset_name, asset_value in params["minimum_asset_constraints"].items():
            minimum_holding_of_one_asset = minimum_limit_single_asset_holding_constraint(portfolio_holdings, asset_name,
                                                                                         asset_value,
                                                                                         params['asset_names'])
            constraints.append(minimum_holding_of_one_asset)

    if params["maximum_limit_industry_constraint"] is True:
        # #constrain at the industry level
        for industry_name, industry_value in params['maximum_industry_constraints'].items():
            maximum_industry_investment_constraints = maximum_limit_weighted_average(portfolio_holdings,
                                                                                     industry_name,
                                                                                     industry_value,
                                                                                     params['asset_classification'],
                                                                                     params['asset_names'])
        constraints.append(maximum_industry_investment_constraints)

    # portfolio optimization, point in time
    portfolio_optimizations(objective, constraints)
    weights = pd.DataFrame(portfolio_holdings.value, index=assets, columns=['Weights'])
    return weights


if __name__ == '__main__':

    # # Date range
    start = '2016-01-01'
    end = '2019-12-30'

    # Tickers of assets
    assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
              'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
              'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
    assets.sort()
    #
    # # Downloading data
    data = yf.download(assets, start = start, end = end)
    data = data.loc[:,('Adj Close', slice(None))]
    data.columns = assets

    Y = data[assets].pct_change().dropna()

    asset_names = Y.columns.tolist()
    expected_returns_list = Y.mean().tolist()
    var_cov_matrix= Y.cov().to_numpy()

    industry = ['Industry_' + str(i) for i in range(6)]
    asset_classification = {x:industry[random.randint(0,5)] for x in asset_names}
    asset_classification = pd.Series(asset_classification.values(),index = list(asset_classification.keys() )).to_dict()

    # constraints name
    risk_target = 15 / (252**0.5 * 100)
    maximum_individual_asset_holding = 0.2

    total_budget = 1.0

    minimum_asset_constraints    = { "BA": 0.1  }
    maximum_industry_constraints = {'Industry_1': 0.2 }


    # ------------------------------------------------------------------------------------------------------------------------------------------------------------
    params = {            "maximum_limit_holding_constraint": True,
                          "long_only_constraint": True,
                          "budget_constraint": True,
                          "limit_risk_constraint": True,
                          "maximum_limit_industry_constraint": True,
                          "minimum_limit_single_asset_holding_constraint": True  ,

                          #---------------------------------------------------------------------

                          "asset_names": asset_names,
                          "maximum_individual_asset_holding": maximum_individual_asset_holding,
                          "expected_returns": expected_returns_list,
                          "total_budget": total_budget,
                          "var_cov_matrix": var_cov_matrix,
                          "risk_target": risk_target,
                          "minimum_asset_constraints": minimum_asset_constraints,
                          "maximum_industry_constraints": maximum_industry_constraints,
                          "asset_classification": asset_classification

                }



    weights = compute_portfolio_weights(  params)

    print(weights)

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