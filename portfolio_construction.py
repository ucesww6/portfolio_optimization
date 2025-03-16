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
import math


class portfolio_construction:
    def __init__(self, asset_name):
        print ('Portfolio Construction Class Initiated')
        self.asset_name = asset_name
        self.mu = None
        self.var_cov_matrix = None
        self.portfolio_holdings_opt = None
        self.relaxation_parameter = 0.5
        self.relaxation_time=1
        self.max_relaxation_time=10
        self.relax_constraint_version="beta"

    def prepare_input_for_portfolio_optimization(self):
        # initiate expected return and var-cov
        return

    def load_basic_info(self):
        # ticker map, time series, industry classification and others?
        return

    def portfolio_holdings(self, common_assets):
        self.portfolio_holdings_opt = cp.Variable((len(common_assets), 1))
        return self.portfolio_holdings_opt

    def expected_returns(self,function_name, expected_returns_dict, common_assets):
        # find across with common and set the non-intersected values to be 0
        no_expected_returns = set(common_assets)-set(expected_returns_dict.keys())
        no_expected_returns = {x:0.0 for x in no_expected_returns}
        expected_returns_dict.update(no_expected_returns)
        expected_returns_dict = list(expected_returns_dict.values())
        mu = expected_returns_dict
        ret = mu @ self.portfolio_holdings_opt
        function_type = 'expected_returns'
        return ret,function_type,function_name

    def portfolio_variance(self,function_name, *argv):
        self.portfolio_variance_opt = cp.quad_form(self.portfolio_holdings_opt, self.var_cov_matrix)
        function_type = 'portfolio_variance'
        return self.portfolio_variance_opt, function_type, function_name

    def mean_variance(self, function_name, expected_returns_dict, common_asset):
        mean_variance = self.expected_returns(self.portfolio_holdings_opt, expected_returns_dict, common_asset,function_name)[0]-\
                        0.5*self.portfolio_variance_opt(self.portfolio_holdings_opt, self.var_cov_matrix, function_name)[0]
        function_type = 'mean_variance'
        return mean_variance, function_type, function_name

    def expense_ratio(self, function_name, expense_ratio_dict, common_asset):
        exp_ratio = self.expected_returns(expense_ratio_dict,common_asset)
        function_type = 'expense_ratio'
        return exp_ratio, function_type, function_name

    def maximum_limit_holding_constraint(self, function_name, maximum_individual_holdings):
        constraints = self.portfolio_holdings_opt <= maximum_individual_holdings
        function_type = 'maximum_limit_holding_constraint'
        return constraints, function_type, function_name

    def minimum_limit_holding_constraint(self, function_name, minimum_individual_holdings):
        constraints = self.portfolio_holdings_opt >= minimum_individual_holdings
        function_type = 'minimum_limit_holding_constraint'
        return constraints, function_type, function_name

    def limit_risk_constraint(self, function_name, risk_target):
        risk = cp.quad_form(self.portfolio_holdings_opt, self.var_cov_matrix)
        constraints = risk <= float(risk_target) ** 2 # variance constraint
        function_type = 'limit_risk_constraint'
        return constraints, function_type,function_name

    def minimum_limit_multiple_asset_aggregate_holding_constraint(self,function_name,asset_name_list,common_assets,
                                                                  minimum_aggregate_holdings):
        # We could use this constraint
        new_array = [0]*len(common_assets)
        for asset_name in asset_name_list:
            new_array[common_assets.get_loc(asset_name)]=1.0
        constraints = new_array @ self.portfolio_holdings_opt >= minimum_aggregate_holdings
        function_type = 'minimum_limit_multiple_asset_aggregate_holding_constraint'
        return constraints, function_type,function_name


    def maximum_limit_multiple_asset_aggregate_holding_constraint(self,function_name,asset_name_list,common_assets,
                                                                  minimum_aggregate_holdings):
        # We could use this constraint
        new_array = [0]*len(common_assets)
        for asset_name in asset_name_list:
            new_array[common_assets.get_loc(asset_name)]=1.0
        constraints = new_array @ self.portfolio_holdings_opt <= minimum_aggregate_holdings
        function_type = 'minimum_limit_multiple_asset_aggregate_holding_constraint'
        return constraints, function_type,function_name


    def budget_constraint(self, function_name, total_budget_values):
        constraints = cp.sum(self.portfolio_holdings_opt) == total_budget_values
        function_type = 'budget_constraint'
        return constraints, function_type,function_name

    def long_only_constraint(self, function_name, *args):
        constraints = self.portfolio_holdings_opt>=0
        function_type = 'long_only_constraint'
        return constraints, function_type,function_name

    def maximum_limit_weighted_average(self, function_name, attribute,maximum_avg_weight):
        weighted_avg = attribute @ self.portfolio_holdings_opt <= maximum_avg_weight
        function_type = 'maximum_limit_weighted_average'
        return weighted_avg, function_type,function_name

    def minimum_limit_weighted_average(self, function_name,attribute,minimum_avg_weight):
        weighted_avg = attribute @ self.portfolio_holdings_opt >= minimum_avg_weight
        function_type = 'minimum_limit_weighted_average'
        return weighted_avg, function_type,function_name

    def maximum_limit_weighted_classification(self, function_name,attribute,maximum_avg_weight,
                                              asset_classification,common_assets):
        new_array = [0] * len(common_assets)
        asset_sel = [name for name, age in asset_classification.items() if age == attribute]
        for asset_name in asset_sel:
            new_array[common_assets.index(asset_name)] = 1.0
        weighted_avg = new_array @ self.portfolio_holdings <= maximum_avg_weight
        function_type = 'maximum_limit_weighted_classification'
        return weighted_avg, function_type,function_name

    def minimum_limit_weighted_classification(self, function_name,attribute, minimum_avg_weight,
                                              asset_classification,common_assets):
        new_array = [0] * len(common_assets)
        asset_sel = [name for name, age in asset_classification.items() if age == attribute]
        for asset_name in asset_sel:
            new_array[common_assets.index(asset_name)] = 1.0
        weighted_avg = new_array @ self.portfolio_holdings_opt >= minimum_avg_weight
        function_type = 'minimum_limit_weighted_classification'
        return weighted_avg,function_type,function_name

    def maximum_number_of_assets(self, function_name, maximum_number_of_assets, *args):
        # an ex-post process
        self.portfolio_holdings_opt=pd.DataFrame(self.portfolio_holdings_opt.value, columns=['Weight'], index=self.asset_name)
        self.portfolio_holdings_opt_top = self.portfolio_holdings_opt.sort_values('Weight').iloc[-maximum_number_of_assets:]
        function_type = 'maximum_number_of_assets'
        return self.portfolio_holdings_opt_top,function_type,function_name

    def convert_weights_to_shares(self, function_name, portfolio_holdings_opt, share_price, total_budget):
        portfolio_holdings_in_shares = total_budget*portfolio_holdings_opt/share_price
        portfolio_holdings_in_shares = portfolio_holdings_in_shares.applymap(math.floor)
        function_type = 'convert_weights_to_shares'
        return portfolio_holdings_in_shares,function_type,function_name

    def optimization_post_processing(self, list_of_post_processing_function):
        return

    def process_objective_function_and_constraints(self, params):
        self.params = params
        constraints = []
        post_optimization_list = []
        for p in params:
            func_type = list(params[p].keys())[0]
            if func_type in ['maximum_number_of_assets', 'convert_weights_to_shares']:
                pass
            elif func_type in ['portfolio_variance', 'expected_returns']:
                if isinstance(params[p][func_type], dict):
                    objective = eval('self.' + func_type)(p, **params[p][func_type])
                else:
                    objective = eval('self.' + func_type)(p, params[p][func_type])
            else:
                if not func_type in ['type_of_optimization']:
                    if isinstance(params[p][func_type], dict):
                        constraints = constraints + [eval('self.' + func_type)(p, **params[p][func_type])]
                    else:
                        constraints = constraints + [eval('self.' + func_type)(p, params[p][func_type])]
                else:
                    type_of_optimization = params[p][func_type]
        return objective, constraints, type_of_optimization, self.relax_constraint_version, post_optimization_list


    def run_optimization(self, params):
        objective, constraints, type_of_optimization, self.relax_constraint_version, post_optimization_list= self.process_objective_function_and_constraints(params)
        self.portfolio_optimizations(objective, constraints, type_of_optimization,self.relax_constraint_version)

    def portfolio_optimizations(self,objective, constraints, type_of_optimization='maximize', relax_constraint_version="beta"):
        print('Solving Portfolio Optimization')
        opt_objective = objective[0]
        opt_constraints = [x[0] for x in constraints]
        if type_of_optimization.lower() in ['maximize', 'max']:
            prob = cp.Problem(cp.Maximize(opt_objective), opt_constraints)
        elif type_of_optimization.lower() in ['minimize', 'min']:
            prob = cp.Problem(cp.Minimize(opt_objective), opt_constraints)
        else:
            raise ValueError('Incorrect Type of Optimization, Please Use: maximize or max or minimize or min')

        prob.solve()
        self.status = prob.status
        if prob.status not in ['infeasible', 'unbounded']:
            return prob.status
        else:
            if relax_constraint_version in ["alpha"]:
                prob = self.optimization_relax_alpha_version(opt_objective,constraints,type_of_optimization)
                return prob.status
            elif relax_constraint_version in ['beta']:
                sol = self.optimization_relax_beta_version()
                # self.sequential_params_iterations()
                return prob.status

    def matching_type_to_parameter(self):
        mapping_dict = {"minimum_limit_multiple_asset_aggregate_holding_constraint": ['minimum_aggregate_holdings', -1.0],
                        'maximum_limit_multiple_asset_aggregate_holding_constraint': ['maximum_aggregate_holdings', 1.0],
                        'maximum_limit_weighted_average': ['maximum_avg_weight',1.0],
                        'minimum_limit_weighted_average': ['minimum_avg_weight',-1.0],
                        'maximum_limit_weighted_classification': ['maximum_avg_weight',1.0],
                        'minimum_limit_weighted_classification': ['minimum_avg_weight',-1.0],
                        'maximum_limit_holding_constraint':['maximum_individual_holdings',1.0],
                        'minimum_limit_holding_constraint':['minimum_individual_holdings',-1.0],
                        'limit_risk_constraint': ['risk_target',1.0]
                        }

        return mapping_dict


    def optimization_relax_alpha_version(self, objective, constraints, type_of_optimization):
        """
        Given that we only have a few constraints and many constraints are considered mandatory, it is possible to
        iterate over all existing constraints and find the one that causes the issue (i.e., which one lead to unsolved
        solution). The method is not recommended for a large scale of optimization. Note that the method is invoked as
        a secondary default to beta version where we relax the constraint sequentially
        :param objective:
        :param constraints:
        :return:
        """
        # add basic constraint
        # long_only_constraint = [x for x in constraints if x[1]=='long_only_constraint'][0]
        # budget_constraint = [x for x in constraints if x[1]=='budget_constraint'][0]
        opt_objective = objective[0]
        constraints_to_add = [x for x in constraints]
        constraints_names_to_add = [x[2] for x in constraints]
        constraints_all_names = [x[2] for x in constraints]
        my_list = range(len(constraints_to_add))
        all_combinations = [list(combinations(my_list, r)) for r in range(1, len(my_list) + 1)]
        all_combinations = [item for sublist in all_combinations for item in sublist]
        # idea is to add as many constraints as possible from the original setting. But some constraints may cause issue
        # In this version, we will REMOVE those constraints and try to solve the problem
        all_combinations = all_combinations[::-1]
        for i in range(len(all_combinations)):
            ac = all_combinations[i]
            constraints_mandatory = []
            constraints_mandatory_name = []
            for a in ac:
                constraints_tmp = constraints_to_add[a][0]
                constraints_mandatory = constraints_mandatory+[constraints_tmp]
                constraints_mandatory_name = constraints_mandatory_name+[constraints_names_to_add[a]]

            if type_of_optimization.lower() in ['maximize', 'max']:
                prob = cp.Problem(cp.Maximize(objective), constraints_mandatory)
            elif type_of_optimization.lower() in ['minimize', 'min']:
                prob = cp.Problem(cp.Minimize(opt_objective), constraints_mandatory)

            prob.solve()
            print ('Removing Constraints to Determine Optimal Solution')
            print (f'Removing Constraints: {set(constraints_all_names)-set(constraints_mandatory_name)}')
            if prob.status in ['optimal']:
                return prob



    def optimization_relax_beta_version(self):
        """
        In this version, we will relax the constraint based on a pre-determined parameter:relaxation_parameter
        It is also an iterative process

        :param objective:
        :param constraints:
        :return:
        """
        # add basic constraint
        newly_added = list(self.params.keys())[-1]
        # basic constraint
        print(f'Relaxing {newly_added} constraint to yield optimal solution. Prop AI Algo Enabled')
        # get relaxation_parameter
        type = list(self.params[newly_added].keys())[0]
        mapping_dict = self.matching_type_to_parameter()
        if self.relaxation_time == 1:
            try:
                self.input_parameter_value = self.params[newly_added][type][mapping_dict[type][0]]
            except:
                self.input_parameter_value = self.params[newly_added][type]
            self.relaxation_parameter = self.input_parameter_value / 2.0

        while self.relaxation_time <= self.max_relaxation_time:
            try:
                self.params[newly_added][type][mapping_dict[type][0]] = self.input_parameter_value + mapping_dict[type][1] * self.relaxation_parameter * self.relaxation_time
                print(self.params[newly_added][type][mapping_dict[type][0]])
            except:
                self.params[newly_added][type] = self.input_parameter_value + mapping_dict[type][1] * self.relaxation_parameter * self.relaxation_time
                print(self.params[newly_added][type])

            objective, constraints, type_of_optimization, self.relax_constraint_version, post_optimization_list = self.process_objective_function_and_constraints(self.params)
            print(f'Relaxing {self.relaxation_time} Time')
            opt_objective = objective[0]
            opt_constraints = [x[0] for x in constraints]
            if type_of_optimization.lower() in ['maximize', 'max']:
                prob = cp.Problem(cp.Maximize(opt_objective), opt_constraints)
            elif type_of_optimization.lower() in ['minimize', 'min']:
                prob = cp.Problem(cp.Minimize(opt_objective), opt_constraints)
            else:
                raise ValueError('Incorrect Type of Optimization, Please Use: maximize or max or minimize or min')
            prob.solve()
            self.status = prob.status
            if prob.status not in ['infeasible', 'unbounded']:
                self.relaxation_time = 1.0
                return prob
            else:
                self.relaxation_time = self.relaxation_time + 1

        if prob.status in ['infeasible', 'unbounded']:
            print(f'There is no Feasible Solution Given the problem. To continue the optimization we will drop this constraint')
            self.params.pop(newly_added)
            self.relaxation_time = 1.0
            objective, constraints, type_of_optimization, self.relax_constraint_version, post_optimization_list = self.process_objective_function_and_constraints(self.params)
            opt_objective = objective[0]
            opt_constraints = [x[0] for x in constraints]
            if type_of_optimization.lower() in ['maximize', 'max']:
                prob = cp.Problem(cp.Maximize(opt_objective), opt_constraints)
            elif type_of_optimization.lower() in ['minimize', 'min']:
                prob = cp.Problem(cp.Minimize(opt_objective), opt_constraints)
            else:
                raise ValueError('Incorrect Type of Optimization, Please Use: maximize or max or minimize or min')
            prob.solve()
            self.status = prob.status
            if prob.status not in ['infeasible', 'unbounded']:
                return prob
            else:
                print(f'There is no After dropping the constraints. Initiate Alpha Relax Version')
                self.optimization_relax_alpha_version(objective, constraints, type_of_optimization)

    def post_optimization_processing(self, share_price, total_budget_in_dollars):
        # convert to shares
        share_price, var_cov_matrix, common_assets = self.attribute_and_cov_alignment(attribute=share_price,var_cov_matrix=self.var_cov_matrix)
        shares = self.convert_weights_to_shares('convert_weights_to_shares', self.portfolio_holdings_opt.value, share_price,
                                              total_budget_in_dollars)
        shares = shares[0].sort_values('previousClose', ascending=False)
        shares = shares[shares != 0].dropna()

        share_price_tmp=share_price.loc[shares.index]

        shares_in_dollar_values = shares*share_price_tmp
        shares_in_dollar_values = shares_in_dollar_values.applymap(math.floor)

        return shares, shares_in_dollar_values

    def use_statistical_risk_model(self,asset_name, start_date="2017-01-01", read_price_from_file=True, price_url=None):
        K = 10
        if read_price_from_file:
            prices = pd.read_csv(price_url, index_col=0)
        else:
            prices = yf.download(asset_name, start=start_date, progress=False)["Adj Close"]
        number_assets = np.size(prices.columns)
        daily_returns = prices.ffill().pct_change().fillna(0)
        mean_returns = annualize_rets(daily_returns, 252)
        factors = pca(returns=daily_returns, n_components=K)
        model = FactorModel(assets=len(prices.columns), k=K)
        # update the model parameters
        model.update(cov=factors.cov * 252, exposure=factors.exposure.values,
                     idiosyncratic_risk=factors.idiosyncratic.std().values * np.sqrt(252),
                     lower_assets=np.zeros(number_assets), upper_assets=np.ones(number_assets),
                     lower_factors=-np.ones(K), upper_factors=np.ones(K))

        asset_covariance = BuildFactorCovariance(factor_covariance=factors.cov * 252,
                                                 factor_exposure=factors.exposure.T,
                                                 specific_risk=factors.idiosyncratic.std() * np.sqrt(252),
                                                 universe=prices.columns.to_list())

        var_cov_matrix = asset_covariance.compute_asset_covariance()
        # assign to var_cov_matrix attribute
        self.var_cov_matrix = var_cov_matrix
        return self.var_cov_matrix

    def attribute_and_cov_alignment(self, attribute, var_cov_matrix):
        """
        attribute and var_cov_matrix alignments due to missing assets occasionally
        :param attribute:
        :param var_cov_matrix:
        :return:
        """
        if attribute is not None:
            portfolio_symbols = attribute.index
            common_assets = var_cov_matrix.index.intersection(portfolio_symbols)
            var_cov_matrix = var_cov_matrix.loc[common_assets][common_assets]
            attribute = attribute.loc[common_assets]
            attribute = attribute[~attribute.index.duplicated(keep='first')]
        else:
            common_assets = var_cov_matrix.index
        self.asset_name = common_assets
        self.var_cov_matrix = var_cov_matrix
        return attribute, var_cov_matrix, common_assets
