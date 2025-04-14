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
# logger= logging.getLogger(__name__)
# logging.basicConfig(level=logger.INFO)

from pyrb.allocation import (RiskBudgetingWithER, ConstrainedRiskBudgeting)
from pyrb import validation
from pyrb.settings import BISECTION_UPPER_BOUND, MAXITER_BISECTION
from scipy.optimize import bisect

# Risk parity is implemented based on jcrichard paper. The code can be found at
# https://github.com/jcrichard/pyrb/blob/master/tests/test_risk_budgeting.py

if __name__ == '__main__':
    import numpy as np
    from pyrb.allocation import (
        EqualRiskContribution,
        RiskBudgeting,
        ConstrainedRiskBudgeting,
    )

    CORRELATIONMATRIX = np.array(
        [
            [1, 0.1, 0.4, 0.5, 0.5],
            [0.1, 1, 0.7, 0.4, 0.4],
            [0.4, 0.7, 1, 0.8, 0.05],
            [0.5, 0.4, 0.8, 1, 0.1],
            [0.5, 0.4, 0.05, 0.1, 1],
        ]
    )
    vol = [0.15, 0.20, 0.25, 0.3, 0.1]
    NUMBEROFASSET = len(vol)
    COVARIANCEMATRIX = CORRELATIONMATRIX * np.outer(vol, vol)
    RISKBUDGET = [0.2, 0.2, 0.3, 0.1, 0.2]
    BOUNDS = np.array([[0.2, 0.3], [0.2, 0.3], [0.05, 0.15], [0.05, 0.15], [0.25, 0.35]])


    def test_erc():
        ERC = EqualRiskContribution(COVARIANCEMATRIX)
        ERC.solve()
        np.testing.assert_almost_equal(np.sum(ERC.x), 1)
        np.testing.assert_almost_equal(
            np.dot(np.dot(ERC.x, COVARIANCEMATRIX), ERC.x) ** 0.5,
            ERC.get_risk_contributions(scale=False).sum(),
            decimal=10,
        )
        np.testing.assert_equal(
            abs(ERC.get_risk_contributions().mean() - 1.0 / NUMBEROFASSET) < 1e-5, True
        )


    def test_rb():
        RB = RiskBudgeting(COVARIANCEMATRIX, RISKBUDGET)
        RB.solve()
        np.testing.assert_almost_equal(np.sum(RB.x), 1, decimal=5)
        np.testing.assert_almost_equal(
            np.dot(np.dot(RB.x, COVARIANCEMATRIX), RB.x) ** 0.5,
            RB.get_risk_contributions(scale=False).sum(),
            decimal=10,
        )
        np.testing.assert_equal(
            abs(RB.get_risk_contributions() - RISKBUDGET).sum() < 1e-5, True
        )


    def test_cerb():
        CRB = ConstrainedRiskBudgeting(
            COVARIANCEMATRIX, budgets=None, pi=None, bounds=BOUNDS
        )
        CRB.solve()
        np.testing.assert_almost_equal(np.sum(CRB.x), 1)
        np.testing.assert_almost_equal(CRB.get_risk_contributions()[1], 0.2455, decimal=5)
        np.testing.assert_almost_equal(np.sum(CRB.x[1]), 0.2)