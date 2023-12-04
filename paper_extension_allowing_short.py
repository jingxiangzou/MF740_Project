import numpy as np
import pandas as pd
from pypfopt import HRPOpt
import cvxpy as cp
import warnings
from scipy.optimize import minimize
from copy import deepcopy
import matplotlib.pyplot as plt
TOLERANCE = 1e-10


def _allocation_risk(weights, covariances):
    """
    :param weights: (n*n) numpy.matrix eg: cov1 = np.matrix('1 2 3 ; 4 5 6 ; 1 6 3')
    :param covariances: (n*1) numpy.matrix
    :return: a double value
    """
    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt(np.dot(np.dot(weights, covariances), weights.T))

    # It returns the risk of the weights distribution
    return portfolio_risk


def _assets_risk_contribution_to_allocation_risk(weights, covariances):
    """
    :param weights: (n*n) numpy.matrix eg: cov1 = np.matrix('1 2 3 ; 4 5 6 ; 1 6 3')
    :param covariances: (n*1) numpy.matrix
    :return: a n * 1 matrix
    """
    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    rc_array = np.squeeze(np.asarray(np.dot(covariances, weights.T)))
    wl = []
    for i in range(len(rc_array)):
        wl.append(weights[i] * (portfolio_risk ** (-1)) * rc_array[i])

    # We calculate the contribution of each asset to the risk of the weights
    # distribution

    # np.multiply is the element-wise multiplication of two arrays
    # It returns the contribution of each asset to the risk of the weights
    # distribution
    return np.array(wl)


def _risk_budget_objective_error(weights, args):
    """
    :param weights: (n*1) np.matrix
    :param args[0] : covariances: (n*n) mp.matrix
    :param args[1] : assets_risk_budget: np.array ,
        The desired contribution of each asset to the portfolio risk
        whose elements sums up to 1, is equal to 0.1 in a 10 asset risk parity
    :return : a double value, is the summation of squared error of RC_i for asset_i (i from 1 to 10)
    """
    covariances = args[0]
    # assets_risk_budget = args[1]

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = \
        _assets_risk_contribution_to_allocation_risk(weights, covariances)

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = 0.1 * portfolio_risk

    # np.asmatrix yields a 1 * n matrix
    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = \
        sum([np.square(x - assets_risk_target) for x in assets_risk_contribution])

    # It returns the calculated error
    return error


def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

    # Optimisation process in scipy
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})

    # Recover the weights from the optimised object
    weights = optimize_result.x

    # It returns the optimised weights
    return weights


class optimization:
    
    def __init__(self, covariance_matrix, expected_returns):
        
        self.covmat = covariance_matrix 
        self.n = len(covariance_matrix.iloc[:, 0])
        # covmat should be a pandas DataFrame with column_names and index being the same tickers
        self.mu = expected_returns
        self.main()
        
    def rp_op(self): # risk parity optimization 
        
        initial_ws = [1/self.n] * self.n
        self.rp_weights = _get_risk_parity_weights(self.covmat, initial_ws, initial_ws)
        
        return 0
        
    def hrp_op(self): # HRP optimization 
    
        hrp = HRPOpt(cov_matrix=self.covmat)
        hrp.optimize()
        self.hrp_weights = pd.Series(hrp.clean_weights()).to_numpy()
        
        return 0
    
    def gmv_op(self): # GMV optimization

        w = cp.Variable(self.covmat.shape[0])
        # Defining risk objective
        risk = cp.quad_form(w, self.covmat)
        objective = cp.Minimize(risk)
        # Budget and weights constraints
        constraints = [cp.sum(w) == 1]
        # Solver 
        prob = cp.Problem(objective, constraints)
        prob.solve()
    
        self.gmv_weights = np.round(w.value, 5)
    
        return 0
    
        
    def main(self):
        
        self.hrp_op()
        self.gmv_op()
        self.rp_op()

        return 0
    
    
class BackTesting:
    
    def __init__(self, choice, rolling_window=130):
        
        self.window = rolling_window
        self.get_weights()
        self.choice = choice
        
        self.weights_list = [self.df_gmv_weights, self.df_hrp_weights, self.df_rp_weights]
        weights_df = self.weights_list[choice] # choice = 0, 1 or 2
        returns_df = self.new_rets
        # weights with time instances
        # with out instances the weights dataframe is not accepted 
        self.weights_df = weights_df.iloc[:, 1:]
        self.rets_df = returns_df # the returns of each asset
        self.dates = returns_df.iloc[0, :].to_list()
        self.crypto_rets = returns_df.iloc[:, 0:5]
        self.rfs = returns_df.iloc[:, [0, 5]]
        self.portfolio_returns()
        self.main()
        
        
    def get_weights(self):
        
        df = pd.read_csv("correct_data.csv")
        df = df[['Date', 'BTC', 'LTC', 'XRP', 'DASH', 'RF']]
        df = df.rename(columns={'BTC': 'Bitcoin', 'LTC': 'Litecoin','XRP': 'Ripple', 'DASH': 'Dash'})
        
        for i in range(1,5):
            df.iloc[:,i]=(df.iloc[:,i]/df.iloc[:,i].shift(periods=1, axis=0))-1
            
        df = df.drop(df.index[0])
        rets = deepcopy(df)
        times = rets.Date.to_list()
        
        rolling_window = self.window
        rw = rolling_window
        new_times = times[rw:]
        self.new_rets = rets.iloc[rw:, :]
        self.df_gmv_weights = pd.DataFrame(columns=['BTC', 'LTC', 'XRP', 'DASH'])
        self.df_hrp_weights = pd.DataFrame(columns=['BTC', 'LTC', 'XRP', 'DASH'])
        self.df_rp_weights = pd.DataFrame(columns=['BTC', 'LTC', 'XRP', 'DASH'])

        index = 0
        
        dates_validation = []
        
        self.covmats = []
        
        
        for i in np.arange(rw+1, len(rets.iloc[:, 0])):

            rets_slice = rets.iloc[(i - rw) : i, 1:5] 
            dates_validation.append(rets.iloc[i, 0])
            covmat = rets_slice.cov()
            self.covmats.append(covmat)
            mean = rets_slice.mean()
            op = optimization(covmat, mean)
            self.df_gmv_weights.loc[index] =  op.gmv_weights 
            self.df_hrp_weights.loc[index] = op.hrp_weights
            self.df_rp_weights.loc[index] = op.rp_weights
            index += 1
            
        for ele in [self.df_gmv_weights, self.df_hrp_weights, self.df_rp_weights]:
            
            ele.insert(0, 'Dates', pd.Series(new_times))
        
        

    def portfolio_returns(self):
        
        self.portfolio_returns = pd.DataFrame(columns=['returns'])
        self.dts = []
        
        for i in range(len(self.weights_df)):
            
            ret = np.array(self.weights_df.iloc[i, :]) @ np.array(self.crypto_rets.iloc[i+1, 1:]).T
            self.dts.append(self.crypto_rets.iloc[i+1, 0])
            self.portfolio_returns.loc[i] = [ret / 5] 
        
        self.portfolio_returns['dates'] = pd.Series(self.dts)
        
        return 0
    
    
    def main(self):
        
        self.rfs.index = self.rfs['Date']        
        self.stds = []
        self.times = []
       
        for i in range(len(self.weights_df)):  
            
            mat = np.array(self.weights_df.iloc[i, :])
            covmat3d = np.array(self.covmats[i])
            self.stds.append(np.sqrt(float(mat @ covmat3d @ mat.T)) * np.sqrt(252/5)) 
                        
        self.sharpes = [(self.portfolio_returns.iloc[i, 0] * 252/5 - self.rfs.iloc[i+1, 1]) / self.stds[i] for i in range(len(self.stds))]
                    
        return 0
    
    


if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")
        
    GMV = BackTesting(choice=0, rolling_window=130) # Global Minimum Variance
    HRP = BackTesting(choice=1, rolling_window=130) # Heirarchy Risk Parity
    RP = BackTesting(choice=2, rolling_window=130) # Risk Parity
    
    sh1, sh2, sh3 = GMV.sharpes, HRP.sharpes, RP.sharpes
    sd1, sd2, sd3 = GMV.stds, HRP.stds, RP.stds
    
    times = GMV.dts
    
    df_sharpes = pd.DataFrame({'Date':times, 'GMV': sh1, 'HRP':sh2, 'RP':sh3})
    df_sharpes['Date'] = pd.to_datetime(times)
    df_sharpes.plot(x='Date', y=['GMV', 'HRP', 'RP'], title='Sharpe Ratios')
    
    df_sds = pd.DataFrame({'Date':times, 'GMV': sd1, 'HRP':sd2, 'RP':sd3})
    df_sds['Date'] = pd.to_datetime(times)
    df_sds.plot(x='Date', y=['GMV', 'HRP', 'RP'], title='Portfolio Returns Standard Deviation')
    
    
        
    
    
    
    
    
        

    
    
 
        
        
        
        
     
        
        
        
        
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
        
    
