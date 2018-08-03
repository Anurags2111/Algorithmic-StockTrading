import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import pykalman
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller


X = pd.read_csv('upload.csv')

feat = pd.DataFrame()
feat = X[['timestamp', 'mp_1','mp_2','mp_3']]
feat = feat.set_index('timestamp')
res = pd.DataFrame()

start_time = X.loc[0,'timestamp']



def mae_loss(y, yfit):
    return np.mean(np.abs(y - yfit))


def _train_predict_dlm(args):
    """
    Helper function to train and predict the dynamic linear model for a train and test set. Seperated from the main
    class to enable the use of the multiprocessing module. This should not be called directly.
    """
    delta, X, y, ntrain, loss = args
#    print delta
    dlm = DynamicLinearModel(include_constant=False)

    # first fit using the training data
    dlm.fit(X[:ntrain], y[:ntrain], delta=delta, method='filter')

    # now run the filter on the whole data set
    ntime, pfeat = X.shape
    observation_matrix = X.reshape((ntime, 1, pfeat))
    k = dlm.kalman
    kalman = pykalman.KalmanFilter(transition_matrices=k.transition_matrices,
                                   observation_matrices=observation_matrix,
                                   observation_offsets=k.observation_offsets,
                                   transition_offsets=k.transition_offsets,
                                   observation_covariance=k.observation_covariance,
                                   transition_covariance=k.transition_covariance,
                                   initial_state_mean=k.initial_state_mean,
                                   initial_state_covariance=k.initial_state_covariance)

    beta, bcov = kalman.filter(y)

    # predict the y-values in the test set
    yfit = np.sum(beta[ntrain-1:-1] * X[ntrain-1:-1], axis=1)

    test_error = loss(y[ntrain:], yfit)

    return test_error

class DynamicLinearModel(object):
    def __init__(self, include_constant=True):
        """
        Constructor for linear regression model with dynamic coefficients.
        """
        self.delta_grid = np.zeros(10)
        self.test_grid = np.zeros(10)
        self.delta = 1e-5
        self.test_error_ = 1.0
        self.kalman = pykalman.KalmanFilter()
        self.beta = np.zeros(2)
        self.beta_cov = np.identity(2)
        self.current_beta = np.zeros(2)
        self.current_bcov = np.identity(2)
        self.include_constant = include_constant

    @staticmethod
    def add_constant_(X):
        """
        Add a constant to the linear model by prepending a column of ones to the feature array.
        @param X: The feature array. Note that it will be overwritten, and the overwritten array will be returned.
        """
        if X.ndim == 1:
            # treat vector-valued X differently
            X = np.insert(X[:, np.newaxis], 0, np.ones(len(X)), axis=1)
        else:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

        return X

    def fit(self, X, y, method='smoother', delta=None, include_constant=None):
        """
        Fit the coefficients for the dynamic linear model.
        @param method: The method used to estimate the dynamic coefficients, either 'smoother' or 'filter'. If
            'smoother', then the Kalman Smoother is used, otherwise the Kalman Filter will be used. The two differ
             in the fact that the Kalman Smoother uses both future and past data, while the Kalman Filter only uses
             past data.
        @param X: The time-varying covariates, and (ntime, pfeat) array.
        @param y: The time-varying response, a 1-D array with ntime elements.
        @param delta: The regularization parameters on the time variation of the coefficients. Default is
            self.delta.
        @param include_constant: Boolean, if true then include a constant in the regression model.
        """
        try:
            method.lower() in ['smoother', 'filter']
        except ValueError:
            "method must be either 'smoother' or 'filter'."

        if delta is None:
            delta = self.delta
        else:
            self.delta = delta

        if include_constant is None:
            include_constant = self.include_constant
        else:
            self.include_constant = include_constant

        if include_constant:
            Xtemp = self.add_constant_(X.copy())
        else:
            Xtemp = X.copy()

        ntime, pfeat = Xtemp.shape

        observation_matrix = Xtemp.reshape((ntime, 1, pfeat))
        observation_offset = np.array([0.0])

        transition_matrix = np.identity(pfeat)
        transition_offset = np.zeros(pfeat)

        mu = (1.0 - delta) / delta
        # Var(beta_t - beta_{t-1}) = 1.0 / mu
        transition_covariance = np.identity(pfeat) / mu

        # parameters to be estimated using MLE
        em_vars = ['initial_state_mean', 'initial_state_covariance']
        kalman = pykalman.KalmanFilter(transition_matrices=transition_matrix, em_vars=em_vars,
                                       observation_matrices=observation_matrix,
                                       observation_offsets=observation_offset, transition_offsets=transition_offset,
                                       observation_covariance=np.array([1.0]),
                                       transition_covariance=transition_covariance)

        kalman.em(y)
        if method is 'smoother':
            beta, beta_covar = kalman.smooth(y)
        else:
            beta, beta_covar = kalman.filter(y)

        self.beta = beta
        self.beta_cov = beta_covar
        self.current_beta = beta[-1]
        self.current_bcov = beta_covar[-1]
        self.kalman = kalman

    def update(self, y, x):
        """
        Update the linear regression coefficients given the new values of the response and features.
        @param y: The new response value, a scalar.
        @param x: The new feature vector.
        """
        if self.include_constant:
            observation_matrix = np.insert(x, 0, 1.0)
        else:
            observation_matrix = x.copy()

        pfeat = observation_matrix.size
        observation_matrix = observation_matrix.reshape((1, pfeat))

        self.current_beta, self.current_bcov = \
            self.kalman.filter_update(self.current_beta, self.current_bcov, observation=y,
                                      observation_matrix=observation_matrix)

        self.beta = np.vstack((self.beta, self.current_beta))
        self.beta_cov = np.dstack((self.beta_cov.T, self.current_bcov)).T

    def predict(self, X, pred_type = 'fut'):
        """
        Predict a value of the response given the input feature array and the current value of the coefficients.
        @param x: The input feature array.
        @param pred_type : string input to specify the type of prediction required [ 'past' , 'fut']
        """
        if self.include_constant:
            Xpredict = self.add_constant_(X)
        else:
            Xpredict = X
            
        if pred_type == 'fut':
            return np.sum(self.beta[-1]*Xpredict, axis = 1)
            
        else:
            return np.sum(self.beta * Xpredict, axis = 1)

    def choose_delta(self, X, y, test_fraction=0.5, verbose=False, ndeltas=20, 
                     include_constant=True, loss=mae_loss, njobs=1):
        """
        Choose the optimal regularization parameters for the linear smoother coefficients by minimizing an input loss
        function on a test set.
        @param X: The time-varying covariates, and (ntime, pfeat) array.
        @param y: The training set, a 1-D array.
        @param ndeltas: The number of grid points to use for the regularization parameter.
        @param test_fraction: The fraction of the input data to use as the test set, default is half.
        @param verbose: If true, then print the chosen regularization parameter and test error.
        @param include_constant: Boolean, include a constant in the linear model?
        @param loss: The loss function to use for evaluating the test error when choosing the regularization parameter.
            Must be of the form result = loss(ytest, yfit).
        @param njobs: The number of processors to use when doing the search over delta. If njobs = -1, all processors
            will be used.
        """

        if include_constant is None:
            include_constant = self.include_constant
        else:
            self.include_constant = include_constant

        if njobs < 0:
            njobs = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(njobs)
        pool.map(int, range(njobs))  # warm up the pool

        # split y into training and test sets
        ntime = y.size
        ntest = int(ntime * test_fraction)
        ntrain = ntime - ntest
        if X.ndim == 1:
            XX = X.reshape((X.size, 1))
        else:
            XX = X.copy()

        if include_constant:
            # add column of ones to feature array
            XX = self.add_constant_(XX)

        # grid of delta (regularization) values, between 1e-4 and 1.0.
        delta_grid = np.logspace(-4.0, np.log10(0.95), ndeltas)

        args = []
        for d in xrange(ndeltas):
            args.append((delta_grid[d], XX, y, ntrain, loss))

#         if verbose:
#             print 'Computing test errors...'

        if njobs == 1:
            test_grid = map(_train_predict_dlm, args)
        else:
            test_grid = pool.map(_train_predict_dlm, args)

        test_grid = np.array(test_grid)
        self.delta = delta_grid[test_grid.argmin()]
        self.test_error_ = test_grid.min()

#         if verbose:
#             print 'Best delta is', self.delta, 'and has a test error of', test_grid.min()

        self.delta_grid = delta_grid
        self.test_grid = test_grid


# Training model 
result_list = []
beta_list = []
start_time = X.loc[0,'timestamp']
#start_time += 14.4e9
start_time = feat.index[feat.index.get_loc(start_time,method='nearest')]
res = pd.DataFrame()
# No. of consecutive intervals of 5 min to be tested


intervals = 30

for i in range(0, intervals):
    
    print("********* counter: ", i, " /", " ", intervals, "********")
    
    df = feat
    df_copy = df.copy()
    train = df_copy.loc[start_time:(start_time+3.6e9)]
    test = df_copy.loc[(start_time+3.6e9):(start_time+3.9e9)]
    initial_value = []
    for j in range(1,4):
        initial_value.append(train.loc[start_time,'mp_'+str(j)])
        train['mp_'+str(j)] = (train['mp_'+str(j)] - initial_value[j-1])/initial_value[j-1]*10000
        test['mp_'+str(j)] = (np.array(test['mp_'+str(j)]) - initial_value[j-1])/initial_value[j-1]*10000
    
    X_train = train.iloc[:,1:].values
    y_train = train.iloc[:,0].values
    X_test = test.iloc[:,1:].values
    y_test = test.iloc[:,0].values
    dynamic = DynamicLinearModel(include_constant=False)
    dynamic.fit(X_train, y_train, method = 'smoother',delta = 1e-10 ) 
    y_pred_train = dynamic.predict(X_train, pred_type='past')
    y_pred_test = dynamic.predict(X_test)
    error_train = y_train -  y_pred_train
    print("********** len(error_train)", len(error_train), "*********")
    #if len(error_train) 
    result = adfuller(error_train)
    
    
    #----------------------------------------------------------------------------------------------------
    
    print(str(i) + '. p-value for set with starttime-'+ str(int(start_time))+' :' + str(result[1]))
    error_test = y_test - y_pred_test
    mean = np.mean(error_train)
    std = np.std(error_train)
    z_score_train = (error_train - mean)/std
    z_score_test = (error_test - mean)/std
    test['z_score'] = z_score_test
    res = pd.concat([res,test])
    beta = dynamic.current_beta
    
    temp = result
    temp1 = beta
    res = res
    
    result_list.append(temp)
    beta_list.append(temp1)
    start_time += 3e8
    start_time = feat.index[feat.index.get_loc(start_time,method='nearest')]
   


#res
res = res.dropna(axis =0)

res = res[~res.index.duplicated(keep='first')]

res.to_csv('res72.csv')

#--------------------------------------------------------------------------------------

#res = pd.csv('res72.csv', index_col = 'timestamp')


def beta_values(time,beta_list,start_time):
    start_time += 3.6e9
    diff = (time-start_time)//300000000
    return beta_list[int(diff)]

#beta_values(1206014191009000,beta_list)
    
position = [0,0,0]
price = [0,0,0]
ret_prev = [0,0,0]
total_trades = 0
timestamp_trades = []
total_profit = [0,0,0]
total_returns = [0,0,0]
transaction_profit = 0
transaction_return = 0
trans_rec = []

action = []
q1 = []
q2 = []
q3 = []
sprice = []

POS = []


res['mp_p_1'] = feat['mp_1']
res['mp_p_2'] = feat['mp_2']
res['mp_p_3'] = feat['mp_3']


start_time = X.loc[0,'timestamp']
#start_time += 14.4e9
start_time = feat.index[feat.index.get_loc(start_time,method='nearest')]

for index, row in res.iterrows():
    
    if row['z_score'] >= 2.00 and position[0] == 0:
        action.append('Bought')
        sprice.append(row['mp_p_1'])
        betas = beta_values(index,beta_list,start_time)
        beta_sum = np.sum(np.abs(betas))
        price[0] = row['mp_p_1']
        price[1] = row['mp_p_2']
        price[2] = row['mp_p_3']
        position[0] -= 1000000/row['mp_p_1']
        position[1] += 1000000*betas[0]/(beta_sum*row['mp_p_2'])
        position[2] += 1000000*betas[1]/(beta_sum*row['mp_p_3'])
        POS.append(position)
        timestamp_trades.append([index , -1 , np.sign(position[1]) , np.sign(position[2]) ])       
        
    elif (row['z_score'] >= 1.00 and position[0] > 0) or (row['z_score'] <= -1.00 and position[0] < 0):
        action.append('Exit')
        sprice.append(row['mp_p_1'])
        total_profit[0] += position[0]*(row['mp_p_1'] - price[0] )
        total_profit[1] += position[1]*(row['mp_p_2'] - price[1] )
        total_profit[2] += position[2]*(row['mp_p_3'] - price[2] )
        transaction_profit = position[0]*(row['mp_p_1'] - price[0] )+\
        position[1]*(row['mp_p_2'] - price[1] )+position[2]*(row['mp_p_3'] - price[2] )
        transaction_return = transaction_profit/100
        timestamp_trades.append([index , -1*np.sign(position[0]) , -1*np.sign(position[1]) , -1*np.sign(position[2]) ])
        trans_rec.append([index,transaction_profit,transaction_return])
        position[0] = 0
        position[1] = 0
        position[2] = 0
        POS.append(position)
        total_trades += 1        
        
    elif row['z_score'] <= -2.00 and position[0] == 0:
        action.append('Sold')
        sprice.append(row['mp_p_1'])
        betas = beta_values(index,beta_list,start_time)
        beta_sum = np.sum(np.abs(betas))
        price[0] = row['mp_p_1']
        price[1] = row['mp_p_2']
        price[2] = row['mp_p_3']
        position[0] += 1000000/row['mp_p_1']
        position[1] -= 1000000*betas[0]/(beta_sum*row['mp_p_2'])
        position[2] -= 1000000*betas[1]/(beta_sum*row['mp_p_3'])
        POS.append(position)
        timestamp_trades.append([index , 1 , np.sign(position[1]) , np.sign(position[2]) ]) 
                                        
                    



timestamp_trades = np.array(timestamp_trades)
trades_1 = timestamp_trades[:,[0,1]]
trades_1_pos = trades_1[trades_1[:,1] > 0][:,0]
trades_1_neg = trades_1[trades_1[:,1] < 0][:,0]
trades_2 = timestamp_trades[:,[0,2]]
trades_2_pos = trades_2[trades_2[:,1] > 0][:,0]
trades_2_neg = trades_2[trades_2[:,1] < 0][:,0]
trades_3 = timestamp_trades[:,[0,3]]
trades_3_pos = trades_3[trades_3[:,1] > 0][:,0]
trades_3_neg = trades_3[trades_3[:,1] < 0][:,0]



trades_1_pos_val = [ res.loc[i,'mp_p_1'] for i in trades_1_pos]
trades_2_pos_val = [ res.loc[i,'mp_p_2'] for i in trades_2_pos]
trades_3_pos_val = [ res.loc[i,'mp_p_3'] for i in trades_3_pos]
trades_1_neg_val = [ res.loc[i,'mp_p_1'] for i in trades_1_neg]
trades_2_neg_val = [ res.loc[i,'mp_p_2'] for i in trades_2_neg]
trades_3_neg_val = [ res.loc[i,'mp_p_3'] for i in trades_3_neg]


## avg profit in basis point
trans_rec_np = np.array(trans_rec)
avg_sum = np.average(trans_rec_np[:,2])
avg_sum

res['z_score'].plot()
plt.show()







######################################  TRADE SHEET  ############################################



trades = pd.DataFrame(data = timestamp_trades, columns = ['Timestamp', 'Bank1', 'Bank2', 'Bank3'])

tradesdf = pd.DataFrame(columns = ['Timestamp', 'B1', 'B2', 'B3', 'B1val', 'B2val', 'B3val'])

tradesdf['Timestamp'] = trades['Timestamp']
tradesdf['B1'] = trades['Bank1']
tradesdf['B2'] = trades['Bank2']
tradesdf['B3'] = trades['Bank3']


neg = 0
pos = 0
for var in range(len(tradesdf)):
    if tradesdf['B1'][var] == 1:
        tradesdf['B1val'][var] = trades_1_pos_val[pos]
        pos += 1
    else:
        tradesdf['B1val'][var] = -trades_1_neg_val[neg]
        neg += 1


neg = 0
pos = 0
for var in range(len(tradesdf)):
    if tradesdf['B2'][var] == 1:
        tradesdf['B2val'][var] = trades_2_pos_val[pos]
        pos += 1
    else:
        tradesdf['B2val'][var] = -trades_2_neg_val[neg]
        neg += 1


neg = 0
pos = 0
for var in range(len(tradesdf)):
    if tradesdf['B3'][var] == 1:
        tradesdf['B3val'][var] = trades_3_pos_val[pos]
        pos += 1
    else:
        tradesdf['B3val'][var] = -trades_3_neg_val[neg]
        neg += 1


tradesdf['Action'] = action

cre = []
deb = []

for var in range(len(tradesdf)):
    credit = 0
    debit = 0
    
    if tradesdf['B1val'][var] >= 0:
        credit += tradesdf['B1val'][var]
    else:
        debit += tradesdf['B1val'][var]
    
    if tradesdf['B2val'][var] >= 0:
        credit += tradesdf['B2val'][var]
    else:
        debit += tradesdf['B2val'][var]

    if tradesdf['B3val'][var] >= 0:
        credit += tradesdf['B3val'][var]
    else:
        debit += tradesdf['B3val'][var]
    
    cre.append(credit)
    deb.append(debit)
    

tradesdf['Credit'] = cre
tradesdf['Debit'] = deb
tradesdf['Price'] = sprice
tradesdf['CuCredit'] = tradesdf['Credit'].cumsum()
tradesdf['CuDebit'] = tradesdf['Debit'].cumsum()

tradesdf['trade'] = tradesdf['Credit'] - tradesdf['Debit']
tradesdf['Turnover'] = tradesdf['CuCredit'] - tradesdf['CuDebit']

tradesdf['position'] = tradesdf['B1'] + tradesdf['B2'] + tradesdf['B3']

tradesdf['PNL'] = tradesdf['CuCredit'] + tradesdf['CuDebit'] + tradesdf['position'] * tradesdf['Price']

tradesdf['PNLperTrade'] = tradesdf['PNL'] / tradesdf['Turnover']

tradesdf['PNLafterTC'] = tradesdf['PNL'] - (0.03/100) * tradesdf['PNL']

plt.plot(tradesdf['PNL'])
plt.plot(tradesdf['PNLafterTC'])
plt.legend()
plt.savefig('KALMANPNL.png')
plt.show()


tradesdf.to_csv('KALMANTRADES.csv', index = False)
















