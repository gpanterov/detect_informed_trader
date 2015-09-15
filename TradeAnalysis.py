import DataTools as tools
reload(tools)
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd


#pickle_file = '/home/gpanterov/MyProjects/Quants/usdjpy_data.obj'
pickle_file = '/home/gpanterov/MyProjects/thesis/detect_informed_trader/raw_data/eurusd_data.obj'

df =tools.load_fx_data(pickle_file) #Load data
df_agg = tools.collapse_fx_data(df, "10m") # Get lower freq returns

#outliers = df_agg.Returns == df_agg.Returns.max()
#df_agg = df_agg[np.invert(outliers)]
#print "DROPPED %s OUTLIERS" %(np.sum(outliers),)

#model, data = tools.run_reg(df_agg)  # Run return-volume regression


# Strategy conditions
#data['e'] = model.resid

#c1 = data.e.shift(1) > data.e.describe()['std']*1.
#c2 = data.Returns.shift(1) >data.Returns.describe()['std']*1.
#C = c1.values * c2.values

#strat_return = data.Returns.values[C][2:]
#tools.test_strategy(data.Returns, strat_return)  #Test and plot result strats

train_window =10000
ret = tools.rolling_estimation(df_agg, train_window, 500)
tools.test_strategy(df_agg.Returns[train_window:], ret)
