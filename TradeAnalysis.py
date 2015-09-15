import DataTools as tools
reload(tools)
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd

df =tools.load_fx_data() #Load data
df_agg = tools.collapse_fx_data(df, "15m") # Get hourly returns
model, data = tools.run_reg(df_agg)  # Run return-volume regression


# Strategy conditions
data['e'] = model.resid

c1 = data.e.shift(1) >data.e.describe()['std']*1.
c2 = data.Returns.shift(1) >data.Returns.describe()['std']*1.
C = c1.values * c2.values


tools.test_strategy(data.Returns, C)  #Test and plot result strats


