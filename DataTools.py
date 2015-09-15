import numpy as np
import pandas as pd
import time
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt

def load_fx_data():
	"""
	Loads the pickled forex data. Returns a data frame.
	"""
	t = time.time()
	f = open('/home/gpanterov/MyProjects/Quants/usdjpy_data.obj', 'r')
	all_data = pickle.load(f)
	print "Loaded data from pickle in ", time.time() - t, " seconds"
	df = all_data[all_data.Volume>0]
	df['Returns'] = np.log(df.Open) - np.log(df.Close)
	return df


def collapse_fx_data(df, how):
	"""
	Collapsed the data frame according to how.
	how: string (e.g 5m or 3h)
	"""
	assert len(how)<=3
	freq = int(how[:-1])
	interval = how[-1].lower()
	if interval == "m":
		df[how] = df.Minute // freq
		groupby_list = [df.Year, df.Week, df.Day, df.Hour, df[how]]
	elif interval == "h":
		df[how] = df.Hour // freq
		groupby_list = [df.Year, df.Week, df.Day, df[how]]
	elif interval == "d":
		df[how] = df.Day // freq
		groupby_list = [df.Year, df.Week, df[how]]

	group = df.groupby(groupby_list)
	ret_vol_df = group[['Returns', 'Volume']].sum()
	return ret_vol_df.reset_index()

def run_reg(df):
	data = df.dropna()

	outliers = data.Returns.abs()>15*data.Returns.describe()['std']
	data = data[np.invert(outliers)]

	y = data.Returns.values
	y = np.abs(y)

	x1 = data.Volume.values
	X = np.column_stack((x1,))
	X = sm.add_constant(X)

	model = sm.OLS(y,X).fit()
	print model.summary()
	return model, data


def test_strategy(S, C):
	"""
	Test the strategy C (boolean) on series S
	"""

	strat_return = 1. * S.values[C][10:]
	print np.sum(strat_return)
	random_strats=[]
	for _ in range(100):
		np.random.shuffle(C)
		res = S.values[C][10:]
		random_strats.append(np.sum(res))

	plt.figure(1)
	plt.subplot(211)
	plt.plot(np.cumsum(strat_return))

	plt.subplot(212)
	plt.hist(random_strats)
	plt.axvline(np.sum(strat_return),color="red")

	plt.show()

