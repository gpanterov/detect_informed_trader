import numpy as np
import pandas as pd
import time
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime

def raw_data(path_to_data):
	""" Loads data in df with appropriate index """
	data = pd.read_csv(path_to_data)
	#time_fmt = '%Y-%m-%d %H:%M:%S'  # use this for the test file
	time_fmt = '%d.%m.%Y %H:%M:%S.%f' # Use this for the full file

	data['Time'] = data.Time.apply(lambda x: datetime.datetime.strptime(x,
							time_fmt))

	year_func = lambda x: x.year
	data['Year'] = data.Time.apply(year_func)

	week_func = lambda x: x.weekofyear
	data['Week'] = data.Time.apply(week_func)

	day_func = lambda x: x.weekday()
	data['Day'] = data.Time.apply(day_func)

	hour_func = lambda x: x.hour
	data['Hour'] = data.Time.apply(hour_func)

	minute_func = lambda x: x.minute
	data['Minute'] = data.Time.apply(minute_func)
	# Pickle
	path_to_folder = '/home/gpanterov/MyProjects/thesis/detect_informed_trader/raw_data/'
	pickle_fname = "usdchf_data.obj"
	f = open(path_to_folder + pickle_fname, 'w')
	pickle.dump(data, f)
	f.close()
	print "Data Frame pickled successfully"
	print pickle_fname

def load_fx_data(pickle_fpath):
	"""
	Loads the pickled forex data. Returns a data frame.
	"""
	t = time.time()
	f = open(pickle_fpath, 'r')
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

def run_reg(df, dis=True):
	data = df.dropna() # drop missing values

	outliers = data.Returns.abs()>6*data.Returns.describe()['std']
	data = data[np.invert(outliers)]

	y = data.Returns.values
	y = np.abs(y)

	x1 = data.Volume.values
	X = np.column_stack((x1,))
	X = sm.add_constant(X)

	model = sm.OLS(y,X).fit()
	if dis:
		print model.summary()
	return model, data

def rolling_estimation(df, train_window, trade_window):
	df.index = np.arange(len(df))
	i = train_window
	all_returns = []
	while i < len(df) - trade_window:
		train_df = df[i - train_window:i]
		#train_df = df[:i]

		model, data = run_reg(train_df, dis=False)
	
		trade_df = df[i:i+trade_window]
		Xexog = sm.add_constant(trade_df.Volume.values)
		Yexog = trade_df.Returns.values
		trade_df['e'] = np.abs(Yexog) - model.predict(Xexog)

		c1 = trade_df.e.shift(1) > np.std(model.resid) * 1.
		c2 = trade_df.Returns.shift(1) > train_df.Returns.describe()['std'] * 1.
		c3 = trade_df.Returns.shift(1) < train_df.Returns.describe()['std'] * 6.

		C = c1.values * c2.values *c3.values

		strat_return = 1. * trade_df.Returns.values[C][2:]
		#all_returns.append(np.sum(strat_return))
		all_returns.extend(strat_return)
		i += trade_window
	return all_returns

def test_strategy(S, strat_return):
	"""
	Test the strategy C (boolean) on series S
	"""

	print np.sum(strat_return)
	random_strats=[]
	for _ in range(100):
		res = np.random.choice(S, len(strat_return))
		random_strats.append(np.sum(res))

	plt.figure(1)
	plt.subplot(211)
	plt.plot(np.cumsum(strat_return))

	plt.subplot(212)
	plt.hist(random_strats)
	plt.axvline(np.sum(strat_return),color="red")

	plt.show()

