import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import scipy.stats
import statsmodels.api as sm



###################
# Load Forex Data #
###################
t = time.time()
f = open('/home/gpanterov/MyProjects/Quants/usdjpy_data.obj', 'r')
all_data = pickle.load(f)
print "Loaded data from pickle in ", time.time() - t, " seconds"


df = all_data[all_data.Volume>0]


df['Returns'] = np.log(df.Open) - np.log(df.Close)

###################################################

group = df.groupby([df.Year, df.Week,df.Day, df.Hour])
group_df = group[['Returns', 'Volume']].sum()
group_df['Open'] = group['Open'].first()
group_df = group_df.reset_index()

df_hourly = group_df

hour_dummies = pd.get_dummies(df_hourly.Hour, prefix='hour')
df_hourly = pd.concat((df_hourly, hour_dummies), axis=1)


group_daily = df.groupby([df.Year, df.Week,df.Day])
group_df_daily = pd.DataFrame(group_daily['Returns'].mean())
group_df_daily = group_df_daily.reset_index()

group_df_daily['DailyReturn'] = group_df_daily['Returns']
group_df_daily = group_df_daily.drop('Returns', axis=1)
group_df_daily['InformedDummy'] = np.abs(group_df_daily.DailyReturn) > \
		group_df_daily.DailyReturn.describe()['std']*1.2

df_hourly = pd.merge(df_hourly, group_df_daily, how='left', 
	left_on=['Year', 'Week', 'Day'], right_on=['Year', 'Week', 'Day'])


##############
# Regression #
##############
data = df_hourly.dropna()
data = data[data.Returns != data.Returns.max()] # drop outliers
#data = data[data.Year==2014]

y = data.Returns.values
#y = np.abs((y - np.mean(y))/np.std(y))
y = np.abs(y)

x1 = data.Volume.values
#x1 = (x1 - np.mean(x1))/np.std(x1)
dummies = data[data.columns[7:-2]].values
X = np.column_stack((x1.reshape(len(x1),1) *	dummies,))
#X = sm.add_constant(X)

model = sm.OLS(y,X).fit()
print model.summary()


new_df =pd.DataFrame( model.resid, columns=['e'])
new_df['sign'] = np.sign(data.Returns.values)
new_df['Returns'] = data.Returns.values
new_df['L1_Return'] = data.Returns.shift(1)
new_df['L1_sign'] = new_df.sign.shift(1)
new_df['L1_e'] = new_df.e.shift(1)
new_df['e_large'] = new_df.e > new_df.e.describe()['std']*0.5
new_df['L1_elarge'] = new_df.e_large.shift(1)




c1 = new_df.e.shift(1) > new_df.e.describe()['std']*1.
c2 = new_df.sign.shift(1) > 0
C = c1.values * c2.values

strat_return = new_df.Returns.values[C][10:]
print np.sum(strat_return)

random_strats=[]
for _ in range(100):
	np.random.shuffle(C)
	res = new_df.Returns.values[C][10:]
	random_strats.append(np.sum(res))



plt.figure(1)
plt.subplot(211)
plt.plot(np.cumsum(strat_return))

plt.subplot(212)
plt.hist(random_strats)
plt.axvline(np.sum(strat_return),color="red")



plt.show()


# Other strats

#c1 = new_df.e.shift(1) < new_df.e.describe()['std']*1.
#c3 = new_df.Returns.shift(1) > new_df.Returns.describe()['std']*1.
#C = c1.values * c3.values


