import pandas as pd
import numpy as np
import datetime as dt
from datetime import date
from datetime import timedelta
import requests
import os
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator

st.title("Single Names")
def seasonals_chart(tick):
	ticker=tick
	cycle_start=1951
	cycle_label='Midterms'
	cycle_var='pre_election'
	adjust=0
	plot_ytd="Yes"
	all_=""
	end_date=dt.datetime(2022,12,30)
	this_yr_end=dt.date.today()


	spx1=yf.Ticker(ticker)
	spx = spx1.history(period="max",end=end_date)
	df= spx1.history(period="max")
	df['200_MA'] = df['Close'].rolling(window=200).mean()
	df['RSI'] = RSIIndicator(df['Close']).rsi()
	df = df[-252:]
	df.reset_index(inplace=True)
	df['date_str'] = range(1,len(df)+1)
	spx_rank=spx1.history(period="max",end=this_yr_end)
	# Calculate trailing 5-day returns
	spx_rank['Trailing_5d_Returns'] = (spx_rank['Close'] / spx_rank['Close'].shift(5)) - 1

	# Calculate trailing 21-day returns
	spx_rank['Trailing_21d_Returns'] = (spx_rank['Close'] / spx_rank['Close'].shift(21)) - 1

	# Calculate percentile ranks for trailing 5-day returns on a rolling 750-day window
	spx_rank['Trailing_5d_percentile_rank'] = spx_rank['Trailing_5d_Returns'].rolling(window=750).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

	# Calculate percentile ranks for trailing 21-day returns on a rolling 750-day window
	spx_rank['Trailing_21d_percentile_rank'] = spx_rank['Trailing_21d_Returns'].rolling(window=750).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

	dr21_rank=(spx_rank['Trailing_21d_percentile_rank'][-1]*100).round(2)
	dr5_rank=(spx_rank['Trailing_5d_percentile_rank'][-1]*100).round(2)

	spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))*100

	spx["day_of_month"] = spx.index.day
	spx['day_of_year'] = spx.index.day_of_year
	spx['month'] = spx.index.month
	spx['Fwd_5dR']=spx.log_return.shift(-5).rolling(window=5).sum().round(2)
	spx['Fwd_10dR']=spx.log_return.shift(-10).rolling(window=10).sum().round(2)
	spx['Fwd_21dR']=spx.log_return.shift(-21).rolling(window=21).sum().round(2)
	spx["year"] = spx.index.year

	#second dataframe explicity to count the number of trading days so far this year
	now = dt.datetime.now()+timedelta(days=1)
	days = yf.download(ticker, start=end_date, end=this_yr_end)
	days["log_return"] = np.log(days["Close"] / days["Close"].shift(1))*100
	days['day_of_year'] = days.index.day_of_year
	days['this_yr']=days.log_return.cumsum()
	days2=days.reset_index(drop=True)
	length=len(days)+adjust


	#create your list of all years
	start=spx['year'].min()
	stop=spx['year'].max()
	r=range(0,(stop-start+1),1)
	print(start)
	years=[]
	for i in r:
		j=start+i
		years.append(j)
	# print(years)


	def yearly(time):
			rslt_df2 = spx.loc[spx['year']==time] 
			grouped_by_day = rslt_df2.groupby("day_of_year").log_return.mean()
			day_by_day=[]
			for day in grouped_by_day:
				cum_return = day
				day_by_day.append(cum_return)
			return day_by_day

	def yearly_5d(time):
		rslt_df2 = spx.loc[spx['year']==time]
		fwd_5_by_day=rslt_df2.groupby("day_of_year").Fwd_5dR.mean()
		day_by_day_5=[]
		for day in fwd_5_by_day:
			cum_return = day
			day_by_day_5.append(cum_return)
		return day_by_day_5

	def yearly_10d(time):
		rslt_df2 = spx.loc[spx['year']==time]
		fwd_5_by_day=rslt_df2.groupby("day_of_year").Fwd_10dR.mean()
		day_by_day_5=[]
		for day in fwd_5_by_day:
			cum_return = day
			day_by_day_5.append(cum_return)
		return day_by_day_5

	def yearly_21d(time):
		rslt_df2 = spx.loc[spx['year']==time]
		fwd_5_by_day=rslt_df2.groupby("day_of_year").Fwd_21dR.mean()
		day_by_day_5=[]
		for day in fwd_5_by_day:
			cum_return = day
			day_by_day_5.append(cum_return)
		return day_by_day_5


	yr_master=[]
	for year in years:
		yearly(year)
		yr_master.append(yearly(year))

	#create list of years corresponding to specific year within 4 year presidential cycle
	l=range(0,19,1)
	years_mid=[]
	for i in l:
		j=cycle_start+i*4
		years_mid.append(j)
	print(years_mid)
	years_mid2=[]
	for i in l:
		j=cycle_start+1+(i*4)
		years_mid2.append(j)

	years_mid3=[]
	for i in l:
		j=cycle_start+2+(i*4)
		years_mid3.append(j)

	years_mid4=[]
	for i in l:
		j=cycle_start-1+(i*4)
		years_mid4.append(j)

	###cycle years start w/ whatever the current year of the cycle is and then move forward from there. EG if it is 2019 then 1. would be pre_election 2. would be election etc.

	###create your empty lists to add your [1d] returns for each year of the cycle.
	yr_mid_master=[]
	yr_mid_master2=[]
	yr_mid_master3=[]
	yr_mid_master4=[]

	###run the one day forward returns function on each set of cycle years to get 4 sets of returns. 
	###Once this for loop is done you'll be left with 4 lists of daily returns for each year that you had in the list
	for year in years_mid:
		yearly(year)
		yr_mid_master.append(yearly(year))
	for year in years_mid2:
		yearly(year)
		yr_mid_master2.append(yearly(year))
	for year in years_mid3:
		yearly(year)
		yr_mid_master3.append(yearly(year))
	for year in years_mid4:
		yearly(year)
		yr_mid_master4.append(yearly(year))

	yr_master2=[]
	for year in years:
		yearly_5d(year)
		yr_master2.append(yearly_5d(year))

	yr_master_mid2=[]
	yr_master_mid22=[]
	yr_master_mid23=[]
	yr_master_mid24=[]
	for year in years_mid:
		yearly_5d(year)
		yr_master_mid2.append(yearly_5d(year))
	for year in years_mid2:
		yearly_5d(year)
		yr_master_mid22.append(yearly_5d(year))
	for year in years_mid3:
		yearly_5d(year)
		yr_master_mid23.append(yearly_5d(year))
	for year in years_mid4:
		yearly_5d(year)
		yr_master_mid24.append(yearly_5d(year))

	yr_master3=[]
	for year in years:
		yearly_10d(year)
		yr_master3.append(yearly_10d(year))

	yr_master_mid3=[]
	yr_master_mid32=[]
	yr_master_mid33=[]
	yr_master_mid34=[]
	for year in years_mid:
		yearly_10d(year)
		yr_master_mid3.append(yearly_10d(year))
	for year in years_mid2:
		yearly_10d(year)
		yr_master_mid32.append(yearly_10d(year))
	for year in years_mid3:
		yearly_10d(year)
		yr_master_mid33.append(yearly_10d(year))
	for year in years_mid4:
		yearly_10d(year)
		yr_master_mid34.append(yearly_10d(year))

	yr_master4=[]
	for year in years:
		yearly_21d(year)
		yr_master4.append(yearly_21d(year))

	yr_master_mid4=[]
	yr_master_mid42=[]
	yr_master_mid43=[]
	yr_master_mid44=[]
	for year in years_mid:
		yearly_21d(year)
		yr_master_mid4.append(yearly_21d(year))
	for year in years_mid2:
		yearly_21d(year)
		yr_master_mid42.append(yearly_21d(year))
	for year in years_mid3:
		yearly_21d(year)
		yr_master_mid43.append(yearly_21d(year))
	for year in years_mid4:
		yearly_21d(year)
		yr_master_mid44.append(yearly_21d(year))

	###you are now converting your lists of returns into dataframes, and then manipulating the resulting data to get averages across all years from the same day.
	###this process is repeated for each cycle year, and for 5d 10d and 21d forward returns. 
	df_all_5d=pd.DataFrame(yr_master2).round(3)
	df_all_5d_mean=df_all_5d.mean().round(2)
	df_all_5d_mad=df_all_5d.mad().round(2)
	df_all_5d_median=df_all_5d.median().round(2)
	rank=df_all_5d.rank(pct=True).round(3)*100

	df_mt_5d=pd.DataFrame(yr_master_mid2).round(3)
	df_mt_5d_mean=df_mt_5d.mean().round(2)
	df_mt_5d_mad=df_mt_5d.mad().round(2)
	df_mt_5d_median=df_mt_5d.median().round(2)
	rank2=df_mt_5d.rank(pct=True).round(3)*100

	df_mt2_5d=pd.DataFrame(yr_master_mid22).round(3)
	df_mt2_5d_mean=df_mt2_5d.mean().round(2)
	rank22=df_mt2_5d.rank(pct=True).round(3)*100

	df_mt3_5d=pd.DataFrame(yr_master_mid23).round(3)
	df_mt3_5d_mean=df_mt3_5d.mean().round(2)
	rank23=df_mt3_5d.rank(pct=True).round(3)*100

	df_mt4_5d=pd.DataFrame(yr_master_mid24).round(3)
	df_mt4_5d_mean=df_mt4_5d.mean().round(2)
	rank24=df_mt4_5d.rank(pct=True).round(3)*100

	df_all_10d=pd.DataFrame(yr_master3).round(3)
	df_all_10d_mean=df_all_10d.mean().round(2)
	df_all_10d_mad=df_all_10d.mad().round(2)
	df_all_10d_median=df_all_10d.median().round(2)
	rank3=df_all_10d.rank(pct=True).round(3)*100

	df_mt_10d=pd.DataFrame(yr_master_mid3).round(3)
	df_mt_10d_mean=df_mt_10d.mean().round(2)
	df_mt_10d_mad=df_mt_10d.mad().round(2)
	df_mt_10d_median=df_mt_10d.median().round(2)
	rank4=df_mt_10d.rank(pct=True).round(3)*100

	df_mt2_10d=pd.DataFrame(yr_master_mid32).round(3)
	df_mt2_10d_mean=df_mt2_10d.mean().round(2)
	rank42=df_mt2_10d.rank(pct=True).round(3)*100

	df_mt3_10d=pd.DataFrame(yr_master_mid33).round(3)
	df_mt3_10d_mean=df_mt3_10d.mean().round(2)
	rank43=df_mt3_10d.rank(pct=True).round(3)*100

	df_mt4_10d=pd.DataFrame(yr_master_mid34).round(3)
	df_mt4_10d_mean=df_mt4_10d.mean().round(2)
	rank44=df_mt4_10d.rank(pct=True).round(3)*100

	df_all_21d=pd.DataFrame(yr_master4).round(3)
	df_all_21d_mean=df_all_21d.mean().round(2)
	df_all_21d_mad=df_all_21d.mad().round(2)
	df_all_21d_median=df_all_21d.median().round(2)
	rank5=df_all_21d_mean.rank(pct=True).round(3)*100

	df_mt_21d=pd.DataFrame(yr_master_mid4).round(3)
	df_mt_21d_mean=df_mt_21d.mean().round(2)
	df_mt_21d_mad=df_mt_21d.mad().round(2)
	df_mt_21d_median=df_mt_21d.median().round(2)
	rank6=df_mt_21d_mean.rank(pct=True).round(3)*100

	df_mt2_21d=pd.DataFrame(yr_master_mid42).round(3)
	df_mt2_21d_mean=df_mt2_21d.mean().round(2)
	rank62=df_mt2_21d_mean.rank(pct=True).round(3)*100

	df_mt3_21d=pd.DataFrame(yr_master_mid43).round(3)
	df_mt3_21d_mean=df_mt3_21d.mean().round(2)
	rank63=df_mt3_21d_mean.rank(pct=True).round(3)*100

	df_mt4_21d=pd.DataFrame(yr_master_mid44).round(3)
	df_mt4_21d_mean=df_mt4_21d.mean().round(2)
	rank64=df_mt4_21d_mean.rank(pct=True).round(3)*100

	pre_election=[]
	election=[]
	post_election=[]
	midterms=[]

	pre_election_list=[df_mt_5d_mean,df_mt_10d_mean,df_mt_21d_mean]
	election_list=[df_mt2_5d_mean,df_mt2_10d_mean,df_mt2_21d_mean]
	post_election_list=[df_mt3_5d_mean,df_mt3_10d_mean,df_mt3_21d_mean]
	midterms_list=[df_mt4_5d_mean,df_mt4_10d_mean,df_mt4_21d_mean]

	for g in pre_election_list:
		pre_election.append(g)
	pre_election_df=pd.DataFrame(pre_election).transpose().rename(columns={0:'Fwd_R5',1:'Fwd_R10',2:'Fwd_R21',})
	pre_election_df['avg']=pre_election_df.mean(axis=1).round(2)

	for g in election_list:
		election.append(g)
	election_df=pd.DataFrame(election).transpose().rename(columns={0:'Fwd_R5',1:'Fwd_R10',2:'Fwd_R21',})
	election_df['avg']=election_df.mean(axis=1).round(2)

	for g in post_election_list:
		post_election.append(g)
	post_election_df=pd.DataFrame(post_election).transpose().rename(columns={0:'Fwd_R5',1:'Fwd_R10',2:'Fwd_R21',})
	post_election_df['avg']=post_election_df.mean(axis=1).round(2)

	for g in midterms_list:
		midterms.append(g)
	midterms_df=pd.DataFrame(midterms).transpose().rename(columns={0:'Fwd_R5',1:'Fwd_R10',2:'Fwd_R21',})
	midterms_df['avg']=midterms_df.mean(axis=1).round(2)

	cycles_df=pd.concat([pre_election_df['avg'],election_df['avg'],post_election_df['avg'],midterms_df['avg']],axis=1,keys=['pre_election','election','post_election','midterms'])
	cycles_df=cycles_df.stack().reset_index()
	cycles_df.columns.values[2]="avg"
	cycles_df.columns.values[1]=ticker
	cycles_df['rnk']=cycles_df.avg.rank(pct=True).round(3)*100


	print_df1=cycles_df[cycles_df[ticker] == cycle_var].reset_index(drop=True)
	trailing_cycle=print_df1.iloc[length-5:length].rnk.mean()
	print_df=print_df1[print_df1['level_0'] == length]
	print_df=print_df.reset_index(drop=True)
	true_cycle_rnk=print_df['rnk'].iat[-1].round(1)


	returns=[]
	tuples=[df_all_5d_mean,df_mt_5d_mean,df_all_5d_mad,df_mt_5d_mad,df_all_10d_mean,df_mt_10d_mean,df_all_10d_mad,df_mt_10d_mad,df_all_21d_mean,df_mt_21d_mean,df_all_21d_mad,df_mt_21d_mad]
	for data in tuples:
		returns.append(data)
	new_df=pd.DataFrame(returns).transpose().rename(columns={
															0:'Fwd_R5',1:'Fwd_R5_MT',2:'Fwd_mad5',3:'Fwd_mad5_MT',
															4:'Fwd_R10',5:'Fwd_R10_MT',6:'Fwd_mad10',7:'Fwd_mad10_MT',
															8:'Fwd_R21',9:'Fwd_R21_MT',10:'Fwd_mad21',11:'Fwd_mad21_MT',

	})

	#5d stuff

	new_df['Variance_rnk']=new_df.Fwd_mad5.rank(pct=True).round(3)*100
	new_df['Variance_rnk_MT']=new_df.Fwd_mad5_MT.rank(pct=True).round(3)*100
	new_df['Returns_5_rnk']=new_df.Fwd_R5.rank(pct=True).round(3)*100
	new_df['Returns_5_rnk_mt']=new_df.Fwd_R5_MT.rank(pct=True).round(3)*100

	r_5=new_df['Fwd_R5'][[length]].round(2)
	r_5_mt=new_df['Fwd_R5_MT'][[length]].round(2)
	r_5_ptile=new_df['Returns_5_rnk'][[length]].round(2)
	r_5_ptile_mt=new_df['Returns_5_rnk_mt'][[length]].round(2)
	variance_ptile=new_df['Variance_rnk'][[length]].round(2)
	variance_ptile_mt=new_df['Variance_rnk_MT'][[length]].round(2)

	#10d stuff

	new_df['Variance_rnk_10d']=new_df.Fwd_mad10.rank(pct=True).round(3)*100
	new_df['Variance_rnk_10d_MT']=new_df.Fwd_mad10_MT.rank(pct=True).round(3)*100
	new_df['Returns_10_rnk']=new_df.Fwd_R10.rank(pct=True).round(3)*100
	new_df['Returns_10_rnk_mt']=new_df.Fwd_R10_MT.rank(pct=True).round(3)*100

	r_10=new_df['Fwd_R10'][[length]].round(2)
	r_10_mt=new_df['Fwd_R10_MT'][[length]].round(2)
	r_10_ptile=new_df['Returns_10_rnk'][[length]].round(2)
	r_10_ptile_mt=new_df['Returns_10_rnk_mt'][[length]].round(2)
	variance_ptile_10=new_df['Variance_rnk_10d'][[length]].round(2)
	variance_ptile_10_mt=new_df['Variance_rnk_10d_MT'][[length]].round(2)

	#21d stuff
	new_df['Variance_rnk_1m']=new_df.Fwd_mad21.rank(pct=True).round(3)*100
	new_df['Variance_rnk_1m_MT']=new_df.Fwd_mad21_MT.rank(pct=True).round(3)*100
	new_df['Returns_21_rnk']=new_df.Fwd_R21.rank(pct=True).round(3)*100
	new_df['Returns_21_rnk_mt']=new_df.Fwd_R21_MT.rank(pct=True).round(3)*100

	##Calculate average ranks across the row
	new_df['Returns_all_avg']=new_df[['Returns_21_rnk','Returns_10_rnk','Returns_5_rnk']].mean(axis=1)
	new_df['Returns_all_avg_mt']=new_df[['Returns_21_rnk_mt','Returns_10_rnk_mt','Returns_5_rnk_mt']].mean(axis=1)
	new_df['Returns_all_avg_10dt']=new_df['Returns_all_avg'].rolling(window=10).mean().shift(1)
	new_df['Returns_all_avg_mt_10dt']=new_df['Returns_all_avg_mt'].rolling(window=10).mean().shift(1)
	new_df['Seasonal_delta']=new_df.Returns_all_avg - new_df.Returns_all_avg_10dt
	new_df['Seasonal_delta_cycle']=new_df.Returns_all_avg_mt - new_df.Returns_all_avg_mt_10dt


	r_21=new_df['Fwd_R21'][[length]].round(2)
	r_21_mt=new_df['Fwd_R21_MT'][[length]].round(2)
	r_21_ptile=new_df['Returns_21_rnk'][[length]].round(2)
	r_21_ptile_mt=new_df['Returns_21_rnk_mt'][[length]].round(2)
	variance_ptile_21=new_df['Variance_rnk_1m'][[length]].round(2)
	variance_ptile_21_mt=new_df['Variance_rnk_1m_MT'][[length]].round(2)


	##Output
	all_5d=r_5_ptile.values[0]
	mt_5d=r_5_ptile_mt.values[0]
	all_10d=r_10_ptile.values[0]
	mt_10d=r_10_ptile_mt.values[0]
	all_21d=r_21_ptile.values[0]
	mt_21d=r_21_ptile_mt.values[0]
	all_avg=((all_5d+all_10d+all_21d)/3).round(2)
	cycle_avg=true_cycle_rnk.round(1)
	total_avg=((all_avg+true_cycle_rnk)/2).round(1) 
	trailing_21_rank=dr21_rank.round(1)
	trailing_5_rank=dr5_rank.round(1)


	if ticker == '^GSPC':
		ticker2 = 'SPX'
	else:
		ticker2 = ticker

	length= len(days) + adjust
	c=days.Close[-1]

	dfm=pd.DataFrame(yr_mid_master)
	dfm1=dfm.mean()
	upper=(np.std(dfm))*2+dfm1
	lower=dfm1-(np.std(dfm))*2

	s4=dfm1.cumsum()
	dfy=pd.DataFrame(yr_master)
	dfy1=dfy.mean()
	s3=dfy1.cumsum()
	##Mean Return paths chart (looks like a classic 'seasonality' chart)

	# Assuming df is your DataFrame and it has 'Close' column
	df['max_rolling'] = df['Close'].rolling(window=41).max().shift(-20)
	df['min_rolling'] = df['Close'].rolling(window=41).min().shift(-20)

	df['pivot_point'] = np.where((df['Close'] == df['max_rolling']) | (df['Close'] == df['min_rolling']), df['Close'], np.nan)

	# Get pivot points for the last 252 days
	pivot_points_last_252 = df[df['pivot_point'].notna()].tail(252)

	fig = go.Figure()

	fig.add_trace(go.Scatter(x=s4.index, y=s4.values, mode='lines', name=cycle_label, line=dict(color='orange')))
	if plot_ytd == 'Yes':
	    fig.add_trace(go.Scatter(x=days2.index, y=days2['this_yr'], mode='lines', name='Year to Date', line=dict(color='green')))
	y1 = max(s4.max(), days2['this_yr'].max()) if plot_ytd == 'Yes' else s4.max()
	y0=min(s4.min(),days2['this_yr'].min(),0)
	# Assuming 'length' variable is defined and within the range of the x-axis
	length_value = length

	# Interpolate Y value at the specified X coordinate
	y_value_at_length = np.interp(length_value, s4.index, s4.values)
	s4_values = s4.values[:length]
	this_year_values = days2['this_yr'][:length]
	this_year_values = np.where(np.isnan(this_year_values), np.nanmean(this_year_values), this_year_values)

	if np.isnan(s4_values).any() or np.isnan(this_year_values).any() or np.var(s4_values) == 0 or np.var(this_year_values) == 0:
	    correlation_coefficient = 'N/A'
	else:
	    correlation_matrix = np.corrcoef(s4_values, this_year_values)
	    correlation_coefficient = correlation_matrix[0, 1]
	    correlation_coefficient = f"{correlation_coefficient:.2f}"
	def sign_agreement(a, b, window):
	    a_changes = a[window:] - a[:-window]
	    b_changes = b[window:] - b[:-window]
	    return np.mean(np.sign(a_changes) == np.sign(b_changes))

# 	correlations = []
# 	window_size = 5

# 	for i in range(len(s4_values) - window_size + 1):
# 		window_s4 = s4_values[i : i + window_size]
# 		window_this_year = this_year_values[i : i + window_size]
# 		if np.isnan(window_s4).any() or np.isnan(window_this_year).any() or np.var(window_s4) == 0 or np.var(window_this_year) == 0:
# 			correlations.append(np.nan)
# 		else:
# 			correlation_matrix = np.corrcoef(window_s4, window_this_year)
# 			correlation_coefficient = correlation_matrix[0, 1]
# 			correlations.append(correlation_coefficient)

# 	average_correlation = pd.Series(correlations).mean()
# 	average_correlation = f"{average_correlation:.2f}"

	# Calculate sign agreement for 5-day, 10-day, and 21-day forward changes
	sign_agreement_1d = sign_agreement(s4_values, this_year_values, window=1).round(2)
	sign_agreement_5d = sign_agreement(s4_values, this_year_values, window=5).round(2)
	sign_agreement_10d = sign_agreement(s4_values, this_year_values, window=10).round(2)
	sign_agreement_21d = sign_agreement(s4_values, this_year_values, window=21).round(2)
	# Add a white dot at the specified X coordinate and the interpolated Y value
	fig.add_trace(go.Scatter(x=[length_value], y=[y_value_at_length], mode='markers', marker=dict(color='white', size=8), name='White Dot' ,showlegend=False))
	def text_color(value, reverse=False):
	    if not reverse:
	        if value >= 85:
	            return 'green'
	        elif value <= 15:
	            return 'red'
	        else:
	            return 'white'
	    else:
	        if value >= 85:
	            return 'red'
	        elif value <= 15:
	            return 'green'
	        else:
	            return 'white'
	def create_annotation(x, y, text, color):
	    return dict(
		x=x,
		y=y,
		xref='paper',
		yref='paper',
		text=text,
		showarrow=False,
		font=dict(size=12, color=color),
		bgcolor='rgba(0, 0, 0, 0.5)',
		bordercolor='grey',
		borderwidth=1,
		borderpad=4,
		align='left'
	    )

	annotations = [
	    create_annotation(0.4, -0.22, f"Cycle Avg: {cycle_avg}", text_color(cycle_avg)),
	    create_annotation(0.55, -0.22, f"Total Avg: {total_avg}", text_color(total_avg)),
	    create_annotation(0.85, -0.22, f"Trailing 21 Rank: {trailing_21_rank}", text_color(trailing_21_rank, reverse=True)),
	    create_annotation(1.04, -0.22, f"Trailing 5 Rank: {trailing_5_rank}", text_color(trailing_5_rank, reverse=True)),
	]
	annotations.append(
	    create_annotation(
		1.02,
		1.10,
		f"5d and 21d Concordance: {sign_agreement_5d}, {sign_agreement_21d}",
		'white'
	    )
	)
# 	annotations.append(create_annotation(0.95, 1.12, f"Average 14-Period Rolling Correlation: {average_correlation}", 'white'))
	fig.update_layout(
	    title=f"Mean return path for {ticker2} in years {start}-present",
	    legend=dict(
		bgcolor='rgba(0,0,0,0)',
		font=dict(color='White'),
		itemclick='toggleothers',
		itemdoubleclick='toggle',
		traceorder='reversed',
		orientation='h',
		bordercolor='grey',
		borderwidth=1,
		x=-0.10,
		y=-0.135 
	    ),
	    xaxis=dict(title='', color='white',showgrid=False),
	    yaxis=dict(title='Mean Return', color='white',showgrid=False),
	    font=dict(color='white'),
	    margin=dict(l=40, r=40, t=40, b=70),  # Increase bottom margin
	    hovermode='x',
	    plot_bgcolor='Black',
	    paper_bgcolor='Black',
	    annotations=annotations  # Use the new annotations list with colored text
	)
	# Create a candlestick chart
	fig2 = go.Figure()

	# Add only the Price (Candlestick) trace and the 200_MA trace
	fig2.add_trace(go.Candlestick(x=df['date_str'],
				     open=df['Open'],
				     high=df['High'],
				     low=df['Low'],
				     close=df['Close'], name='Price'))

	fig2.add_trace(go.Scatter(x=df['date_str'], y=df['200_MA'], name='200_MA', line=dict(color='purple')))

	# Add pivot point rays
	for _, row in pivot_points_last_252.iterrows():
	    fig2.add_shape(type='line',
			  x0=row['date_str'], y0=row['pivot_point'], x1=df['date_str'].iloc[-1], y1=row['pivot_point'],
			  xref='x', yref='y',
			  line=dict(color='Orange', width=1))

	# Finalize layout
	fig2.update_layout(height=800,
			  width=1200,
			  xaxis=dict(
			      rangeslider=dict(
				  visible=False
			      )
			  ))

	fig2.update_xaxes(showgrid=False)
	fig2.update_yaxes(showgrid=False)

	st.plotly_chart(fig)
	st.plotly_chart(fig2)

megas_list=['AAPL','AMD','AMZN','DIS','GOOG','GS','HD','JNJ','JPM','LIN','LMT','MCD','MRK','MSFT','NFLX','NKE','NVDA','OXY','TSLA','TSM','TXN','UNH','V','WMT','XOM']
for stock in megas_list:
	seasonals_chart(stock)
