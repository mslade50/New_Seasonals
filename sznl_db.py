import pandas as pd
import yfinance as yf
from datetime import timedelta
import datetime as dt
from pandas_datareader import data as pdr
import sqlite3
import numpy as np


ticker='^GSPC'
cycle_start=1953
cycle_label='Third Year of Cycle'
cycle_var='pre_election'

def seasonal_return_ranks(ticker,end):
	spx1=yf.Ticker(ticker)
	spx = spx1.history(period="max")

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
	days = yf.download(ticker, start="2024-12-31", end=now)
	days["log_return"] = np.log(days["Close"] / days["Close"].shift(1))*100
	days['day_of_year'] = days.index.day_of_year
	days['this_yr']=days.log_return.cumsum()


	#create your list of all years
	start=spx['year'].min()
	stop=end
	r=range(0,(stop+3-start+1),1)
	# print(start)
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

	#create list of midterm years
	l=range(0,19,1)
	years_mid=[]
	for i in l:
		j=end-i*4
		years_mid.append(j)
	# print(years_mid)
	years_mid2=[]
	for i in l:
		j=end-1-(i*4)
		years_mid2.append(j)

	years_mid3=[]
	for i in l:
		j=end-2-(i*4)
		years_mid3.append(j)

	years_mid4=[]
	for i in l:
		j=end+1-(i*4)
		years_mid4.append(j)

	###cycle years are 1.pre-election 2.election 3.post election 4.midterm

	yr_mid_master=[]
	yr_mid_master2=[]
	yr_mid_master3=[]
	yr_mid_master4=[]
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
	rank=df_all_5d.rank(pct=True).round(3)*100

	df_mt_5d=pd.DataFrame(yr_master_mid2).round(3)
	df_mt_5d_mean=df_mt_5d.mean().round(2)
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
	rank3=df_all_10d.rank(pct=True).round(3)*100

	df_mt_10d=pd.DataFrame(yr_master_mid3).round(3)
	df_mt_10d_mean=df_mt_10d.mean().round(2)
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
	rank5=df_all_21d_mean.rank(pct=True).round(3)*100

	df_mt_21d=pd.DataFrame(yr_master_mid4).round(3)
	df_mt_21d_mean=df_mt_21d.mean().round(2)
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


	length=len(days)

	print_df=cycles_df[cycles_df[ticker] == cycle_var].reset_index(drop=True)
	# print_df=print_df[print_df['level_0'] == length]
	# print_df=print_df.reset_index(drop=True)
	# true_cycle_rnk=print_df['rnk'].iat[-1].round(1)


	returns=[]
	tuples=[df_all_5d_mean,df_mt_5d_mean,df_all_10d_mean,df_mt_10d_mean,df_all_21d_mean,df_mt_21d_mean]
	for data in tuples:
	    returns.append(data)
	new_df=pd.DataFrame(returns).transpose().rename(columns={
	                                                        0:'Fwd_R5',1:'Fwd_R5_MT',
	                                                        2:'Fwd_R10',3:'Fwd_R10_MT',
	                                                        4:'Fwd_R21',5:'Fwd_R21_MT'      
	})

	#5d stuff
	new_df['Returns_5_rnk']=new_df.Fwd_R5.rank(pct=True).round(3)*100
	new_df['Returns_5_rnk_mt']=new_df.Fwd_R5_MT.rank(pct=True).round(3)*100

	r_5=new_df['Fwd_R5'][[length]].round(2)
	r_5_mt=new_df['Fwd_R5_MT'][[length]].round(2)
	r_5_ptile=new_df['Returns_5_rnk'][[length]].round(2)
	r_5_ptile_mt=new_df['Returns_5_rnk_mt'][[length]].round(2)

	#10d stuff
	length=len(days)
	new_df['Returns_10_rnk']=new_df.Fwd_R10.rank(pct=True).round(3)*100
	new_df['Returns_10_rnk_mt']=new_df.Fwd_R10_MT.rank(pct=True).round(3)*100

	r_10=new_df['Fwd_R10'][[length]].round(2)
	r_10_mt=new_df['Fwd_R10_MT'][[length]].round(2)
	r_10_ptile=new_df['Returns_10_rnk'][[length]].round(2)
	r_10_ptile_mt=new_df['Returns_10_rnk_mt'][[length]].round(2)

	#21d stuff
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
	new_df['Returns_rnk_avg']=new_df[['Returns_21_rnk','Returns_10_rnk','Returns_5_rnk']].mean(axis=1)


	df2=pd.DataFrame().assign(Cycle_returns_rank=new_df['Returns_all_avg_mt'],Returns_rank=new_df['Returns_rnk_avg'])

	df2['Rank'] = (3 * df2['Cycle_returns_rank'] + df2['Returns_rank']) / 4
	df2['Rank_5d_avg'] = df2['Rank'].rolling(window=5, center=True, min_periods=1).mean().round(1)
	df3=pd.DataFrame().assign(Expectancy=df2['Rank_5d_avg'])
	# df3=pd.DataFrame().assign(Expectancy=df2['Cycle_returns_rank'])
	df4=df3.values.tolist()

	return df4 


# Set up the database connection
conn = sqlite3.connect('past_sznl.db')
# conn = sqlite3.connect('seasonals.db')
c = conn.cursor()

# Create the table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS seasonal_ranks (
                Date TEXT,
                seasonal_rank REAL,
                ticker TEXT
            )''')
  
def fig_creation(ticker):
    end_date = dt.datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, end=end_date)

    if isinstance(data.columns, pd.MultiIndex):  
        data.columns = data.columns.droplevel(1)

    # Find the year of data inception
    start_year = data.index.min().year
    start_year += 4  # Start the loop 8 years after data inception
    
    # Define the current year
    current_year = dt.datetime.strptime(end_date, '%Y-%m-%d').year

    last_close = data['Close'][-1]
    
    def mdm(ticker, years=list(range(start_year, current_year-3))):	 
        dfs = []
        for year in years:
            try:
                g = seasonal_return_ranks(ticker, year)
                df_year = pd.DataFrame(g, columns=[ticker])
                df_year['Year'] = year + 4  # Adjust here
                df_year['IndexWithinYear'] = df_year.groupby('Year').cumcount()
                df_year['Average_rnk'] = df_year[ticker] / 10
                dfs.append(df_year)
            except Exception as e:
                print(f"No data on {ticker} for the year {year}: {e}")

        df_final = pd.concat(dfs)
        
        # Load your CSV file
        trading_dates_df = pd.read_csv('trading_days.csv')

        # Convert 'Date' column to datetime and extract year
        trading_dates_df['Date'] = pd.to_datetime(trading_dates_df['Date'])
        trading_dates_df['Year'] = trading_dates_df['Date'].dt.year
        trading_dates_df['IndexWithinYear'] = trading_dates_df.groupby('Year').cumcount()

        # Merge df_final and trading_dates_df based on 'Year' and 'IndexWithinYear'
        df_final = pd.merge(df_final.reset_index(drop=True), trading_dates_df, on=['Year', 'IndexWithinYear']).set_index('Date')
        
        return df_final

    # Run the MDM calculation
    df = mdm(ticker)

    # Add ticker column
    df['ticker'] = ticker

    # Select only Date, seasonal_rank (which is labeled as ticker in df), and ticker columns
    df_to_store = df[[ticker, 'ticker']].reset_index().rename(columns={ticker: 'seasonal_rank'})

    # Delete any existing data for this ticker
    conn.execute(f"DELETE FROM seasonal_ranks WHERE ticker = '{ticker}'")

    # Save the new results to the SQL table
    df_to_store.to_sql('seasonal_ranks', conn, if_exists='append', index=False)


# Example usage
tickers = [
  "WMT"
]
  # List of tickers
for ticker in tickers:
    fig_creation(ticker)

# Close the database connection
conn.close()