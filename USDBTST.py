import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
pd.plotting.register_matplotlib_converters()
pd.set_option('display.max_column', 5000)
pd.set_option('display.max_row', 5000)
pd.set_option('display.width', 5000)
plt.style.use('seaborn')

slippage = 0.001

df = pd.read_csv(r'C:\Users\sudipto\Dropbox\QUANTORA\Research\Trading\dataset\NSEpydata\USD.csv')
df[['Date', 'Expiry']] = df[['Date', 'Expiry']].apply(pd.to_datetime)
df.sort_values('Date', inplace=True)
df.drop(df.columns[[1, 3, 4, 9, 10, 11, 12]], axis=1, inplace=True)
df.set_index('Date', inplace=True)

Long_state = (df['Expiry'] == df['Expiry'].shift(1)) & (df['Expiry'] == df['Expiry'].shift(-1)) & (df['Close'] > df['Close'].shift(1))
Short_state = (df['Expiry'] == df['Expiry'].shift(1)) & (df['Expiry'] == df['Expiry'].shift(-1)) & (df['Close'] < df['Close'].shift(1))
State_condition = [Long_state, Short_state]
State_choice = [1, -1]
df['State'] = np.select(State_condition, State_choice)

LkaSL = (df['State'].shift(1) == 1) & (df['Low'].shift(1) > df['Low'])
SkaSL = (df['State'].shift(1) == -1) & (df['High'].shift(1) < df['High'])
SL_condition = [LkaSL, SkaSL]
SL_choice = [((df['Low'].shift(1) - df['Close'].shift(1) - (df['Close'].shift(1) * slippage))/df['Close'].shift(1)),
             ((df['Close'].shift(1) - df['High'].shift(1) - (df['Close'].shift(1) * slippage))/df['Close'].shift(1))]

df['SL'] = np.select(SL_condition, SL_choice)

Long_exit = (df['SL'] == 0) & (df['State'].shift(1) == 1) & (df['State'] != 1)
Long_carry = (df['SL'] == 0) & (df['State'].shift(1) == 1) & (df['State'] == 1)
Short_exit = (df['SL'] == 0) & (df['State'].shift(1) == -1) & (df['State'] != -1)
Short_carry = (df['SL'] == 0) & (df['State'].shift(1) == -1) & (df['State'] == -1)
SL_exit = df['SL'] != 0
Payoff_condition = [Long_exit, Long_carry, Short_exit, Short_carry, SL_exit]
Payoff_choice = [df['Close'].pct_change() - slippage, df['Close'].pct_change(),
                 -1*(df['Close'].pct_change()) - slippage, -1 * (df['Close'].pct_change()), df['SL']]
df['Payoff'] = np.around(np.select(Payoff_condition, Payoff_choice), decimals=6)
df['Cumulative'] = df['Payoff'].cumsum()

# df.to_csv(r'D:\out.csv')
print(df)

print(' \n-o-o-o-o-o  RISK MEASURES  o-o-o-o-o-\n')

netPL = np.around(df['Cumulative'].iloc[-1]*100, decimals=2)
print('Net P/L:\n\t', netPL, '% (Unleveraged)')

df['Payoff'].replace(0, np.NaN, inplace=True)
sharpe = np.around((252**0.5)*(df['Payoff'].mean()/np.std(df['Payoff'])), decimals=4)
print('Sharpe Ratio:\n\t', sharpe)

hit_rate = np.around((len(df.loc[df.Payoff > 0])/len(df['Payoff']))*100, decimals=2)
print('Hit Rate:\n\t', hit_rate, '%')

gain_mean = df[df['Payoff'] >= 0].Payoff.mean()
loss_mean = df[df['Payoff'] <= 0].Payoff.mean()
win_loss = np.around(abs(gain_mean/loss_mean), decimals=2)
print('Win/Loss:\n\t', win_loss)

max_gain = np.around((np.nanmax(df['Payoff'].values)*100), decimals=2)
print('Max Gain:\n\t', max_gain, '%')

max_loss = np.around((np.nanmin(df['Payoff'].values)*100), decimals=2)
print('Max Loss:\n\t', max_loss, '%')

prev = float("-inf")
drawdown = []
for i in df['Cumulative']:
    drawdown.append((max(prev, i) - i))
    prev = max(prev, i)
df['Drawdown'] = drawdown
max_drawdown = np.around((np.nanmax(df['Drawdown'].values*100)), decimals=2)
print('Max Drawdown:\n\t', max_drawdown, '%')
'''
Plotting two different plots in one figure.
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

Strategy, = ax1.plot(df.index, df['Cumulative'], label='Strategy', color='g', marker='.')
Close, = ax2.plot(df.index, df['Close'], label='USD Close', color='b', marker='.')

curves = [Strategy, Close]
ax1.legend(curves, [curve.get_label() for curve in curves])
ax1.set_xlabel('Date', color='k')
ax1.set_ylabel('Strategy (%age)', color=Strategy.get_color())
ax2.set_ylabel('Close', color=Close.get_color())

ax1.tick_params(axis='y', colors=Strategy.get_color())
ax2.tick_params(axis='y', colors=Close.get_color())

plt.title('USDINR Equity Curve')
ax1.grid()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()
'''
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

ax1.plot(df.index, df['Cumulative'], 'b', marker='', label='USDINR')
ax2.plot(df.index, df['Close'], 'k', marker='', label='Close')

ax1.legend()
ax1.set_title('Equity Curve USDINR')
ax1.set_ylabel('Strategy %age returns')

ax2.legend()
ax2.set_xlabel('Date')
ax2.set_ylabel('USDINR %age returns')

plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()
