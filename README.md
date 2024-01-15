# CMO-Indicator
The Chande Momentum Indicator (CMO) is a technical analysis tool created by Tushar Chande to gauge the momentum of a security's price. Diverging from other momentum indicators such as the Relative Strength Index (RSI), the CMO considers both upward and downward price movements in its calculation, resulting in a more comprehensive assessment of price momentum.

With a range oscillating between -100 and +100, the CMO offers insights into potential overbought and oversold conditions. Typically, readings surpassing +50 suggest overbought circumstances, hinting at a potential price decline. Conversely, readings below -50 indicate oversold conditions, suggesting a potential price increase. Traders often combine the CMO with other indicators to enhance the robustness of their trading signals.


Strategy
Long Entry

Slope of the long-term market MA is positive (long-term market trend filter)

Slope of the long-term stock close MA is positive (long-term trend filter)

CMO crosses negative threshold from below (momentum signal)

VROC with lookback period is positive (volume confirmation)

Long Exit

CMO crosses positive threshold from below (overbought signal)

Trailing stop (risk management)

Stop loss (risk management).

Game Plan
Fetch NASDAQ 100 symbols

For each symbol: Fetch hourly prices for 5 years and

Either perform strategy optimization or perform backtest

Collect results for backtest or optimization

Plot new maximum or minimum backtest results

Print final results for all stocks.
