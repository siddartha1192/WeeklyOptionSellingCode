FileName- 
## Nifty SuperTrend Option Selling Strategy

**Automated Options Spread Trading Bot** for NSE NIFTY50 using Fyers API

### Key Features
- **Strategy**: SuperTrend + EMA based option spread selling
- **Instruments**: NIFTY50 index options (CE/PE spreads)
- **Entry Signals**: 
  - Long Put Spread: Green SuperTrend + Price above EMA
  - Short Call Spread: Red SuperTrend + Price below EMA
- **Delta-Based Selection**: 
  - Sells options at ~0.4 delta, buys protection at ~0.25 delta
  - Automatic strike selection using Black-Scholes delta calculation
- **Expiry Management**: 
  - Monday-Thursday: Uses next week expiry
  - Friday: Uses current week expiry
  - Auto-exit 16 minutes before expiry
- **Trading Modes**: Paper trading and live trading support

### Technical Indicators
- SuperTrend (configurable period & multiplier)
- Exponential Moving Average (EMA)
- Black-Scholes delta calculation for option selection
- Implied volatility estimation

### Risk Management
- Credit spread strategies (limited risk)
- Signal-based exits on SuperTrend reversal
- Time-based exits before expiry
- Comprehensive trade logging and state persistence

*Designed for systematic option income generation with defined risk parameters.*
