import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime,timedelta

def get_us_nominal_yield():
    # gives us 10 year treasury note yield 
    us_nominal_yield_ticker = yf.Ticker("^TNX")
    end = datetime.today()
    start = end - timedelta(days=60)
    us_nominal_yield_data = us_nominal_yield_ticker.history(start=start, end=end)
    if us_nominal_yield_data.empty:
        print("Could not fetch 10 year treas180ury note yield. Please check the ticker or internet connection.")
        return pd.DataFrame()
    us_nominal_yield_data = us_nominal_yield_data.reset_index()
    us_nominal_yield_data['Date'] = pd.to_datetime(us_nominal_yield_data['Date']).dt.date
    us_nominal_yield_data = us_nominal_yield_data[['Date', 'Close']]
    us_nominal_yield_data.rename(columns={'Close': 'US_Nominal_Yield'}, inplace=True)
    return us_nominal_yield_data  

def get_us_inflation_expectation(): # Inflation Expectation
    end = datetime.today()
    start = end - timedelta(days=180)

    infl_data = web.DataReader("T10YIE", "fred", start, end)
    infl_data = infl_data.reset_index()
    infl_data.rename(columns={"T10YIE": "Infl_Expectation"}, inplace=True)
    infl_data['Date'] = infl_data['DATE'].dt.date
    infl_data = infl_data[['Date','Infl_Expectation']]
    return infl_data

def get_xau_data(): # Gold
    xau_ticker = yf.Ticker("GC=F")
    end = datetime.today()
    start = end - timedelta(days=180)
    xau_data = xau_ticker.history(start=start, end=end)

    if xau_data.empty:
        print("Could not fetch XAU data. Please check the ticker or internet connection.")
        return pd.DataFrame()

    xau_data = xau_data.reset_index()
    xau_data['Date'] = pd.to_datetime(xau_data['Date']).dt.date
    xau_data = xau_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    xau_data.rename(columns={'Close': 'XAU_Close'}, inplace=True)
    return xau_data

def get_dxy_data(): # Dollar Index

    dxy_ticker = yf.Ticker("DX=F") 
    end = datetime.today()
    start = end - timedelta(days=60)
    dxy_data = dxy_ticker.history(start=start, end=end) 

    if dxy_data.empty:
        print("Could not fetch DXY data. Please check the ticker or internet connection.")
        return pd.DataFrame()

    dxy_data = dxy_data.reset_index()
    dxy_data['Date'] = pd.to_datetime(dxy_data['Date']).dt.date
    dxy_data = dxy_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    dxy_data.rename(columns={'Close': 'DXY_Close'}, inplace=True)  
    return dxy_data

def get_vix_data(): # VIX
    vix_ticker = yf.Ticker("^VIX")
    end = datetime.today()
    start = end - timedelta(days=60)
    vix_data = vix_ticker.history(start=start, end=end) 

    if vix_data.empty:
        print("Could not fetch VIX data. Please check the ticker or internet connection.")
        return pd.DataFrame()

    vix_data = vix_data.reset_index()
    vix_data['Date'] = pd.to_datetime(vix_data['Date']).dt.date
    vix_data = vix_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    vix_data.rename(columns={'Close': 'VIX_Close'}, inplace=True)  
    return vix_data

def get_spx_data(): # S&P 500
    spx_ticker = yf.Ticker("^GSPC")   
    end = datetime.today()
    start = end - timedelta(days=60)
    spx_data = spx_ticker.history(start=start, end=end)

    if spx_data.empty:
        print("Could not fetch SPX data. Please check the ticker or internet connection.")
        return pd.DataFrame()

    spx_data = spx_data.reset_index()
    spx_data['Date'] = pd.to_datetime(spx_data['Date']).dt.date
    spx_data = spx_data[['Date', 'Close']]
    spx_data.rename(columns={'Close': 'SPX_Close'}, inplace=True)
    return spx_data

def get_oil_data(): # Crude Oil
    oil_ticker = yf.Ticker("CL=F")   
    end = datetime.today()
    start = end - timedelta(days=60)
    oil_data = oil_ticker.history(start=start, end=end)

    if oil_data.empty:
        print("Could not fetch Oil data. Please check the ticker or internet connection.")
        return pd.DataFrame()

    oil_data = oil_data.reset_index()
    oil_data['Date'] = pd.to_datetime(oil_data['Date']).dt.date
    oil_data = oil_data[['Date', 'Close']]
    oil_data.rename(columns={'Close': 'OIL_Close'}, inplace=True)
    return oil_data

def main_data():
    xau = get_xau_data()
    dxy = get_dxy_data()    
    vix_data = get_vix_data()
    us_nominal_yield_data = get_us_nominal_yield() 
    us_infl_expectation_data = get_us_inflation_expectation()
    spx_data = get_spx_data()
    oil_data = get_oil_data()
    merged_data_xau_dxy = pd.merge(xau, dxy, on='Date', how='inner')
    merged_data = pd.merge(merged_data_xau_dxy, us_nominal_yield_data, on='Date', how='inner')
    merged_data = pd.merge(merged_data, us_infl_expectation_data, on='Date', how='inner') 
    merged_data = pd.merge(merged_data, vix_data, on='Date', how='inner')
    merged_data = pd.merge(merged_data, spx_data, on='Date', how='inner')
    merged_data = pd.merge(merged_data, oil_data, on='Date', how='inner')
    merged_data['Real_Yield'] = merged_data['US_Nominal_Yield'] - merged_data['Infl_Expectation']
    merged_data = merged_data.sort_values('Date')

    # Rolling correlations
    merged_data['oil_corr_curr'] = merged_data['XAU_Close'].corr(merged_data['OIL_Close'])
    merged_data['oil_corr_7d'] = merged_data['XAU_Close'].rolling(7).corr(merged_data['OIL_Close'])
    merged_data['oil_corr_30d'] = merged_data['XAU_Close'].rolling(30).corr(merged_data['OIL_Close'])
    merged_data['spx_corr_curr'] = merged_data['XAU_Close'].corr(merged_data['SPX_Close'])
    merged_data['spx_corr_7d'] = merged_data['XAU_Close'].rolling(7).corr(merged_data['SPX_Close'])
    merged_data['spx_corr_30d'] = merged_data['XAU_Close'].rolling(30).corr(merged_data['SPX_Close'])
    merged_data['vix_corr_curr'] = merged_data['XAU_Close'].corr(merged_data['VIX_Close'])
    merged_data['vix_corr_7d'] = merged_data['XAU_Close'].rolling(7).corr(merged_data['VIX_Close'])
    merged_data['vix_corr_30d'] = merged_data['XAU_Close'].rolling(30).corr(merged_data['VIX_Close'])
    merged_data['nominal_yield_curr'] = merged_data['XAU_Close'].corr(merged_data['US_Nominal_Yield'])
    merged_data['nominal_yield_corr_7d'] = merged_data['XAU_Close'].rolling(7).corr(merged_data['US_Nominal_Yield'])
    merged_data['nominal_yield_corr_30d'] = merged_data['XAU_Close'].rolling(30).corr(merged_data['US_Nominal_Yield'])
    merged_data['real_yield_curr'] = merged_data['XAU_Close'].corr(merged_data['Real_Yield'])
    merged_data['real_yield_corr_7d'] = merged_data['XAU_Close'].rolling(7).corr(merged_data['Real_Yield'])
    merged_data['real_yield_corr_30d'] = merged_data['XAU_Close'].rolling(30).corr(merged_data['Real_Yield'])
    merged_data['dxy_curr'] = merged_data['XAU_Close'].corr(merged_data['DXY_Close'])
    merged_data['dxy_corr_7d'] = merged_data['XAU_Close'].rolling(7).corr(merged_data['DXY_Close'])
    merged_data['dxy_corr_30d'] = merged_data['XAU_Close'].rolling(30).corr(merged_data['DXY_Close'])
    merged_data = merged_data[['Date', 'XAU_Close','OIL_Close','SPX_Close','VIX_Close','US_Nominal_Yield','Real_Yield','DXY_Close','oil_corr_curr', 'oil_corr_7d', 'oil_corr_30d', 'spx_corr_curr', 'spx_corr_7d', 'spx_corr_30d', 'nominal_yield_curr', 'nominal_yield_corr_7d', 'nominal_yield_corr_30d', 'real_yield_curr', 'real_yield_corr_7d', 'real_yield_corr_30d', 'dxy_curr', 'dxy_corr_7d', 'dxy_corr_30d','vix_corr_curr','vix_corr_7d','vix_corr_30d']]
    merged_data = merged_data.sort_values('Date',ascending=False)    
    return merged_data

def convert_signal_to_score(signal):
    if signal == "BUY":
        return +1
    elif signal == "SELL":
        return -1
    else:
        return 0

if __name__ == '__main__':
    data = main_data()
    #data.to_csv('backtest\\xau_macro_driver.csv', index=False)
    #data = pd.read_csv(r'backtest\xau_macro_driver.csv')
    #data = data[data['Date']< '2025-11-21']
    print(f"Data last updated on: {data['Date'].max()}")
    data_last_7 = data.head(7)

    # Compute average correlation for each macro driver over last 7 days
    macro_corr = {
        "Nominal_Yield": data_last_7["nominal_yield_corr_7d"].mean(),
        "Real_Yield": data_last_7["real_yield_corr_7d"].mean(),
        "DXY": data_last_7["dxy_corr_7d"].mean(),
        "VIX": data_last_7["vix_corr_7d"].mean(),
        "SPX": data_last_7["spx_corr_7d"].mean(),
        "OIL": data_last_7["oil_corr_7d"].mean()
    }

    # Convert to DataFrame
    macro_corr_df = pd.DataFrame.from_dict(macro_corr, orient='index', columns=['Correlation'])

    # Add absolute value column for ranking
    macro_corr_df["Abs_Corr"] = macro_corr_df["Correlation"].abs()

    # Sort by strongest absolute correlation
    macro_corr_df = macro_corr_df.sort_values("Abs_Corr", ascending=False)

    print("\n--- MACRO DRIVER RANKING (Last 7 Days) ---")
    print(macro_corr_df)
    # Pick the strongest driver
    top_driver = macro_corr_df.index[0]
    top_corr   = macro_corr_df.loc[top_driver, "Correlation"]
    second_driver = macro_corr_df.index[1]
    second_corr   = macro_corr_df.loc[second_driver, "Correlation"] 

    # Map driver name to its column in data
    driver_col_map = {
        "Real_Yield":    "Real_Yield",
        "Nominal_Yield": "US_Nominal_Yield",
        "DXY":           "DXY_Close",
        "VIX":           "VIX_Close",
        "SPX":           "SPX_Close",
        "OIL":           "OIL_Close",
    }

    driver_col = driver_col_map[top_driver]
    driver_col_2 = driver_col_map[second_driver]

    # Take yesterday and day-before for that driver
    yesterday     = data.iloc[0]   # most recent row
    day_before    = data.iloc[1]   # previous row

    driver_yest   = yesterday[driver_col]
    driver_prev   = day_before[driver_col]
    driver_yest_2   = yesterday[driver_col_2]
    driver_prev_2   = day_before[driver_col_2]

    driver_change = driver_yest - driver_prev
    driver_change_2 = driver_yest_2 - driver_prev_2

    print(f"\nTop macro driver: {top_driver}")
    print(f"Smoothed correlation (7d avg): {top_corr:.4f}")
    print(f"Yesterday {driver_col}: {driver_yest:.4f}, day before: {driver_prev:.4f}, change: {driver_change:.4f}")

    print(f"\nSecond macro driver: {second_driver}")
    print(f"Smoothed correlation (7d avg): {second_corr:.4f}")
    print(f"Yesterday {driver_col_2}: {driver_yest_2:.4f}, day before: {driver_prev_2:.4f}, change: {driver_change_2:.4f}")

    # Macro-only signal logic
    abs_top_corr = abs(top_corr)
    abs_second_corr = abs(second_corr)
    MIN_STRENGTH = 0.4  # ignore weak regimes

    if np.isnan(top_corr) or abs_top_corr < MIN_STRENGTH:
        top_signal = "NO_TRADE"
    else:
        if top_corr > 0:
            # moves WITH driver
            if driver_change > 0:
                top_signal = "BUY"
            elif driver_change < 0:
                top_signal = "SELL"
            else:
                top_signal = "NO_TRADE"
        else:
            # moves OPPOSITE driver
            if driver_change > 0:
                top_signal = "SELL"
            elif driver_change < 0:
                top_signal = "BUY"
            else:
                top_signal = "NO_TRADE"

    print(f"\n=== MACRO BIAS SIGNAL FOR TOP DRIVER: {top_signal} ===")

    top_score = convert_signal_to_score(top_signal)


    # ============================
    #   SIGNAL FOR SECOND DRIVER
    # ============================

    if np.isnan(second_corr) or abs_second_corr < MIN_STRENGTH:
        second_signal = "NO_TRADE"
    else:
        if second_corr > 0:
            if driver_change_2 > 0:
                second_signal = "BUY"
            elif driver_change_2 < 0:
                second_signal = "SELL"
            else:
                second_signal = "NO_TRADE"
        else:
            if driver_change_2 > 0:
                second_signal = "SELL"
            elif driver_change_2 < 0:
                second_signal = "BUY"
            else:
                second_signal = "NO_TRADE"

    print(f"\n=== MACRO BIAS SIGNAL FOR SECOND DRIVER: {second_signal} ===")

    second_score = convert_signal_to_score(second_signal)
    combined_raw = (0.7 * top_score) + (0.3 * second_score)

    # Convert to final signal
    if combined_raw >= 0.5:
        final_signal = "STRONG BUY"
    elif 0.1 <= combined_raw < 0.5:
        final_signal = "BUY (Weak)"
    elif -0.1 < combined_raw < 0.1:
        final_signal = "NEUTRAL"
    elif -0.5 < combined_raw <= -0.1:
        final_signal = "SELL (Weak)"
    else:
        final_signal = "STRONG SELL"

    # Convert raw score to 0–100 confidence
    confidence = int(abs(combined_raw) * 100)

    print("\n===============================")
    print("      FINAL MACRO SIGNAL")
    print("===============================\n")
    print(f"Signal: {final_signal}")
    print(f"Conviction Score: {confidence}/100")
    print(f"Top Driver: {top_driver} → {top_signal}")
    print(f"Second Driver: {second_driver} → {second_signal}")
    print(f"Weighted Score: {combined_raw:.2f}")
    print("\n===============================\n")