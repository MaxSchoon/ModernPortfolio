import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define Treasury symbols (^TNX is 10-year, ^IRX is 13-week, ^FVX is 5-year, ^TYX is 30-year)
treasury_symbols = ['^IRX', '^FVX', '^TNX', '^TYX']
maturities = [0, 0.25, 5, 10, 30]  # in years (0 represents cash)

# Fetch Treasury data
yields = []
current_savings_rate = 4.35  # Current US savings rate (you may want to update this)
yields.append(current_savings_rate)

for symbol in treasury_symbols:
    df_data = yf.download(symbol, period='1d', interval='1d')
    current_yield = df_data['Close'].iloc[-1] if not df_data.empty else 0
    yields.append(current_yield)

# Create DataFrame
df = pd.DataFrame({
    'Maturity': maturities,
    'Yield': yields
})

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['Maturity'], df['Yield'], 'b-o', linewidth=2, markersize=8)
plt.title('US Treasury Yield Curve')
plt.xlabel('Time to Maturity (Years)')
plt.ylabel('Yield (%)')
plt.grid(True)
plt.show()
