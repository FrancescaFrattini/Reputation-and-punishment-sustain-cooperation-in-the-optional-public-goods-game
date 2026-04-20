import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

"""
read .csv files containing actions divided by strategy, compute the mean and std for each action,
and plot the results.
"""

rename_map = {
    'IV_NNN_1': "IV Strategy Agents - Good Reputation",
    'V_NNN_1': "V Strategy Agents - Good Reputation",
    'IV_NNN_0': "IV Strategy Agents - Not bad Reputation",
    'V_NNN_0': "V Strategy Agents - Not bad Reputation",
    'V_NNN_-1': "V Strategy Agents - Bad Reputation",
    'IV_NNN_-1': "IV Strategy Agents - Bad Reputation",
    'VI_NNN_1': "VI Strategy Agents - Good Reputation",
    'VII_NNN_1': "VII Strategy Agents - Good Reputation",
    'VI_NNN_0': "VI Strategy Agents - Not bad Reputation",
    'VII_NNN_0': "VII Strategy Agents - Not bad Reputation",
    'VI_NNN_-1': "VI Strategy Agents - Bad Reputation",
    'VII_NNN_-1': "VII Strategy Agents - Bad Reputation",
    'VIII_NNN_1': "VIII Strategy Agents - Good Reputation",
    'IX_NNN_1': "IX Strategy Agents - Good Reputation",
    'VIII_NNN_0': "VIII Strategy Agents - Not bad Reputation",
    'IX_NNN_0': "IX Strategy Agents - Not bad Reputation",
    'VIII_NNN_-1': "VIII Strategy Agents - Bad Reputation",
    'IX_NNN_-1': "IX Strategy Agents - Bad Reputation",
    'X_NNN_1': "X Strategy Agents - Good Reputation",
    'XI_NNN_1': "XI Strategy Agents - Good Reputation",
    'X_NNN_0': "X Strategy Agents - Not bad Reputation",
    'XI_NNN_0': "XI Strategy Agents - Not bad Reputation",
    'X_NNN_-1': "X Strategy Agents - Bad Reputation",
    'XI_NNN_-1': "XI Strategy Agents - Bad Reputation",
    'XII_V_NNN_-1': "Q-Learner with C^(1, D) - Bad reputation",
    'XII_IV_NNN_1': "Q-Learner with C^(0, D) - Good Reputation",
    'XII_V_NNN_1': "Q-Learner with C^(1, D) - Good Reputation",
    'XII_IV_NNN_-1': "Q-Learner with C^(0, D) - Bad Reputation",
    'XII_V_NNN_0': "Q-Learner with C^(1, D) - Not bad Reputation",
    'XII_IV_NNN_0': "Q-Learner with C^(0, D) - Not bad Reputation",
    'XII_VI_NNN_1': "Q-Learner with C^(0, L) - Good Reputation",
    'XII_VII_NNN_1': "Q-Learner with C^(1, L) - Good Reputation",
    'XII_VI_NNN_0': "Q-Learner with C^(0, L) - Not bad Reputation",
    'XII_VII_NNN_0': "Q-Learner with C^(1, L) - Not bad Reputation",
    'XII_VI_NNN_-1': "Q-Learner with C^(0, L) - Bad Reputation",
    'XII_VII_NNN_-1': "Q-Learner with C^(1, L) - Bad Reputation",
    'XII_VIII_NNN_1': "Q-Learner with D^(0, L) - Good Reputation",
    'XII_IX_NNN_1': "Q-Learner with D^(1, L) - Good Reputation",
    'XII_VIII_NNN_0': "Q-Learner with D^(0, L) - Not bad Reputation",
    'XII_IX_NNN_0': "Q-Learner with D^(1, L) - Not bad Reputation",
    'XII_VIII_NNN_-1': "Q-Learner with D^(0, L) - Bad Reputation",
    'XII_IX_NNN_-1': "Q-Learner with D^(1, L) - Bad Reputation",
    'XII_X_NNN_1': "Q-Learner with L^(0, D) - Good Reputation",
    'XII_XI_NNN_1': "Q-Learner with L^(1, D) - Good Reputation",
    'XII_X_NNN_0': "Q-Learner with L^(0, D) - Not bad Reputation",
    'XII_XI_NNN_0': "Q-Learner with L^(1, D) - Not bad Reputation",
    'XII_X_NNN_-1': "Q-Learner with L^(0, D) - Bad Reputation",
    'XII_XI_NNN_-1': "Q-Learner with L^(1, D) - Bad Reputation",
}

columns_to_plot = [
    "XII_IV_NNN_-1",
    "XII_IV_NNN_1",
    "XII_IV_NNN_0",
    'IV_NNN_0',
    'IV_NNN_-1',
    'IV_NNN_1',
]

dfs = []

csv_files = glob.glob("csv/j*_reputations_*.csv")

for file in csv_files:
    df = pd.read_csv(file, index_col=0)
    df = df.fillna(0)
    cols = sorted(df.columns.intersection(columns_to_plot))
    dfs.append(df[cols])

df = pd.concat(dfs).groupby(level=0).mean()

df.rename(columns=rename_map, inplace=True)
df = df.loc[:, (df != 0).any()]
window = 50

num_cols = len(df.columns)
colors = plt.get_cmap("tab10").colors
df_linestyle = ['--', '-.', ':', '-']

plt.figure(figsize=(15, 6))

for col, color, linestyle in zip(df.columns, colors, cycle(df_linestyle)):
    rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[col].rolling(window=window, min_periods=1).std()

    plt.plot(df.index, rolling_mean, label=col, color=color, linestyle=linestyle)
    plt.fill_between(df.index, rolling_mean - rolling_std, rolling_mean + rolling_std, color=color, alpha=0.2)

plt.title(f'Reputations over Time by strategy (window={window})')
plt.xlabel('Round')
plt.ylabel('# of agents per reputation type (Moving Avg ± Std)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(title="Reputation per Strategy")
plt.grid(True)
plt.tight_layout(rect=[0.01, 0, 1, 1])
plt.savefig('reputations_plotC(0, D).png', dpi=300)