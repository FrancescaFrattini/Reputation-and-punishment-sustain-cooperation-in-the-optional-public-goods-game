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
    'XII_IV_NNN_1': "Q-Learner with C^(0, D) - Cooperate",
    'XII_IV_NNN_0': "Q-Learner with C^(0, D) - Defect",
    'XII_IV_NNN_None': "Q-Learner with C^(0, D) - Loner",
    'XII_V_NNN_1': "Q-Learner with C^(1, D) - Cooperate",
    'XII_V_NNN_0': "Q-Learner with C^(1, D) - Defect",
    'XII_V_NNN_None': "Q-Learner with C^(1, D) - Loner",
    'XII_VI_NNN_1': "Q-Learner with C^(0, L) - Cooperate",
    'XII_VI_NNN_0': "Q-Learner with C^(0, L) - Defect",
    'XII_VI_NNN_None': "Q-Learner with C^(0, L) - Loner",
    'XII_VII_NNN_1': "Q-Learner with C^(1, L) - Cooperate",
    'XII_VII_NNN_0': "Q-Learner with C^(1, L) - Defect",
    'XII_VII_NNN_None': "Q-Learner with C^(1, L) - Loner",
    'XII_VIII_NNN_1': "Q-Learner with D^(0, L) - Cooperate",
    'XII_VIII_NNN_0': "Q-Learner with D^(0, L) - Defect",
    'XII_VIII_NNN_None': "Q-Learner with D^(0, L) - Loner",
    'XII_IX_NNN_1': "Q-Learner with D^(1, L) - Cooperate",
    'XII_IX_NNN_0': "Q-Learner with D^(1, L) - Defect",
    'XII_IX_NNN_None': "Q-Learner with D^(1, L) - Loner",
    'XII_X_NNN_1': "Q-Learner with L^(0, D) - Cooperate",
    'XII_X_NNN_0': "Q-Learner with L^(0, D) - Defect",
    'XII_X_NNN_None': "Q-Learner with L^(0, D) - Loner",
    'XII_XI_NNN_1': "Q-Learner with L^(1, D) - Cooperate",
    'XII_XI_NNN_0': "Q-Learner with L^(1, D) - Defect",
    'XII_XI_NNN_None': "Q-Learner with L^(1, D) - Loner",
}

columns_to_plot = [
    "XII_XI_NNN_1",
    "XII_XI_NNN_0",
    "XII_XI_NNN_None"
]

dfs = []

csv_files = glob.glob("csv/j*_granular_actions_*.csv")

for file in csv_files:
    df = pd.read_csv(file, index_col=0)
    dfs.append(df[columns_to_plot])

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

plt.title(f'Chosen Actions Over Time by strategy (window={window})')
plt.xlabel('Round')
plt.ylabel('# of agents per action (Moving Avg ± Std)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(title="Action per Strategy")
plt.grid(True)
plt.tight_layout(rect=[0.01, 0, 1, 1])
plt.savefig('granular_actions_plotL(1, D).png', dpi=300)