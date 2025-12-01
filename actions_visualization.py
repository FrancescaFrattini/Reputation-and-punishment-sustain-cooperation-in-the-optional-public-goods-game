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
    'XII_I_NNN_1': "Q-Learner with Cooperators - Cooperate",
    'XII_I_NNN_0': "Q-Learner with Cooperators - Defect",
    'XII_I_NNN_None': "Q-Learner with Cooperators - Loner",
    'XII_II_NNN_1': "Q-Learner with Defectors - Cooperate",
    'XII_II_NNN_0': "Q-Learner with Defectors - Defect",
    'XII_II_NNN_None': "Q-Learner with Defectors - Loner",
    'XII_III_NNN_1': "Q-Learner with Loners - Cooperate",
    'XII_III_NNN_0': "Q-Learner with Loners - Defect",
    'XII_III_NNN_None': "Q-Learner with Loners - Loner",
    "I_NNN_1": "Cooperators only - Cooperate",
    "II_NNN_0": "Defector only - Defect",
    "III_NNN_None": "Loner only - Loner"
}

csv_files = glob.glob("csv/j*_granular_actions_*.csv")

dfs = [pd.read_csv(file, index_col=0) for file in csv_files]

df = sum(dfs) / len(dfs)

df.rename(columns=rename_map, inplace=True)
df = df.loc[:, (df != 0).any()]
window = 1

num_cols = len(df.columns)
colors = plt.cm.Dark2(np.linspace(0, 1, num_cols))
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
plt.savefig('granular_actions_plot.png', dpi=300)