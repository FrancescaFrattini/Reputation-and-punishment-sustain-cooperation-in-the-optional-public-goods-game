import glob
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

"""
read .csv files containing actions divided by strategy, compute the mean and std for each action,
and plot the results.
"""

rename_map = {
    'XII_NNN_1': "Q-Learner - Cooperate",
    'XII_NNN_0': "Q-Learner - Defect",
    'XII_NNN_None': "Q-Learner - Loner",
    "I_NNN_1": "Cooperators only - Cooperate",
    "I_NNN_0": "Cooperators only - Defect",
    "I_NNN_None": "Cooperators only - Loner",
    "II_NNN_1": "Defector only - Cooperate",
    "II_NNN_0": "Defector only - Defect",
    "II_NNN_None": "Defector only - Loner",
    "III_NNN_1": "Loner only - Cooperate",
    "III_NNN_0": "Loner only - Defect",
    "III_NNN_None": "Loner only - Loner"
}


csv_files = glob.glob("csv/j*_granular_actions_0.csv")

dfs = [pd.read_csv(file, index_col=0) for file in csv_files]

df = sum(dfs) / len(dfs)

df.rename(columns=rename_map, inplace=True)

window = 20

df_colors = ['maroon', 'blue', 'green', 'red', 'slategray', 'indigo', 'purple', 'cyan']
df_linestyle = ['--', '-.', ':']

plt.figure(figsize=(15, 6))

for col, color, style in zip(df.columns, cycle(df_colors), cycle(df_linestyle)):
    rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[col].rolling(window=window, min_periods=1).std()

    plt.plot(df.index, rolling_mean, label=col, color=color, linestyle=style)
    plt.fill_between(df.index, rolling_mean - rolling_std, rolling_mean + rolling_std, color=color, alpha=0.2)

plt.title(f'Chosen Actions Over Time by strategy (window={window})')
plt.xlabel('Round')
plt.ylabel('# of agents per action (Moving Avg ± Std)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(title="Action per Strategy")
plt.grid(True)
plt.tight_layout(rect=[0.01, 0, 1, 1])
plt.savefig('granular_actions_plot.png', dpi=300)