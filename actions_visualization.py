import glob
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

"""
read .csv files containing actions divided by strategy, compute the mean and std for each action,
and plot the results.
"""

csv_files = glob.glob("csv/j*_granular_actions_0.csv")

dfs = [pd.read_csv(file, index_col=0) for file in csv_files]

df = sum(dfs) / len(dfs)

df_grouped_mean = df.groupby(df.index // 10).mean()
df_grouped_std = df.groupby(df.index // 10).std()


rename_map = {
    "I_NNN_1": "Unconditionally Cooperate - Cooperate",
    "I_NNN_0": "Unconditionally Cooperate - Defect",
    "I_NNN_None": "Unconditionally Cooperate - Abstain",
    "XII_NNN_1": "Q-Learning - Cooperate",
    "XII_NNN_0": "Q-Learning - Defect",
    "XII_NNN_None": "Q-Learning - Abstain",
    "II_NNN_1": "Unconditionally Defect - Cooperate",
    "II_NNN_0": "Unconditionally Defect - Defect",
    "II_NNN_None": "Unconditionally Defect - Abstain"
}

x = df_grouped_mean.index * 10  

window = 100

#df_grouped_mean = df_grouped_mean.rename(columns=rename_map)
#df_grouped_std = df_grouped_std.rename(columns=rename_map)

df_colors = ['maroon', 'blue', 'green', 'red', 'slategray', 'indigo']

plt.figure(figsize=(15, 6))

for col, color in zip(df_grouped_mean.columns, cycle(df_colors)):
    rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[col].rolling(window=window, min_periods=1).std()

    plt.plot(df.index, rolling_mean, label=col, color=color)
    plt.fill_between(df.index, rolling_mean - rolling_std, rolling_mean + rolling_std, color=color, alpha=0.2)

plt.title(f'Chosen Actions Over Time by strategy (window={window})')
plt.xlabel('Timestep')
plt.ylabel('# of agents per action (Moving Avg ± Std)')
plt.legend(title="Action per Strategy", 
           loc='center left',
            bbox_to_anchor=(1.02, 0.5), 
            borderaxespad=0,
            frameon=True)
plt.grid(True)
plt.tight_layout(rect=[0.01, 0, 1, 1])
plt.savefig('granular_actions_plot.png', dpi=300)