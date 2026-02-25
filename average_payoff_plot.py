import glob
import pandas as pd
import matplotlib.pyplot as plt


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
}

columns_to_plot = [
    "XII_VII_NNN_1",
    "XII_VII_NNN_0",
    "XII_VII_NNN_None"
]

csv_files = glob.glob("csv/j*_payoffs_0.csv")

dfs = []

for file in csv_files:
    df = pd.read_csv(file, index_col=0)
    dfs.append(df[columns_to_plot])

df = pd.concat(dfs).groupby(level=0).mean()

window = 50

plt.figure(figsize=(10, 6))

for col in df.columns:
    label = rename_map.get(col, col)

    rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[col].rolling(window=window, min_periods=1).std()
    plt.plot(df.index, rolling_mean, label=label)
    plt.fill_between(
        df.index,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.2
    )

plt.xlabel("Round")
plt.ylabel("Average Payoff (Moving Avg ± Std)")
plt.title(f"Average Payoffs per Action (window={window})")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(title="Action")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig('average_payoff_by_actionL(1, D).png', dpi=300)