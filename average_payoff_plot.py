import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("csv/j0_payoffs_0.csv")

rename_map = {
    "XII_NNN_None": "Q-Learner - Loner",
    "XII_NNN_0": "Q-Learner - Defector",
    "XII_NNN_1": "Q-Learner - Cooperator"
}

#rolling window for mean
window = 2

plt.figure(figsize=(10, 6))

for col in df.columns[1:]:
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

plt.xlabel("Timestep")
plt.ylabel("Average Payoff (Moving Avg ± Std)")
plt.title(f"Average Payoffs per Action (window={window})")
plt.legend(title="Action")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig('average_payoff_by_action.png', dpi=300)