from collections import defaultdict
import glob
import ast
import math
import pandas as pd
import matplotlib.pyplot as plt

files = sorted(glob.glob("csv/j*_q_values_rankings_0.csv"))

dfs = [pd.read_csv(f, index_col=0).map(ast.literal_eval) for f in files]

all_cols = sorted(set().union(*[df.columns for df in dfs]))

mean_df = pd.DataFrame(index=dfs[0].index, columns=all_cols)

for col in all_cols:
    for idx in mean_df.index:
        accumulator = defaultdict(float)
        count = defaultdict(int)

        for df in dfs:
            if col in df.columns:
                d = df.at[idx, col]
                if isinstance(d, dict):
                    for a, v in d.items():
                        accumulator[a] += v
                        count[a] += 1

        if count:
            mean_df.at[idx, col] = {a: accumulator[a] / count[a] for a in [0, 1, 2]}
        else:
            mean_df.at[idx, col] = {0: 0, 1: 0, 2: 0}

window = 50
smoothed_df = pd.DataFrame(index=mean_df.index, columns=all_cols)

for col in all_cols:
    series = {a: [] for a in [0, 1, 2]}

    for idx in mean_df.index:
        d = mean_df.at[idx, col]
        if d is None:
            d = {0: 0, 1: 0, 2: 0}
        for a in [0, 1, 2]:
            series[a].append(d.get(a, 0))
    smoothed = {a: pd.Series(values).rolling(window=window, min_periods=1).mean().tolist()
                for a, values in series.items()}
    for i, idx in enumerate(mean_df.index):
        smoothed_df.at[idx, col] = {a: smoothed[a][i] for a in [0, 1, 2]}

rows = []
action_labels = {0: "Defect", 1: "Cooperate", 2: "Loner"}

for period, row in smoothed_df.iterrows():
    for payoff, d in row.items():
        for action, count in d.items():
            rows.append({
                "period": period,
                "avg_payoff": float(payoff),
                "action_id": action,
                "action": action_labels[action],
                "count": count
            })

long_df = pd.DataFrame(rows)

action_colors = {0: "tab:orange", 1: "tab:green", 2: "tab:blue"}

avgs = sorted(long_df["avg_payoff"].unique())
ncols = 4
nrows = math.ceil(len(avgs) / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3), sharex=True, sharey=True)

for ax, avg in zip(axes.flat, avgs):
    sub = long_df[long_df["avg_payoff"] == avg]
    pivot = sub.pivot(index="period", columns="action_id", values="count")
    pivot = pivot.rename(columns=action_labels)
    pivot.plot(
        ax=ax,
        color=[action_colors[k] for k in sorted(action_labels.keys())]
    )
    ax.set_title(f"average payoff = {avg}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Agents' count")
    ax.legend(title="Action")
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.tick_params(labelbottom=True)

plt.suptitle(
    f"Q-Learners' Q-values rankings over time (subplots by average payoff) window = {window}",
    fontsize=14
)
plt.tight_layout()
plt.savefig("q_values_subplot.png", dpi=300)
