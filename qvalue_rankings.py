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

#mean_df = dfs[0].copy()
for i in mean_df.index:
    for j in all_cols:
        combined = defaultdict(list)
        for df in dfs:
            if j in df.columns:
                d = df.at[i, j]
                for k, v in d.items():
                    combined[k].append(v)
        mean_df.at[i, j] = {k: sum(v)/len(v) 
            for k, v in combined.items()} if combined else None

smoothed_df = mean_df.copy()

window = 30

for col in mean_df.columns: 
    if mean_df[col].isna().all():
        continue
    action_series = {a: [] for a in [0, 1, 2]}
    for i in range(len(mean_df)):
        d = mean_df.at[i, col]
        if d is None:
            for a in [0, 1, 2]:
                action_series[a].append(d.get(a, 0))
        else:
            for a in [0, 1, 2]:
                action_series[a].append(0)


smoothed = {a: pd.Series(v).rolling(window=window, min_periods=1).mean().tolist()
            for a, v in action_series.items()}

for i in range(len(mean_df)):
    smoothed_df.at[i, col] = {a: smoothed[a][i] for a in [0, 1, 2]}
        #smoothed_df.at[i, col] = {a: smoothed[a][i] for a in [0, 1, 2]}

action_labels = {0: "Defect", 1: "Cooperate", 2: "Loner"}
action_colors = {0: "tab:orange", 1: "tab:green", 2: "tab:blue"}

rows = []
for period, row in smoothed_df.iterrows():
    for avg_payoff, d in row.items():
        for ranking, count in d.items():
            rows.append({
                "period": period,
                "avg_payoff": float(avg_payoff),
                "ranking": ranking,
                "action": action_labels[ranking],
                "count": count
            })

long_df = pd.DataFrame(rows)

avgs = sorted(long_df["avg_payoff"].unique())
ncols = 4
nrows = math.ceil(len(avgs) / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*3), sharex=True, sharey=True)

for ax, avg in zip(axes.flat, avgs):
    sub = long_df[long_df["avg_payoff"] == avg]
    pivot = sub.pivot(index="period", columns="ranking", values="count")
    pivot = pivot.rename(columns=action_labels)
    pivot.plot(ax=ax, color=[action_colors[k] for k in sorted(action_labels.keys())])
    ax.set_title(f"average payoff = {avg}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Agents' count")
    ax.tick_params(labelbottom=True)
    ax.legend(title="Action")
    ax.autoscale(enable=True, axis='x', tight=True)

plt.suptitle(f"Q-Learners' Q-values rankings over time (subplots by average payoff) window = {window}", size = 14)
plt.tight_layout()
plt.savefig("q_values_subplot.png", dpi=300)
