import glob
import ast
import pandas as pd
import matplotlib.pyplot as plt

files = sorted(glob.glob("csv/j*_q_values_rankings_0.csv"))

window = 50

action_labels = {0: "Defect", 1: "Cooperate", 2: "Loner"}
action_colors = {0: "tab:blue", 1: "tab:green", 2: "tab:orange"}

def parse_cell_to_dict(x):
    try:
        d = ast.literal_eval(str(x))
    except Exception:
        d = {}
    return {0: d.get(0, 0), 1: d.get(1, 0), 2: d.get(2, 0)}

def load_and_parse_csv(path):
    df = pd.read_csv(path, index_col=0, keep_default_na=False)
    return df.applymap(parse_cell_to_dict)

def parse_triple(colname):
    try:
        t = ast.literal_eval(colname)
        return tuple(int(x) for x in t)
    except Exception:
        s = colname.strip().lstrip("(").rstrip(")").replace(" ", "")
        parts = s.split(",")
        if len(parts) == 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        raise ValueError(f"Cannot parse column: {colname}")

dfs = [load_and_parse_csv(f) for f in files]

shapes = {df.shape for df in dfs}

nfiles = len(dfs)
rows = dfs[0].index
cols = dfs[0].columns

df_sum = dfs[0].copy()
for r in rows:
    for c in cols:
        df_sum.at[r, c] = {0: 0, 1: 0, 2: 0}

for df in dfs:
    for r in rows:
        for c in cols:
            d_cur = df.at[r, c]
            d_acc = df_sum.at[r, c]
            df_sum.at[r, c] = {
                0: d_acc.get(0, 0) + d_cur.get(0, 0),
                1: d_acc.get(1, 0) + d_cur.get(1, 0),
                2: d_acc.get(2, 0) + d_cur.get(2, 0),
            }

df_mean = df_sum.copy()
for r in rows:
    for c in cols:
        d = df_sum.at[r, c]
        df_mean.at[r, c] = {k: float(v) / nfiles for k, v in d.items()}


group_0, group_1, group_2, group_mix = [], [], [], []

for c in cols:
    a, b, cc = parse_triple(c)
    if a >= 3:
        group_0.append(c)
    elif b >= 3:
        group_1.append(c)
    elif cc >= 3:
        group_2.append(c)
    else:
        group_mix.append(c)

groups = [
    ("Majority of group has chosen Defect ", group_0),
    ("Majority of group has chosen Cooperate ", group_1),
    ("Majority of group has chosen to Abstain ", group_2),
    ("Mixed choices", group_mix),
]

group_action_series = {} 

for title, group_cols in groups:
    if len(group_cols) == 0:
        for action in [0, 1, 2]:
            vals = pd.Series([0.0] * len(rows), index=rows)
            group_action_series[(title, action)] = vals.rolling(window=window, min_periods=1).mean()
        continue

    for action in [0, 1, 2]:
        df_action = pd.DataFrame({col: df_mean[col].apply(lambda d: d.get(action, 0.0)) for col in group_cols})
        mean_series = df_action.mean(axis=1)
        rolled = mean_series.rolling(window=window, min_periods=1).mean()
        group_action_series[(title, action)] = rolled


fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
fig.subplots_adjust(hspace=0.35)
fig.suptitle(f"Q-Values rankings for each Q-Table row", fontsize=16)

for ax, (title, _) in zip(axes, groups):
    ax.set_title(title, loc="left", fontsize=13)
    ax.set_ylabel("# agents", fontsize=10)
    for action in [0, 1, 2]:
        s = group_action_series[(title, action)]
        ax.plot(s.index, s.values, label=action_labels[action], color=action_colors[action], linewidth=2)
    ax.tick_params(labelbottom=True)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("# Round", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("q_values_subplot0.png", dpi=300)
