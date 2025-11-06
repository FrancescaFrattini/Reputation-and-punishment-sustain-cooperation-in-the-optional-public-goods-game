import pandas as pd
import matplotlib.pyplot as plt
import glob
import ast

def load_and_expand(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, keep_default_na=False)
    df.columns = ["1", "0", "None"]

    df = df.applymap(ast.literal_eval)

    out = pd.DataFrame(index=df.index)

    want = {
        "1": ["0", "None"],     
        "0": ["1", "None"],     
        "None": ["1", "0"],     
        }

    for frm, tos in want.items():
        for to in tos:
            def getter(d):
                key = int(to) if to.isdigit() else None
                return d.get(key, 0)
            out[f"{frm}->{to}"] = df[frm].apply(getter)

    return out

transition_colors = {
    "1->0": "tab:blue",
    "1->None": "tab:orange",
    "0->1": "tab:green",
    "0->None": "tab:red",
    "None->1": "tab:purple",
    "None->0": "tab:cyan",
}

labels = {
    "1->0": "Cooperate → Defect",
    "1->None": "Cooperate → Loner",
    "0->1": "Defect → Cooperate",
    "0->None": "Defect → Loner",
    "None->1": "Loner → Cooperate",
    "None->0": "Loner → Defect",
}

files = glob.glob("csv/j*_transitions_per_timestep_0.csv")
dfs = [load_and_expand(f) for f in files]

df_sum = sum(dfs) / len(dfs)

window = 50
df_roll = df_sum.rolling(window=window, min_periods=1).mean()

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

groups = {
    "From Cooperate": ["1->0", "1->None"],
    "From Defect": ["0->1", "0->None"],
    "From Abstain": ["None->1", "None->0"],
}

fig.supxlabel("Round")

for ax, (title, cols) in zip(axes, groups.items()):
    for col in cols:
        ax.plot(df_roll.index, df_roll[col], label=labels[col], color=transition_colors[col])
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_title(title, loc="left")
    ax.set_ylabel(f"# of transitions (rolling mean, w={window})")
    ax.tick_params(labelbottom=True)
    ax.legend()

plt.suptitle(f"Transitions per Timestep grouped by source action (window = {window})", fontsize=14)
plt.tight_layout()
plt.suptitle("Number of transitions per timestep from each action", fontsize=14, y=0.99)
plt.savefig("transitions_subplot.png") 
