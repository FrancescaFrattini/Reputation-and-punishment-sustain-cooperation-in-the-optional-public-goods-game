import glob
from matplotlib import pyplot as plt
import pandas as pd

csv_files = glob.glob("csv/j?_actions_0.csv")

dfs = [pd.read_csv(f, index_col=0) for f in csv_files]

df_mean = sum(dfs) / len(dfs)

rename_map = {
    "Cooperative": "Cooperate",
    "Non-Cooperative": "Defect",
    "Loner": "Loner"
}
df_mean.rename(columns=rename_map, inplace=True)

ax = df_mean.plot(
    kind="bar",
    stacked=True,
    color=["tab:green", "tab:blue", "tab:orange"],
    figsize=(10, 6)
)

ax.set_xlabel("Timestep")
ax.set_ylabel("Numero di agenti (media)")
ax.set_title("Distribuzione media delle azioni per timestep")
plt.xticks(rotation=0)
plt.legend(title="Azione")
plt.tight_layout()
plt.savefig("stacked_bar_mean.png", dpi=300)