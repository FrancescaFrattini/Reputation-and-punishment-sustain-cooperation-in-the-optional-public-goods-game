import pandas as pd
import matplotlib.pyplot as plt
import glob

csv_files = glob.glob("csv/j?_actions_0.csv")

dfs = [pd.read_csv(file, index_col=0) for file in csv_files]
df = pd.concat(dfs).groupby(level=0).mean()

target_cols = ["Cooperative", "Non-Cooperative", "Loner"]
rename_map = {
    "Cooperative": "Cooperate",
    "Non-Cooperative": "Defect",
    "Loner" : "Loner"
}

df = df[[col for col in df.columns if col in target_cols]]
df.rename(columns=rename_map, inplace=True)

window = 1

grouped = df.groupby(df.index // window).mean()

ax = grouped.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Paired")

ax.set_xlabel(f"Timestep")
ax.set_ylabel("Actions' Mean Distribution")
ax.set_title(f"Actions' Mean Distribution Every {window} Steps")
ax.set_xticks(range(len(grouped)))
ax.set_xticklabels([f'{i*window}-{(i+1)*window - 1}' for i in grouped.index], rotation=45)
plt.legend(title="Actions", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)

plt.savefig(f'barplot_{window}_steps_groups.png', dpi=300)