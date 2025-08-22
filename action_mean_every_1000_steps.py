import pandas as pd
import matplotlib.pyplot as plt
import glob

csv_files = glob.glob("csv/j*_actions_0.csv")

dfs = [pd.read_csv(file, index_col=0) for file in csv_files]
df = pd.concat(dfs).groupby(level=0).mean()

target_cols = ["Cooperative", "Non-Cooperative", "Loner"]
df = df[[col for col in df.columns if col in target_cols]]

grouped = df.groupby(df.index // 1000).mean()

ax = grouped.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Paired")

ax.set_xlabel("Timesteps (grouped every 1000)")
ax.set_ylabel("Actions' Mean Distribution")
ax.set_title("Actions' Mean Distribution Every 1000 Steps")
#plt.xticks(rotation=0)
ax.set_xticks(range(len(grouped)))
ax.set_xticklabels([f'{i*1000}-{(i+1)*1000 - 1}' for i in grouped.index], rotation=45)
plt.legend(title="Actions", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)

plt.savefig('barplot_100_steps_groups.png', dpi=300)