import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

label_map= {
    "1": "Cooperate",
    "0": "Defect", 
    "None": "Loner"
}

csv_files = glob.glob("csv/j*_transitions_0_9.csv")  
dfs = [pd.read_csv(file) for file in csv_files]

base = dfs[0][["Source", "Destination"]].copy()
base["#"] = pd.concat([df.iloc[:, -1] for df in dfs], axis=1).mean(axis=1, numeric_only=True)
df = base[["Source", "Destination", "#"]]
df["Source"] = df["Source"].map(label_map)
df["Destination"] = df["Destination"].map(label_map)

transition_matrix = df.pivot_table(
    index="Source",
    columns="Destination",
    values="#",
    aggfunc="sum",
    fill_value=0
)

transition_matrix = transition_matrix / transition_matrix.values.sum()

avg_freq = transition_matrix.sum(axis=0)

fig = plt.figure(figsize=(7, 9))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)

ax_bar = fig.add_subplot(gs[0])
sns.barplot(x=avg_freq.index, y=avg_freq.values, ax=ax_bar, hue=avg_freq.index, palette="husl", legend=False)
ax_bar.set_ylabel("Average Frequency", labelpad=20)
ax_bar.set_xlabel("")
ax_bar.set_xticks([]) 
ax_bar.set_title("Actions Transition Matrix")

ax_hm = fig.add_subplot(gs[1])
sns.heatmap(
    transition_matrix * 100,   
    annot=True,
    fmt=".1f",
    cmap="Greens",
    cbar=False,
    ax=ax_hm
)

ax_hm.set_xlabel("Target Strategy", labelpad=20)
ax_hm.set_ylabel("Source Strategy", labelpad=20)

plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.4)
plt.savefig("transition_matrix9.png", dpi=300)