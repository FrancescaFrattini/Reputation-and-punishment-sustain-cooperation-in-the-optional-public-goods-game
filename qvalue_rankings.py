import pandas as pd
import matplotlib.pyplot as plt
import glob

csv_files = glob.glob("csv/j*_q_values_rankings_0.csv")

dfs = [pd.read_csv(file, index_col=0) for file in csv_files]

df = sum(dfs) / len(dfs)

action_map = {0: "Cooperate", 1: "Defect", 2: "Loner"}

def rename_ranking(ranking_str):
    ranking = tuple(int(x) for x in ranking_str.strip("()").split(","))
    return " > ".join([action_map[i] for i in ranking])

ranking_cols = df.columns.tolist()
new_col_names = {col: rename_ranking(col) for col in ranking_cols}
df = df.rename(columns=new_col_names)

window = 1

df_ma = df.rolling(window=window, min_periods=1).mean()

plt.figure(figsize=(14, 8))
for i, col in enumerate(new_col_names.values()):
    plt.plot(df_ma.index, df_ma[col], label=col, linewidth=2, marker='o')


plt.xlabel("# Timestep")
plt.ylabel("# of agents")
plt.title(f"Evolution of Q-values rankings (moving average window={window})")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.autoscale(enable=True, axis='x', tight=True)
plt.grid(True)
plt.savefig("qvalues_rankings.png", dpi=300)