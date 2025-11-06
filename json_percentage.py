import json
import glob
from collections import defaultdict

# total = (total agents * timestep * omega) - total agents (first iteration has no learning)
total = 490  

files = sorted(glob.glob("json/j*_table_changes.json"))

sums = defaultdict(float)
counts = defaultdict(int)

for fname in files:
    with open(fname, "r") as f:
        data = json.load(f)
        for entry in data:
            for k, v in entry.items():
                sums[k] += v
                counts[k] += 1

averages = {k: sums[k] / counts[k] for k in sums}

results = {
    k: {
        "value_avg": round(averages[k], 4),
        "percentage": round((averages[k] / total) * 100, 4)
    }
    for k in averages
}

with open("json/average_table_changes.json", "w") as f:
    json.dump(results, f, indent=2)

