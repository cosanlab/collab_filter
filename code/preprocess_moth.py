"""
This file loads in the Moth data (moth_orig) and creates a new file called moth.csv where participants have been filtered by jump counts (> 20 jumps). Jumps were pre-calculate by Max in the collected data based on the derivative of rating movement overtime.
"""
# %%
import numpy as np
import pandas as pd
from pathlib import Path

# %%
base_dir = Path("/Users/Esh/Documents/dartmouth/cosan/projects/collab_filter/")
data_dir = base_dir / "data"
result_file = data_dir / "moth.csv"

# Set jump threshold for filtering
jump_thresh = 20

# %%
moth = pd.read_csv(data_dir / "moth_orig.csv")
# 133 subjects in take 1 and 112 in take 2
assert np.allclose(moth.groupby("take").workerId.nunique().values, [133, 112])

# %%
# Filter out participants that have more that 20 jumps
jump_counts = moth.groupby(["workerId", "movie"]).jump_count.mean().reset_index()
filter_list = (
    jump_counts.loc[jump_counts.jump_count > jump_thresh, ["workerId", "movie"]]
    .apply(lambda row: f"{row['movie']}-good_{row['workerId']}", axis=1)
    .to_list()
)
moth_filtered = moth.query("combo_name not in @filter_list").reset_index(drop=True)

print(f"Participants after filtering by > {jump_thresh} jumps")
print(moth_filtered.groupby("take").workerId.nunique())

# %%
# Save it
moth_filtered.to_csv(result_file, index=False)
print(f"Cleaned file saved to: {result_file}")
