"""
This file loads in the vanBaar data (vanBaar_data.csv) and creates a new file called decisions.csv. The preprocessing includes:

1) Remove subjects that don't have complete data (80 trials)
2) Removing trials in which investment amount == 0
3) Making unique trial identifiers from using investment_multiplier__duplicate_counter; where duplicate_counter is an integer that ensures multiple trials with the same investment and multiplier are ultimately treated as separate "items" in a user x item matrix
"""
# %%
import pandas as pd
from pathlib import Path

# %%
base_dir = Path("/Users/Esh/Documents/dartmouth/cosan/projects/collab_filter/")
data_dir = base_dir / "data"
result_file = data_dir / "decisions.csv"

# %%
decisions = pd.read_csv(data_dir / "vanBaar_data.csv")

# Create unique trial identifiers which are the combo of multiplier and investment amount
decisions = decisions.assign(
    Trial=decisions.Investment_Amount.astype("str")
    + "_"
    + decisions.Multiplier.astype("str")
)

# %%
print(f"Original number of subjects: {decisions.Subject.nunique()}")
print(
    f"Original trials per subject: {decisions.groupby('Subject').Trial.nunique().unique()}"
)

# Filter out subjects with < 80 trials which leaves 60 subjects
good_participants = decisions.Subject.unique()[
    decisions.groupby("Subject").size() == 80
]
# Also filter out trials in which the investment amount was 0 this leaves 30 trials per subject
decisions_filtered = decisions.query(
    "Subject in @good_participants and Investment_Amount != 0"
).reset_index(drop=True)


print(f"Filtered number of subjects: {decisions_filtered.Subject.nunique()}")
print(
    f"Filtered trials per subject: {decisions_filtered.groupby('Subject').Trial.nunique().unique()}"
)

# %%
# We don't have a unique identifier per "item" in this dataset
# So use combinations of the investment amount, multiplier, and the number of repeats of that combo as unique identifiers
subjects = decisions_filtered.groupby("Subject")


def uniquify_trials(grp):
    "Makes trials unique by appending an number to the end of the trial identifier if its repeated more than once"
    trials = grp.Trial.values
    seen = {}
    new_trials = []
    for trial in trials:
        if trial in seen:
            seen[trial] += 1
            new_trials.append(f"{trial}__{seen[trial]}")
        else:
            seen[trial] = 1
            new_trials.append(trial)
    grp["Trial"] = new_trials
    return grp


decisions_filtered = subjects.apply(uniquify_trials)

# %%
# Save it
decisions_filtered.to_csv(result_file, index=False)
print(f"Cleaned file saved to: {result_file}")
