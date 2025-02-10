import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load all the pupil data
dir = "/Users/gustxsr/Documents/Stanford/PoldrackLab/PAPERS/paper1_loss_aversion_pupil/eye_data/NARPS_MG_asc_processed"
pupil_data = []
for file in os.listdir(dir):
    if file.endswith("timeseries_start.csv"):
        csv = pd.read_csv(os.path.join(dir, file))
        pupil_data.append(csv)
pupil_data = pd.concat(pupil_data)

# Remove trials with nan values (these are trials that were too short)
pupil_data["sub_trial"] = pupil_data["sub"].astype(str) + "_" + pupil_data["trial"].astype(str)
nan_sub_trial = pupil_data[pupil_data["ps_preprocessed"].isna()]["sub_trial"].unique()
pupil_data = pupil_data[~pupil_data["sub_trial"].isin(nan_sub_trial)]
pupil_data["outofbounds"] = pupil_data["outofbounds"].fillna(False)

# Load the behavioral data
behav = pd.read_csv("data/behavioral_data.csv")

# Merge behav and pupil_data
data = pupil_data.merge(behav, on=["sub", "trial"])
# remove RT < 0.2 and response_int.isna()
data = data.query("RT > 0.2")
# Align timebins to the response time
# data["timebin"] -= 1.5
# if trial is 1, 65, 129, 193, remove it
data = data.query("trial not in [1, 65, 129, 193]")


save_pth = "/Users/gustxsr/Documents/Stanford/PoldrackLab/PAPERS/paper1_loss_aversion_pupil/supplementary_figures"
# EqualIndifference
equalIndifference_subjs = data.query("condition == 'equalIndifference'")["sub"].unique()
fig, axs = plt.subplots(6, 5, figsize=(30, 25))
subj_axs = axs.flatten()[:len(equalIndifference_subjs)]

for ax, subj_ID in zip(subj_axs, equalIndifference_subjs):
  print(subj_ID)
  subj_df = data.query(f"sub == {subj_ID}")
  sns.lineplot(subj_df, x = "timebin", y = "ps_preprocessed", hue = "accept", ax = ax, legend = subj_ID)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# Save
plt.savefig(os.path.join(save_pth, "equalIndifference_pupil_figures_start.png"), dpi=300)
plt.close()

# EqualRange
equalRange_subjs = data.query("condition == 'equalRange'")["sub"].unique()
fig, axs = plt.subplots(6, 5, figsize=(30, 25))
subj_axs = axs.flatten()[:len(equalRange_subjs)]

for ax, subj_ID in zip(subj_axs, equalRange_subjs):
  print(subj_ID)
  subj_df = data.query(f"sub == {subj_ID}")
  sns.lineplot(subj_df, x = "timebin", y = "ps_preprocessed", hue = "accept", ax = ax, legend = subj_ID)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# Save
plt.savefig(os.path.join(save_pth, "equalRange_pupil_figures_start.png"), dpi=300)
plt.close()



