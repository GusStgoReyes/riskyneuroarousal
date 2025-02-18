import pandas as pd
import os
from riskyneuroarousal.config import Config

def load_config():
    """
        Load the configuration file.
    """
    # Get path to the data from the config file
    config = Config()
    username = config.username
    return config, username

def load_behavioral_data(min_RT = 0.2):
    """
        Load behavioral data from the specified file path.
    """
    config, username = load_config()
    data_path = os.path.join(config.data_path[f"{username}"], "behavioral_data.csv")

    # Load the data and remove trials with RT < 0.2 (includes no response trials)
    data = pd.read_csv(data_path).query(f'RT > {min_RT}')

    return data

def load_pt_results():
    """
        Load the Prospect theory parameters modeled
    """
    config, username = load_config()
    data_path = os.path.join(config.data_path[f"{username}"], "pt_results.csv")

    # Load the data
    data = pd.read_csv(data_path)

    return data

def load_ddm_results():
    """
        Load the DDM parameters modeled
    """
    config, username = load_config()
    data_path = os.path.join(config.data_path[f"{username}"], "ddm_parameters.csv")

    # Load the data
    data = pd.read_csv(data_path)

    return data

def load_pupil_results():
    """
        Load the pupil results
    """
    config, username = load_config()
    data_path = os.path.join(config.data_path[f"{username}"], "pupil_coefs.csv")

    # Load the data
    data = pd.read_csv(data_path)

    return data

def load_pupil_data():
    """
    Load binned pupil data aligned to start time!
    """
    config, username = load_config()
    data_path = os.path.join(config.data_path[f"{username}"], "pupil_data_all.csv")

    # Load the data
    data = pd.read_csv(data_path)

    return data

def load_baseline_data():
    """
    Load baseline pupil data
    """
    config, username = load_config()
    data_path = os.path.join(config.data_path[f"{username}"], "baseline_data.csv")

    # Load the data
    data = pd.read_csv(data_path)

    return data
## SCRAP!!!
# # Load all the pupil data
# dir = "/Users/gustxsr/Documents/Stanford/PoldrackLab/PAPERS/paper1_loss_aversion_pupil/eye_data/NARPS_MG_asc_processed"
# pupil_data = []
# for file in os.listdir(dir):
#     if file.endswith("timeseries_start.csv"):
#         csv = pd.read_csv(os.path.join(dir, file))
#         pupil_data.append(csv)
# pupil_data = pd.concat(pupil_data)

# # Remove trials with nan values (these are trials that were too short)
# pupil_data["sub_trial"] = pupil_data["sub"].astype(str) + "_" + pupil_data["trial"].astype(str)
# nan_sub_trial = pupil_data[pupil_data["ps_preprocessed"].isna()]["sub_trial"].unique()
# pupil_data = pupil_data[~pupil_data["sub_trial"].isin(nan_sub_trial)]
# pupil_data["outofbounds"] = pupil_data["outofbounds"].fillna(False)

# # Load the behavioral data
# behav = pd.read_csv("../../data/behavioral_data.csv")

# # Merge behav and pupil_data
# data = pupil_data.merge(behav, on=["sub", "trial"])
# # remove RT < 0.2 and response_int.isna()
# data = data.query("RT > 0.2")