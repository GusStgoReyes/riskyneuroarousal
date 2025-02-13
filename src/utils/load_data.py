import pandas as pd
import os
from config import Config

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