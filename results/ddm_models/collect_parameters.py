import pandas as pd
import os

all_params_fit = []
pth_params = "./parameters"
pth_result = "./ddm_parameters.csv"

for file in os.listdir(pth_params):
    if file.endswith(".csv"):
        params_fit = pd.read_csv(os.path.join(pth_params, file))
    else:
        continue

    params_fit["sub"] = int(file.split("_")[0][3:])
    params_fit["model_ID"] = int(file.split("_")[1].split(".")[0][5:])
    all_params_fit.append(params_fit)

all_params_fit = pd.concat(all_params_fit).reset_index(drop=True)
all_params_fit.rename(columns={"Unnamed: 0" : "param_name"}, inplace=True)

all_params_fit.to_csv(pth_result)
