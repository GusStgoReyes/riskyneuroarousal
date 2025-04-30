from riskyneuroarousal.modeling.behavioral_models import (
                                                    negLL_EV_binary, 
                                                    negLL_EV_ordinal, 
                                                    negLL_CPT_binary,
                                                    negLL_CPT_ordinal,
                                                    fit_model)
from riskyneuroarousal.utils.load_data import load_behavioral_data
import numpy as np
import pandas as pd
import sys
import os

model_params_info = {
    "EV_binary" : {
        "negLL": negLL_EV_binary,
        "bounds": ((-10, 10), (0.1, 10), (0, 0)),
        "pars0": [-1, 1, 0],
        "param_names": ["c", "sigma", "delta"],
    }, 
    "EV_ordinal" : {
        "negLL": negLL_EV_ordinal,
        "bounds": ((0, 10), (-10, 0), (-10, 10), (0.1, 10), (0, 0)),
        "pars0": [2, -2, 1, 1, 0],
        "param_names": ["a_1", "a_2", "c", "sigma", "delta"],
    },
    "EV_binary_history" : {
        "negLL": negLL_EV_binary,
        "bounds": ((-10, 10), (0.1, 10), (-3, 3)),
        "pars0": [-1, 1, 0],
        "param_names": ["c", "sigma", "delta"],
    }, 
    "EV_ordinal_history" : {
        "negLL": negLL_EV_ordinal,
        "bounds": ((0, 10), (-10, 0), (-10, 10), (0.1, 10), (-3, 3)),
        "pars0": [2, -2, 1, 1, 0],
        "param_names": ["a_1", "a_2", "c", "sigma", "delta"],
    },
    "CPT_binary" : {
        "negLL": negLL_CPT_binary,
        "bounds": ((0.1, 2), (0, 5), (-10, 10), (0.1, 10), (0, 0)),
        "pars0": [0.9, 1.5, 1, 1, 0],
        "param_names": ["alpha", "lambd", "c", "sigma", "delta"],
    },
    "CPT_ordinal" : {
        "negLL": negLL_CPT_ordinal,
        "bounds": ((0.1, 2), (0, 5), (0, 10), (-10, 0), (-10, 10), (0.1, 10), (0, 0)),
        "pars0": [0.9, 1, 10, -2, 0.5, 1, 0],
        "param_names": ["alpha", "lambd", "a_1", "a_2", "c", "sigma", "delta"],
    },
    "CPT_binary_history" : {
        "negLL": negLL_CPT_binary,
        "bounds": ((0.1, 2), (0, 5), (-10, 10), (0.1, 10), (-3, 3)),
        "pars0": [0.9, 1.5, 1, 1, 0],
        "param_names": ["alpha", "lambd", "c", "sigma", "delta"],
    },
    "CPT_ordinal_history" : {
        "negLL": negLL_CPT_ordinal,
        "bounds": ((0.1, 2), (0, 5), (0, 10), (-10, 0), (-10, 10), (0.1, 10), (-3, 3)),
        "pars0": [0.9, 1, 10, -2, 0.5, 1, 0],
        "param_names": ["alpha", "lambd", "a_1", "a_2", "c", "sigma", "delta"],
    },
}

# Define the parameter ranges for the simulations
model_param_ranges = {
    "EV_binary" : {"c" : np.linspace(-2.5, 2.5, 10),
                   "sigma" : np.linspace(0.5, 10.5, 10),
                   "delta" : [0]},

    "EV_ordinal" : {"a_1" : [3],
                    "a_2" : [-3],
                    "c" : np.linspace(-2.5, 2.5, 10),
                    "sigma" : np.linspace(0.5, 10.5, 10),
                    "delta" : [0],},

    "EV_binary_history" : {"c" : np.linspace(-2.5, 2.5, 10),
                            "sigma" : np.linspace(0.5, 10.5, 10),
                            "delta" : [-1.5, 1.5],},

    "EV_ordinal_history" : {"a_1" : [3],
                            "a_2" : [-3],
                            "c" : np.linspace(-2.5, 2.5, 10),
                            "sigma" : np.linspace(0.5, 10.5, 10),
                            "delta" : [-1.5, 1.5],},

    "CPT_binary" : {"alpha" : np.linspace(0.6, 1.1, 5),
                    "lambd" : np.linspace(0.8, 3.5, 5),
                    "c" : np.linspace(-2.5, 2.5, 5),
                    "sigma" : np.linspace(0.5, 10.5, 5),
                    "delta" : [0],},

    "CPT_ordinal" : {"alpha" : np.linspace(0.6, 1.1, 5),
                    "lambd" : np.linspace(0.8, 3.5, 5),
                    "a_1" : [3],
                    "a_2" : [-3],
                    "c" : np.linspace(-2.5, 2.5, 5),
                    "sigma" : np.linspace(0.5, 10.5, 5),
                    "delta" : [0],},

    "CPT_binary_history" : {"alpha" : np.linspace(0.6, 1.1, 5),
                            "lambd" : np.linspace(0.8, 3.5, 5),
                            "c" : np.linspace(-2.5, 2.5, 5),
                            "sigma" : np.linspace(0.5, 10.5, 5),
                            "delta" : [-1.5, 1.5],},

    "CPT_ordinal_history" : {"alpha" : np.linspace(0.6, 1.1, 5),
                            "lambd" : np.linspace(0.8, 3.5, 5),
                            "a_1" : [3],
                            "a_2" : [-3],
                            "c" : np.linspace(-2.5, 2.5, 5),
                            "sigma" : np.linspace(0.5, 10.5, 5),
                            "delta" : [-1.5, 1.5],},
}

if __name__ == "__main__":
    # Get the inputs to the program
    model_name = ["EV_binary", "EV_ordinal",
                  "CPT_binary", "CPT_ordinal", 
                  "EV_binary_history", "EV_ordinal_history",
                  "CPT_binary_history", "CPT_ordinal_history"][int(sys.argv[1])]
    
    thread_indx = int(sys.argv[2])
    total_threads = int(sys.argv[3])

    save_pth = sys.argv[4]
    config_pth = sys.argv[5]

    print(f"Running {model_name}  on thread {thread_indx+1} of {total_threads}", flush=True)

    # Load the data
    df = load_behavioral_data(min_RT=0.2, config_pth = config_pth).reset_index(drop=True)
    subs = df["sub"].unique()

    # To save parameters
    recovered_params = {"param_names" : [], 
                    "estimates" : [], 
                    "sub" : [], 
                    "condition" : []}
    
    print(f"Number of subjects: {len(subs)}", flush=True)
    for i in range(thread_indx, len(subs), total_threads):
        trials_behavior = df.query(f"sub == {subs[i]}").reset_index(drop=True)
        trials_behavior["history"] = trials_behavior["accept"].shift(1).fillna(0).astype(int)
        # Fit model
        pars, loss = fit_model(trials_behavior, 
                            model_params_info[model_name]["negLL"], 
                            pars0 = model_params_info[model_name]["pars0"],
                            bounds = model_params_info[model_name]["bounds"], 
        )
        if pars is not None:
            curr_params_names = model_params_info[model_name]["param_names"]
            recovered_params["param_names"].extend(curr_params_names)
            recovered_params["param_names"].extend(["loss"])
            recovered_params["estimates"].extend(pars)
            recovered_params["estimates"].extend([loss])
            recovered_params["sub"].extend([subs[i]] * (len(curr_params_names)+1))
            condition = trials_behavior["condition"].unique()[0]
            recovered_params["condition"].extend([condition] * (len(curr_params_names)+1))
            print(f"Subject {subs[i]} fitted successfully!")
        else:
            print(f"Subject {subs[i]} failed :(!!")
        
    # Save the results
    recovered_params = pd.DataFrame(recovered_params)
    recovered_params["model"] = model_name

    file_name = f"{model_name}_{condition}_subjectFit_{thread_indx}.csv"
    file_path = os.path.join(save_pth, file_name)
    if os.path.exists(file_path):
        recovered_params.to_csv(file_path, mode='a', header=False, index=False)
    else:
        recovered_params.to_csv(file_path, index=False)
