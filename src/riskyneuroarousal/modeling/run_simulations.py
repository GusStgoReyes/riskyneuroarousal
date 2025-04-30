from riskyneuroarousal.modeling.behavioral_models import (get_trials, 
                                                    simulate_behavior, 
                                                    negLL_EV_binary, 
                                                    negLL_EV_ordinal, 
                                                    negLL_CPT_binary,
                                                    negLL_CPT_ordinal,
                                                    fit_model)
import numpy as np
import itertools
import pandas as pd
import sys
import os

model_params_info = {
    "EV_binary" : {
        "negLL": negLL_EV_binary,
        "bounds": ((-10, 10), (0.1, 6), (0, 0)),
        "pars0": [-1, 1, 0],
        "param_names": ["c", "sigma", "delta"],
    }, 
    "EV_ordinal" : {
        "negLL": negLL_EV_ordinal,
        "bounds": ((0, 10), (-10, 0), (-10, 10), (0.1, 6), (0, 0)),
        "pars0": [2, -2, 1, 1, 0],
        "param_names": ["a_1", "a_2", "c", "sigma", "delta"],
    },
    "EV_binary_history" : {
        "negLL": negLL_EV_binary,
        "bounds": ((-10, 10), (0.1, 6), (-3, 3)),
        "pars0": [-1, 1, 0],
        "param_names": ["c", "sigma", "delta"],
    }, 
    "EV_ordinal_history" : {
        "negLL": negLL_EV_ordinal,
        "bounds": ((0, 10), (-10, 0), (-10, 10), (0.1, 6), (-3, 3)),
        "pars0": [2, -2, 1, 1, 0],
        "param_names": ["a_1", "a_2", "c", "sigma", "delta"],
    },
    "CPT_binary" : {
        "negLL": negLL_CPT_binary,
        "bounds": ((0.1, 2), (0, 5), (-10, 10), (0.1, 6), (0, 0)),
        "pars0": [0.9, 1.5, 1, 1, 0],
        "param_names": ["alpha", "lambd", "c", "sigma", "delta"],
    },
    "CPT_ordinal" : {
        "negLL": negLL_CPT_ordinal,
        "bounds": ((0.1, 2), (0, 5), (0, 10), (-10, 0), (-10, 10), (0.1, 6), (0, 0)),
        "pars0": [0.9, 1, 10, -2, 0.5, 1, 0],
        "param_names": ["alpha", "lambd", "a_1", "a_2", "c", "sigma", "delta"],
    },
    "CPT_binary_history" : {
        "negLL": negLL_CPT_binary,
        "bounds": ((0.1, 2), (0, 5), (-10, 10), (0.1, 6), (-3, 3)),
        "pars0": [0.9, 1.5, 1, 1, 0],
        "param_names": ["alpha", "lambd", "c", "sigma", "delta"],
    },
    "CPT_ordinal_history" : {
        "negLL": negLL_CPT_ordinal,
        "bounds": ((0.1, 2), (0, 5), (0, 10), (-10, 0), (-10, 10), (0.1, 6), (-3, 3)),
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

    condition = ["equalIndifference", "equalRange"][int(sys.argv[4])]
    save_pth = sys.argv[5]

    print(f"Running {model_name} with condition {condition} on thread {thread_indx+1} of {total_threads}", flush=True)

    # Get the trials
    if condition == "equalIndifference":
        gains = np.arange(10, 41, 2)
        losses = np.arange(5, 21, 1)
    elif condition == "equalRange":
        gains = np.arange(5, 21, 1)
        losses = np.arange(5, 21, 1)
    trials = get_trials(gains, losses, repetitions = 1)

    # Define the parameter ranges for the simulations
    param_ranges = model_param_ranges[model_name]

    # Create a grid of parameter combinations
    param_combinations = list(itertools.product(*param_ranges.values()))
    param_combinations = pd.DataFrame(param_combinations, columns=param_ranges.keys())
    param_combinations["indx"] = param_combinations.index

    # To save parameters
    recovered_params = {"param_names" : [], 
                    "estimates" : [], 
                    "real" : [],
                    "indx" : []}
    
    print("Total simulations = ", len(param_combinations), flush=True)
    for i in range(thread_indx, len(param_combinations), total_threads):
        curr_params = param_combinations.iloc[i]

        curr_params_names = model_params_info[model_name]["param_names"]
        # Simulate behavior
        trials_behavior = simulate_behavior(trials, curr_params, model = model_name).reset_index(drop=True)
        if np.mean(trials_behavior["accept"]) > 0.9 or np.mean(trials_behavior["accept"]) < 0.1:
            print(f"Row {curr_params['indx']} not plausible! :(")
            continue
        trials_behavior['history'] = trials_behavior['accept'].shift(1).fillna(0).astype(int)
        trials_behavior = trials_behavior.reset_index(drop=True)

        # Fit model
        pars, loss = fit_model(trials_behavior, 
                            model_params_info[model_name]["negLL"], 
                            pars0 = model_params_info[model_name]["pars0"],
                            bounds = model_params_info[model_name]["bounds"], 
        )
        if pars is not None:
            recovered_params["param_names"].extend(curr_params_names)
            recovered_params["param_names"].extend(["loss"])
            recovered_params["estimates"].extend(pars)
            recovered_params["estimates"].extend([loss])
            recovered_params["real"].extend(curr_params[curr_params_names].values)
            recovered_params["real"].extend([np.nan])
            recovered_params["indx"].extend([curr_params['indx']] * (len(curr_params_names)+1))
            print(f"Row {curr_params['indx']} fitted successfully!")
        else:
            print(f"Row {curr_params['indx']} not successful!!! :(")
        
    # Save the results
    recovered_params = pd.DataFrame(recovered_params)
    recovered_params["model"] = model_name
    recovered_params["condition"] = condition

    file_name = f"{model_name}_{condition}_simulations_{thread_indx}.csv"
    file_path = os.path.join(save_pth, file_name)
    if os.path.exists(file_path):
        recovered_params.to_csv(file_path, mode='a', header=False, index=False)
    else:
        recovered_params.to_csv(file_path, index=False)


