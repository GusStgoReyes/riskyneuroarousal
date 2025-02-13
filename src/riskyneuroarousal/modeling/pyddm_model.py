import pyddm
from pyddm import Sample
import sys
import pandas as pd
import pickle as pkl
import numpy as np

def get_model_samples(sub, ddm_model, samples_per_condition = 200):
    df_model = {"gain" : [], "loss" : [], "rt": [], "accept" : []}

    for indx, row in sub.df.iterrows():
        sample = ddm_model.solve(conditions={"gain" : row['gain'], "loss": row['loss']}).sample(samples_per_condition)
        for s in sample.choice_upper:
            df_model["gain"].append(row['gain'])
            df_model["loss"].append(row['loss'])
            df_model["rt"].append(s)
            df_model["accept"].append(1)
        for s in sample.choice_lower:
            df_model["gain"].append(row['gain'])
            df_model["loss"].append(row['loss'])
            df_model["rt"].append(s)
            df_model["accept"].append(0)

    return df_model

def ddm(data, 
        response_bias_func, 
        drift_func, 
        nondecision_func, 
        bound_func,
        parameters, 
        conditions, 
        T_dur = 4, 
        dx = 0.002, 
        dt = 0.002, 
        rt_column_name='RT',
        choice_column_name='accept'):
    
    model = pyddm.gddm(starting_position=response_bias_func,
                        drift=drift_func,
                        nondecision=nondecision_func,
                        bound=bound_func,
                        parameters= parameters,
                        conditions=conditions,
                        T_dur = T_dur,
                        dx = dx,
                        dt = dt)

    
    # Remove any outliers (ddm is sensitive to outliers)
    data = data.query(f"{choice_column_name} != -1 & {rt_column_name} > .2")
    samples = Sample.from_pandas_dataframe(data, rt_column_name=rt_column_name, choice_column_name=choice_column_name)

    # Fit the model
    model.fit(sample = samples, 
              fitting_method = 'differential_evolution',
              )

    return model

def drift_function(model_ID):
    if model_ID == 1:
        # Model 1: Drift has alpha, drift_gain, drift_loss parameters
        return lambda alpha, drift_gain, drift_loss, gain, loss: alpha + drift_gain * gain - drift_loss * loss
    elif model_ID == 2:
        # Model 2: Drift has alpha, drift parameter
        return lambda alpha, drift, gain, loss: alpha + drift * ( gain - loss )
    elif model_ID == 3:
        # Model 3: Drift has drift_gain, drift_loss parameters
        return lambda drift_gain, drift_loss, gain, loss: drift_gain * gain - drift_loss * loss
    else:
        raise ValueError(f"Model ID {model_ID} is not valid")
    
def parameters(model_ID):
    if model_ID == 1:
        return {"IC" : (-1, 1),
                "ndtime" : (0.1, 1.5),
                "alpha" : (-1, 1),
                "drift_gain" : (-0.1, 0.75),
                "drift_loss" : (-0.1, 0.75),
                "B" : (0.5, 1.75),}
    elif model_ID == 2:
        return {"IC" : (-1, 1),
                "ndtime" : (0.25, 1.5),
                "alpha" : (-0.5, 0.5),
                "drift" : (-0.5, 0.5),
                "B" : (0.5, 1.5)}
    elif model_ID == 3:
        return {"IC" : (-1, 1),
                "ndtime" : (0.25, 1.5),
                "drift_gain" : (0, 0.5),
                "drift_loss" : (0, 0.5),
                "B" : (0.5, 1.5)}
    else:
        raise ValueError(f"Model ID {model_ID} is not valid")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit a DDM model to a subject's data")
    parser.add_argument("--subj_ID", type=int, help="The subject ID to fit the model to")
    parser.add_argument("--model", type=str, help="The model to fit the data to (from 1 to 3)")
    parser.add_argument("--results_pth", type=str, help="The path to save the results")
    args = parser.parse_args()

    # Load a subjects data (TODO: make dynamic src path)
    behav_data = pd.read_csv("/scratch/users/gustxsr/PoldrackLab/riskyneuroarousal/data/behavioral_data.csv").query(f"sub == {int(args.subj_ID)}")

    # Select a model
    model_ID = int(args.model)
    # Define model parameters
    response_bias_func = "IC"
    nondecision_func = "ndtime"
    bound_func = "B"
    drift_func = drift_function(model_ID)
    params = parameters(model_ID)
    conditions = ['gain', 'loss']
    
    # Run the ddm fit
    ddm_model = ddm(behav_data, response_bias_func, drift_func, nondecision_func, bound_func, params, conditions)
    
    # Save the parameters
    params_fit = ddm_model.parameters()
    with open(f"{args.results_pth}/sub{args.subj_ID}_model{args.model}.pkl", 'wb') as file:
        # Dump the dictionary into the file
        pkl.dump(params_fit, file)

    params_save = {}
    for param_category in params_fit:
        for param_key in params_fit[param_category]:
            if type(params_fit[param_category][param_key]) == float or type(params_fit[param_category][param_key]) == int:
                params_save[f"{param_category}_{param_key}"] = params_fit[param_category][param_key]
            else:
                params_save[f"{param_category}_{param_key}"] = params_fit[param_category][param_key].default()

    params_save[ddm_model.fitresult.loss] = ddm_model.fitresult.value()
    params_save['model_ID'] = model_ID
    params_save['sub'] = int(args.subj_ID)

    pd.DataFrame(params_save).to_csv(f"{args.results_pth}/sub{args.subj_ID}_model{args.model}.csv")
