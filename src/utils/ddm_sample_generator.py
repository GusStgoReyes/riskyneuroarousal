import pyddm
import pandas as pd
from utils.load_data import (load_behavioral_data, 
                                 load_ddm_results)


def get_ddm_model(x0 = 0.5, alpha = 1, nondecision = 0.3, drift_gain = 1, drift_loss = 1, bound = 1, theta =  0.6):
    model = pyddm.gddm(starting_position=x0,
                drift=lambda gain, loss: alpha + drift_gain * gain + drift_loss * loss,
                nondecision=nondecision,
                bound=lambda t: bound - theta * t,
                conditions=['gain', 'loss'],
                T_dur=4,
                dx=0.01,
                dt=0.01)
    
    return model

def get_model_samples(sub, ddm_model, samples_per_condition = 50):
    df_model = {"gain" : [], "loss" : [], "RT": [], "accept" : []}

    for indx, row in sub.iterrows():
        sample = ddm_model.solve(conditions={"gain" : row['gain'], "loss": row['loss']}).sample(samples_per_condition)
        # Add the samples to the dataframe
        num_accept = len(sample.choice_upper)
        df_model["gain"].extend([row['gain']] * num_accept)
        df_model["loss"].extend([row['loss']] * num_accept)
        df_model["RT"].extend(sample.choice_upper)
        df_model["accept"].extend([1] * num_accept)
        num_reject = len(sample.choice_lower)
        df_model["gain"].extend([row['gain']] * num_reject)
        df_model["loss"].extend([row['loss']] * num_reject)
        df_model["RT"].extend(sample.choice_lower)
        df_model["accept"].extend([0] * num_reject)


    df_model = pd.DataFrame(df_model)
    df_model = df_model.groupby(['gain', 'loss'])[["RT", "accept"]].mean().reset_index()
    df_model["accept"] = df_model["accept"].round()
    df_model["sub"] = sub['sub'].unique()[0]
    df_model["condition"] = sub['condition'].unique()[0]

    return df_model


def model_behavior(model_ID):
    params_fit = load_ddm_results()
    data = load_behavioral_data(min_RT = 0)

    model_samples = []
    for sub in data["sub"].unique():
        sub_params_fit = params_fit.query(f"sub == {sub} & model_ID == {model_ID}")
        sub_data =  data.query(f"sub == {sub}")
        # Extract the parameters
        bound = sub_params_fit.query("param_name == 'a'")["mean"].values[0]/2
        x0 = sub_params_fit.query("param_name == 'z'")["mean"].values[0]
        x0 = 2*bound * x0  - bound
        nondecision = sub_params_fit.query("param_name == 't'")["mean"].values[0]
        if model_ID == 2:
            alpha = 0
        else:
            alpha = sub_params_fit.query("param_name == 'v_Intercept'")["mean"].values[0]
        if model_ID == 3:
            drift_gain = sub_params_fit.query("param_name == 'v_gain_plus_loss'")["mean"].values[0]
            drift_loss = -drift_gain
        else:
            drift_gain = sub_params_fit.query("param_name == 'v_gain'")["mean"].values[0]
            drift_loss = sub_params_fit.query("param_name == 'v_loss'")["mean"].values[0]
        if model_ID == 4:
            theta = 0
        else:
            theta = sub_params_fit.query("param_name == 'theta'")["mean"].values[0]
        
        # Get the model
        model = get_ddm_model(x0 = x0, 
                            alpha = alpha, 
                            nondecision = nondecision, 
                            drift_gain = drift_gain, 
                            drift_loss = drift_loss, 
                            bound = bound, 
                            theta =  theta)
        
        # Get samples
        sub_samples = get_model_samples(sub_data, model, samples_per_condition = 20)
        model_samples.append(sub_samples)
    
    model_samples = pd.concat(model_samples)
    return model_samples
