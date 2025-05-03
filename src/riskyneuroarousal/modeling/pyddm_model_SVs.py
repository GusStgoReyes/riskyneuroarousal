import pyddm
from pyddm import Sample
import sys
import pandas as pd
import pickle as pkl
import numpy as np


def get_model_samples(sub, ddm_model, samples_per_condition=200):
    df_model = {"gain": [], "loss": [], "rt": [], "accept": []}

    for indx, row in sub.df.iterrows():
        sample = ddm_model.solve(
            conditions={"gain": row["gain"], "loss": row["loss"]}
        ).sample(samples_per_condition)
        for s in sample.choice_upper:
            df_model["gain"].append(row["gain"])
            df_model["loss"].append(row["loss"])
            df_model["rt"].append(s)
            df_model["accept"].append(1)
        for s in sample.choice_lower:
            df_model["gain"].append(row["gain"])
            df_model["loss"].append(row["loss"])
            df_model["rt"].append(s)
            df_model["accept"].append(0)

    return df_model


def ddm(
    data,
    parameters,
    conditions,
    T_dur=4,
    dx=0.002,
    dt=0.002,
    rt_column_name="RT",
    choice_column_name="accept",
):
    drift_func = lambda drift, c, alpha, lambd_drift, gain, loss: c + drift*np.power(gain, alpha) - lambd_drift * np.power(loss, alpha)
    model = pyddm.gddm(
        starting_position='x0',
        drift=drift_func,
        nondecision='nondecision',
        bound= lambda bound, theta, t: bound - theta * t,
        parameters=parameters,
        conditions=conditions,
        T_dur=T_dur,
        dx=dx,
        dt=dt,
    )

    # Remove any outliers (ddm is sensitive to outliers)
    data = data.query(f"{choice_column_name} != -1 & {rt_column_name} > .2")
    samples = Sample.from_pandas_dataframe(
        data, rt_column_name=rt_column_name, choice_column_name=choice_column_name
    )

    # Fit the model
    model.fit(
        sample=samples,
        fitting_method="differential_evolution",
    )

    return model

def parameters():
    return {
        "drift" : (0, 0.4), 
        "alpha" : (0.1, 1.4),
        "c" : (-0.3, 0.3),
        "lambd_drift" : (0, 0.3),
        "x0": (-1, 1),
        "nondecision": (0.1, 1.5),
        "theta": (0, 1.5),
        "bound" : (0.8, 3),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit a DDM model to a subject's data")
    parser.add_argument(
        "--subj_ID", type=int, help="The subject ID to fit the model to"
    )
    parser.add_argument("--results_pth", type=str, help="The path to save the results")
    args = parser.parse_args()

    # Load a subjects data (TODO: make dynamic src path)
    behav_data = pd.read_csv(
        "/scratch/users/gustxsr/riskyneuroarousal/data/behavioral_data.csv"
    ).query(f"sub == {int(args.subj_ID)}")

    # Select a model
    params = parameters()
    conditions = ["gain", "loss"]

    # Run the ddm fit
    ddm_model = ddm(
        behav_data,
        params,
        conditions,
        T_dur=4,
        dx=0.002,
        dt=0.002,
        rt_column_name="RT",
        choice_column_name="accept",
    )

    # Save the parameters
    params_fit = ddm_model.parameters()
    with open(
        f"{args.results_pth}/sub{args.subj_ID}.pkl", "wb"
    ) as file:
        # Dump the dictionary into the file
        pkl.dump(params_fit, file)

    params_save = {}
    for param_category in params_fit:
        for param_key in params_fit[param_category]:
            if (
                type(params_fit[param_category][param_key]) == float
                or type(params_fit[param_category][param_key]) == int
            ):
                params_save[f"{param_category}_{param_key}"] = params_fit[
                    param_category
                ][param_key]
            else:
                params_save[f"{param_category}_{param_key}"] = params_fit[
                    param_category
                ][param_key].default()

    params_save[ddm_model.fitresult.loss] = ddm_model.fitresult.value()
    params_save["sub"] = int(args.subj_ID)

    pd.DataFrame(params_save).to_csv(
        f"{args.results_pth}/sub{args.subj_ID}.csv"
    )
