import numpy as np
from scipy.optimize import minimize, basinhopping

import pandas as pd
from riskyneuroarousal.utils.load_data import load_behavioral_data
from scipy import stats
import random
"""
This script fits cumulative prospect theory to the data. Assumes that the responses are from a 4-point scale (1-4) 
and that the data is in a pandas DataFrame. 
- Save parameters to a csv from all the subjects
- Convert the prospects into their subjective values. 
- Convert to expected value of the 4-point scale. 
- To this, we can fit a sigmoid function to represent how the subjective value are converted to probability of accepting. 
"""

def value_function(x, alpha, lambd):
    """
    Value function for prospect theory.
    """
    if x >= 0:
        return np.power(x, alpha)
    else:
        return -lambd * np.power(np.abs(x), alpha)

def inv_value_function(v, alpha, lambd):
    """
    Inverse value function for prospect theory.
    """
    if v >= 0:
        return np.power(v, 1 / alpha)
    else:
        return -np.power(np.abs(v) / lambd, 1 / alpha)

def get_trials(gains, losses, repetitions = 1):
    trials = []
    for g in gains:
        for l in losses:
            trials += [{"gain": g, "loss": l}] * repetitions
    
    random.shuffle(trials)
    return pd.DataFrame(trials)

def simulate_behavior(trials, pars, model = "EV_binary"):
    model_description = model.split("_")
    if len(model_description) == 3:
        history = True
    else:
        history = False
  
    if model_description[0] == "EV":
        c, sigma = pars["c"], pars["sigma"]
        # Compute the decision value based on parameters
        trials["decision_value"] = (trials["gain"] - trials["loss"]) / 2 
        trials["decision_value"] -= c
        # Add noise
        trials["decision_value"] += np.random.normal(loc = 0, scale = sigma, size = len(trials))
    elif model_description[0] == "CPT":
        alpha, lambd, c, sigma = pars["alpha"], pars["lambd"], pars["c"], pars["sigma"]
        # Compute the decision value based on parameters
        trials["subjective_gain"] = trials.apply(lambda x: value_function(x["gain"], alpha = alpha, lambd = lambd), axis = 1)
        trials["subjective_loss"] = trials.apply(lambda x: value_function(-x["loss"], alpha = alpha, lambd = lambd), axis = 1)
        trials["subj_value"] = trials["subjective_gain"] + trials["subjective_loss"]
        trials["decision_value"] = trials.apply(lambda x: inv_value_function(x["subj_value"], alpha = alpha, lambd = lambd), axis = 1)
        trials["decision_value"] -= c
        # Add noise
        trials["decision_value"] += np.random.normal(loc = 0, scale = sigma, size = len(trials))
    
    if not history:
        if model_description[1] == "binary":
            # Compute the response based on decision value
            trials["response_int"] = trials["decision_value"].apply(lambda x: 1 if x > 0 else 0)
            trials["accept"] = trials["response_int"].apply(lambda x: 1 if x == 1 else 0)
        elif model_description[1] == "ordinal":
            a_1, a_2 = pars["a_1"], pars["a_2"]
            trials["response_int"] = trials["decision_value"].apply(lambda x: 4 if x > a_1 else (3 if x > 0 else (2 if x > a_2 else 1)))
            trials["accept"] = trials["response_int"].apply(lambda x: 1 if x > 2 else 0)
    
    else:
        # Add history bias
        delta = pars["delta"]
        
        history = [0]
        response_int = []
        if model_description[1] == "binary":
            for _, trial in trials.iterrows():
                if history[-1] == 1:
                    dv = trial["decision_value"] +  delta
                else:
                    dv = trial["decision_value"]
                if dv > 0:
                    response_int.append(1)
                    history.append(1)
                else:
                    response_int.append(0)
                    history.append(0)
            trials["response_int"] = response_int
            trials["accept"] = trials["response_int"].apply(lambda x: 1 if x == 1 else 0)
        elif model_description[1] == "ordinal":
            for _, trial in trials.iterrows():
                if history[-1] == 1:
                    dv = trial["decision_value"] +  delta
                else:
                    dv = trial["decision_value"]
                if dv > pars["a_1"]:
                    response_int.append(4)
                    history.append(1)
                elif dv <= pars["a_1"] and dv > 0:
                    response_int.append(3)
                    history.append(1)
                elif dv <= 0 and dv > pars["a_2"]:
                    response_int.append(2)
                    history.append(0)
                else:
                    response_int.append(1)
                    history.append(0)
            trials["response_int"] = response_int
            trials["accept"] = trials["response_int"].apply(lambda x: 1 if x > 2 else 0)

    trials = trials[["gain", "loss", "accept", "response_int"]]   
    return trials

def negLL_CPT_binary(pars0, *args):
    """
    Fit the model to the data.
        pars0 = [alpha, lambd, c, sigma, delta]
        args = [trials] (pandas DataFrame)
    """

    alpha, lambd, c, sigma, delta = pars0
    trials = args[0]

    # Compute decision value based on parameters
    trials["subjective_gain"] = trials.apply(lambda x: value_function(x["gain"], alpha = alpha, lambd = lambd), axis = 1)
    trials["subjective_loss"] = trials.apply(lambda x: value_function(-x["loss"], alpha = alpha, lambd = lambd), axis = 1)
    trials["subj_value"] = trials["subjective_gain"] + trials["subjective_loss"]
    trials["decision_value"] = trials.apply(lambda x: inv_value_function(x["subj_value"], alpha = alpha, lambd = lambd), axis = 1)
    trials["decision_value"] -= c
    trials["decision_value"] += delta * trials["history"]

    # Calculate acceptance probability
    prob_accept = stats.norm.cdf(trials["decision_value"], loc=0, scale=sigma)
    prob_accept = np.clip(prob_accept, 1e-10, 1 - 1e-10)

    # Bernoulli log-likelihood
    nll = -np.sum(trials["accept"] * np.log(prob_accept) + (1 - trials["accept"]) * np.log(1 - prob_accept))

    return nll

def negLL_CPT_ordinal(pars0, *args):
    """
    Fit the model to the data.
        pars0 = [alpha, lambd, a_1, a_2, c, sigma, delta]
        args = [trials] (pandas DataFrame)
    """

    alpha, lambd, a_1, a_2, c, sigma, delta = pars0
    trials = args[0]

    # Compute decision value based on parameters
    trials["subjective_gain"] = trials.apply(lambda x: value_function(x["gain"], alpha = alpha, lambd = lambd), axis = 1)
    trials["subjective_loss"] = trials.apply(lambda x: value_function(-x["loss"], alpha = alpha, lambd = lambd), axis = 1)
    trials["subj_value"] = trials["subjective_gain"] + trials["subjective_loss"]
    trials["decision_value"] = trials.apply(lambda x: inv_value_function(x["subj_value"], alpha = alpha, lambd = lambd), axis = 1)
    trials["decision_value"] -= c
    trials["decision_value"] += delta* trials["history"]

    # Gather decision value based on responses
    response4 = trials.query("response_int == 4")["decision_value"].values
    response3 = trials.query("response_int == 3")["decision_value"].values
    response2 = trials.query("response_int == 2")["decision_value"].values
    response1 = trials.query("response_int == 1")["decision_value"].values

    # Compute probabilities of the responses
    prob_4 = 1 - stats.norm.cdf(a_1 - response4, loc = 0, scale = sigma)

    prob_3 = (stats.norm.cdf(a_1 - response3, loc = 0, scale = sigma) - 
              stats.norm.cdf(0 - response3, loc = 0, scale = sigma))
    
    prob_2 = (stats.norm.cdf(0 - response2, loc = 0, scale = sigma) - 
              stats.norm.cdf(a_2 - response2, loc = 0, scale = sigma))
    
    prob_1 = stats.norm.cdf(a_2 - response1, loc = 0, scale = sigma)

    # Clip the probabilities to not run into instability issues
    prob_4 = np.clip(prob_4, 1e-10, 1 - 1e-10)
    prob_3 = np.clip(prob_3, 1e-10, 1 - 1e-10)
    prob_2 = np.clip(prob_2, 1e-10, 1 - 1e-10)
    prob_1 = np.clip(prob_1, 1e-10, 1 - 1e-10)

    # Compute negative log likelihood
    nll = -np.sum(np.log(prob_4)) - np.sum(np.log(prob_3)) - np.sum(np.log(prob_2)) - np.sum(np.log(prob_1))

    return nll

def negLL_EV_binary(pars0, *args):
    """
    Fit the model to the data.
        pars0 = [c, sigma, delta]
        args = [trials] (pandas DataFrame)
    """

    c, sigma, delta = pars0
    trials = args[0]

    trials["decision_value"] = (trials["gain"] - trials["loss"]) / 2 
    trials["decision_value"] -= c
    trials["decision_value"] += delta* trials["history"]

    # Calculate acceptance probability
    prob_accept = stats.norm.cdf(trials["decision_value"], loc=0, scale=sigma)
    prob_accept = np.clip(prob_accept, 1e-10, 1 - 1e-10)

    # Bernoulli log-likelihood
    nll = -np.sum(trials["accept"] * np.log(prob_accept) + (1 - trials["accept"]) * np.log(1 - prob_accept))

    return nll

def negLL_EV_ordinal(pars0, *args):
    a_1, a_2, c, sigma, delta = pars0
    trials = args[0]

    # Compute decision value based on parameters
    trials["decision_value"] = (trials["gain"] - trials["loss"]) / 2 
    trials["decision_value"] -= c
    trials["decision_value"] += delta * trials["history"]

    # Gather decision value based on responses
    response4 = trials.query("response_int == 4")["decision_value"].values
    response3 = trials.query("response_int == 3")["decision_value"].values
    response2 = trials.query("response_int == 2")["decision_value"].values
    response1 = trials.query("response_int == 1")["decision_value"].values

    # Compute probabilities of the responses
    prob_4 = 1 - stats.norm.cdf(a_1 - response4, loc = 0, scale = sigma)

    prob_3 = (stats.norm.cdf(a_1 - response3, loc = 0, scale = sigma) - 
              stats.norm.cdf(0 - response3, loc = 0, scale = sigma))
    
    prob_2 = (stats.norm.cdf(0 - response2, loc = 0, scale = sigma) - 
              stats.norm.cdf(a_2 - response2, loc = 0, scale = sigma))
    
    prob_1 = stats.norm.cdf(a_2 - response1, loc = 0, scale = sigma)

    # Clip the probabilities to not run into instability issues
    prob_4 = np.clip(prob_4, 1e-10, 1 - 1e-10)
    prob_3 = np.clip(prob_3, 1e-10, 1 - 1e-10)
    prob_2 = np.clip(prob_2, 1e-10, 1 - 1e-10)
    prob_1 = np.clip(prob_1, 1e-10, 1 - 1e-10)

    # Compute negative log likelihood
    nll = -np.sum(np.log(prob_4)) - np.sum(np.log(prob_3)) - np.sum(np.log(prob_2)) - np.sum(np.log(prob_1))

    return nll

def convert_subjective(df, pars, model="EV_binary"):
    """
    Convert the subjective values to probabilities.
    """
    model_description = model.split("_")
    if len(model_description) == 3:
        history = True
    else:
        history = False

    if model_description[0] == "EV":
        # Compute decision value based on parameters
        df["decision_value"] = (df["gain"] - df["loss"]) / 2 
        df["decision_value"] -= pars["c"].values[0]

    elif model_description[0] == "CPT":
        # Compute decision value based on parameters
        df["subjective_gain"] = df.apply(lambda x: value_function(x["gain"], alpha = pars["alpha"].values[0], lambd = pars["lambd"].values[0]), axis = 1)
        df["subjective_loss"] = df.apply(lambda x: value_function(-x["loss"], alpha = pars["alpha"].values[0], lambd = pars["lambd"].values[0]), axis = 1)
        df["subj_value"] = df["subjective_gain"] + df["subjective_loss"]
        df["decision_value"] = df.apply(lambda x: inv_value_function(x["subj_value"], alpha = pars["alpha"].values[0], lambd = pars["lambd"].values[0]), axis = 1)
        df["decision_value"] -= pars["c"].values[0]
    
    if history:
        # Add history bias
        df["decision_value"] += pars["delta"].values[0] * df["history"]
    
    # Compute probabilities of the responses
    if model_description[1] == "ordinal":
        df["prob_4"] = 1 - stats.norm.cdf(pars["a_1"].values[0] - df["decision_value"], loc = 0, scale = pars["sigma"].values[0])
        df["prob_3"] = (stats.norm.cdf(pars["a_1"].values[0] - df["decision_value"], loc = 0, scale = pars["sigma"].values[0]) -
                      stats.norm.cdf(0 - df["decision_value"], loc = 0, scale = pars["sigma"].values[0]))
        df["prob_2"] = (stats.norm.cdf(0 - df["decision_value"], loc = 0, scale = pars["sigma"].values[0]) -
                      stats.norm.cdf(pars["a_2"].values[0] - df["decision_value"], loc = 0, scale = pars["sigma"].values[0]))
        df["prob_1"] = stats.norm.cdf(pars["a_2"].values[0] - df["decision_value"], loc = 0, scale = pars["sigma"].values[0])

        df["EV_R"] = ((df["prob_4"] * 4 + df["prob_3"] * 3 + df["prob_2"] * 2 + df["prob_1"] * 1) - 1) / 3
        df["accept_pred"] = df["prob_4"] + df["prob_3"] > 0.5
        df["response_pred"] = np.argmax(df[["prob_1", "prob_2", "prob_3", "prob_4"]], axis = 1) + 1
    elif model_description[1] == "binary":
        df["prob_1"] = 1 - stats.norm.cdf(- df["decision_value"], loc = 0, scale = pars["sigma"].values[0])
        df["prob_0"] = stats.norm.cdf(- df["decision_value"], loc = 0, scale = pars["sigma"].values[0])
        df["EV_R"] = (df["prob_1"] * 1 + df["prob_0"] * 0) 
        df["accept_pred"] = df["prob_1"] > 0.5
    
    return df

def fit_model(df, negFunc, pars0=None, bounds=None, method="L-BFGS-B"):
    """
    Fit the model to the data.

    """
    if bounds is None:
        bounds = ((0.1, 5), (0, 10), (0.1, None), (None, 0.1), (None, None), (0.1, 10))
    if pars0 is None:
        # defaults based loosely on prior papers
        pars0 = [0.9, 1, 10, -2, 0.5, 1]

    minimizer_kwargs = {"method": method, 
                        "args": (df,), 
                        "bounds": bounds,
                        "tol":1e-8}
    print("Going into basinhop", flush = True) 
    output = basinhopping(
        func = negFunc,
        x0 = pars0,
        niter = 50,
        minimizer_kwargs = minimizer_kwargs,
        disp = True,
    )
    if output.success:
        return output.x, output.fun
    else:
        print(f"Subject {df['sub'].unique()[0]} failed!")
        raise RuntimeError(output.message)

if __name__ == "__main__":
    # load data
    df = load_behavioral_data()
    df = df.query("RT > 0.2")
    
    param_names = ["alpha", "lambd", "a_1", "a_2", "c", "sigma"]
    params = {"param_names" : [], 
              "estimates" : [], 
              "sub" : [], 
              "condition" : []
    }
    for sub in df["sub"].unique()[:5]:
        # Extract data for subject
        print(f"Fitting subject {sub}")
        sub_df = df.query(f"sub == {sub}").reset_index(drop=True)
        condition = sub_df["condition"].unique()[0]

        # fit model
        pars, _ = fit_model(sub_df)
        if pars is not None:
            params["param_names"].extend(param_names)
            params["estimates"].extend(pars)
            params["sub"].extend([sub] * len(param_names))
            params["condition"].extend([condition] * len(param_names))
            print(f"Subject {sub} fitted successfully!")
    
    # save parameters
    params_df = pd.DataFrame(params)
    params_df.to_csv("params.csv", index=False)
