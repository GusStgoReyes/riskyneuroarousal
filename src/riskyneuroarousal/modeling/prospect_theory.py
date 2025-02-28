# taken from https://github.com/poldrack/ResearchMethods/blob/main/src/psych125/prospect_theory.py

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from riskyneuroarousal.utils.load_data import load_behavioral_data
from riskyneuroarousal.config import Config
import os


# Subjective utility using prospect theory equation
def calc_subjective_utility(vals, lam, rho):
    """
    Inputs:
    Returns:
    """
    if vals >= 0:
        retval = vals**rho
    else:
        retval = (-1 * lam) * ((-1 * vals) ** rho)
    return retval


# Subjective value of the prospects
def calc_utility_diff(su_gain, su_loss, su_cert=None):
    """
    Inputs:
    Returns:
    """
    su_gain = np.array(su_gain)
    su_loss = np.array(su_loss)
    if su_cert is None:
        su_cert = np.zeros(len(su_gain))
    else:
        su_cert = np.array(su_cert)
    # gamble subjective utility (su) = .5 times gain subjective utility plus
    # .5 times loss subjective utility. Then take the difference with certain
    return (0.5 * su_gain + 0.5 * su_loss) - su_cert


# Calculate the probability of accepting a gamble,
# given a difference in subjective utility and mu (Equation (3) above)
def calc_prob_accept(gamble_cert_diff, mu):
    return (1 + np.exp(-mu * (gamble_cert_diff))) ** -1


# Log-likelihood
def LL_prospect(par):
    lambda_par, rho_par, mu_par = par
    assert not np.isnan(lambda_par)
    cert_su = [calc_subjective_utility(i, lambda_par, rho_par) for i in data.cert]
    loss_su = [calc_subjective_utility(-1 * i, lambda_par, rho_par) for i in data.loss]
    gain_su = [calc_subjective_utility(i, lambda_par, rho_par) for i in data.gain]
    gamble_cert_diff = calc_utility_diff(gain_su, loss_su, cert_su)
    prob_accept = calc_prob_accept(gamble_cert_diff, mu=mu_par)
    prob_accept = np.clip(prob_accept, 1e-8, 1 - 1e-8)
    # Calculate log likelihood on this slightly altered amount
    log_likelihood_trial = data.accept.values * np.log(prob_accept) + (
        1 - data.accept.values
    ) * np.log(1 - prob_accept)
    LL = -1 * np.sum(log_likelihood_trial)
    if np.isnan(LL):
        raise RuntimeError("LL is nan")
    return LL


# Fit the model
def fit_pt_model(df, pars0=None, bounds=None, method="L-BFGS-B"):
    if bounds is None:
        bounds = ((0, None), (0, None), (1, 1))
    # need to make data global so that it can be accessed by LL_prospect
    global data
    data = df
    if "cert" not in data.columns:
        data["cert"] = 0
    if pars0 is None:
        # defaults based loosely on prior papers
        pars0 = [1.5, 0.9, 1]
    output = minimize(
        LL_prospect,
        pars0,
        method=method,
        tol=1e-8,
        bounds=bounds,
        options={"maxiter": 3000},
    )
    if output.success:
        return output.x, output.fun
    else:
        print(f"Subject {df['sub'].unique()[0]} failed!")
        raise RuntimeError(output.message)


# Create pandas with all outputs
def get_predicted_output(sub_pars, subdata):
    pred_output = []
    for sub, pars in sub_pars.items():
        data = subdata.query(f"sub == {sub}")
        if "cert" not in data.columns:
            data["cert"] = 0
        cert_su = [calc_subjective_utility(i, pars[0], pars[1]) for i in data.cert]
        loss_su = [calc_subjective_utility(-1 * i, pars[0], pars[1]) for i in data.loss]
        gain_su = [calc_subjective_utility(i, pars[0], pars[1]) for i in data.gain]
        gamble_cert_diff = calc_utility_diff(gain_su, loss_su, cert_su)
        prob_accept = calc_prob_accept(gamble_cert_diff, mu=pars[2])
        n_pred_accepted = np.sum(prob_accept > 0.5)
        n_accepted = np.sum(data.accept)
        pred_acc = np.mean((prob_accept > 0.5) == data.accept)
        pred_output.append(
            [n_pred_accepted, n_accepted, pred_acc, sub, pars[0], pars[1], pars[2]]
        )

    return pd.DataFrame(
        pred_output,
        columns=["pred_accept", "accept", "predacc", "sub", "lambda", "rho", "mu"],
    )


def get_prob_accept(sub_pars, subdata):
    prob_accepts = dict()
    for sub, pars in sub_pars.items():
        data = subdata.query(f"sub == {sub}")
        if "cert" not in data.columns:
            data["cert"] = 0
        cert_su = [calc_subjective_utility(i, pars[0], pars[1]) for i in data.cert]
        loss_su = [calc_subjective_utility(-1 * i, pars[0], pars[1]) for i in data.loss]
        gain_su = [calc_subjective_utility(i, pars[0], pars[1]) for i in data.gain]
        gamble_cert_diff = calc_utility_diff(gain_su, loss_su, cert_su)
        prob_accept = calc_prob_accept(gamble_cert_diff, mu=pars[2])
        prob_accepts[sub] = prob_accept
    return prob_accepts


if __name__ == "__main__":
    # Load data
    all_data = load_behavioral_data(min_RT=0.2)

    # Fit model
    sub_pars = {
        "sub": [],
        "condition": [],
        "param_name": [],
        "param_value": [],
        "model": [],
    }

    # Lambda (loss aversion), rho (curvature), mu (temperature)
    # Model 1: All free parameters
    # Model 2: Mu = 1 (no temperature)
    # Model 3: Rho = 1 (no curvature)
    # Model 4: Lambda = 1 (no loss aversion)
    # Model 5: Only Loss aversion
    # Model 6: Only curvature
    # Model 7: Linear utility (no loss aversion or curvature)
    models_bounds = [
        # ((0, None), (0.001, None), (0, None)),
        ((0, None), (0, None), (1, 1)),
        # ((0, None), (1, 1), (0, None)),
        # ((1, 1), (0.001, None), (0, None)),
        # ((0, None), (1, 1), (1, 1)),
        # ((1, 1), (0.001, None), (1, 1)),
        # ((1, 1), (1, 1), (0, None)),
    ]

    for i, bounds in enumerate(models_bounds):
        for sub in all_data["sub"].unique():
            sub_data = all_data.query(f"sub == {sub}")
            pars, loss_value = fit_pt_model(sub_data, bounds=bounds)
            sub_pars["sub"].extend([sub] * 4)
            sub_pars["condition"].extend([sub_data["condition"].values[0]] * 4)
            sub_pars["param_name"].extend(["lambda", "rho", "mu", "loss"])
            sub_pars["param_value"].extend(pars)
            sub_pars["param_value"].append(loss_value)
            sub_pars["model"].extend([f"model_{i + 1}"] * 4)

    sub_pars = pd.DataFrame(sub_pars)

    # Get path to the data from the config file
    config = Config()
    username = config.username
    result_path = os.path.join(config.data_path[f"{username}"], "pt_results.csv")
    sub_pars.to_csv(result_path)
