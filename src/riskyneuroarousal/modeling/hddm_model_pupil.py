import hddm


# TODO: Make paths come from config file (but also, caring for the installation of package...)
# HDDM runs in a container.

def load_data(subj_ID):
    data = hddm.load_csv(
        "/scratch/users/gustxsr/riskyneuroarousal/data/hddm_pupil_data.csv"
    ).query(f"sub == {subj_ID}")
    data = data.rename(columns={"RT": "rt", "accept": "response", "sub": "subj_idx"})
    data = data.query("rt > 0.2")
    data = hddm.utils.flip_errors(data)
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit a DDM model to a subject's data")
    parser.add_argument(
        "--subj_ID", type=int, help="The subject ID to fit the model to"
    )
    parser.add_argument("--data_pth", type=str, help="The path to the data")
    parser.add_argument(
        "--model", type=str, help="The model to fit the data to (from 1 to 3)"
    )
    parser.add_argument("--results_pth", type=str, help="The path to save the results")
    parser.add_argument("--model_pth", type=str, help="The path to save the model")
    args = parser.parse_args()

    # Load a subjects data (TODO: make dynamic src path)
    data = load_data(int(args.subj_ID))

    
    v_reg = {"model": f"v ~ C(pupil_bin):dir_opposite_prev_response + decision_value_norm", "link_func": lambda x: x}
    reg_descr = [v_reg]
    m_reg = hddm.HDDMnnRegressor(
            data,
            reg_descr,
            model="angle",
            informative=False,
            is_group_model=False,
            include=["v", "a", "t", "z", "theta"],
        )

    m_reg.sample(
        10000,
        burn=2000,
        dbname=f"{args.model_pth}/sub{args.subj_ID}_model{args.model}_SV.db",
        db="pickle",
    )
    stats = m_reg.gen_stats()
    print(stats.head())
    stats.loc["BIC"] = [m_reg.bic, 0, 0, 0, 0, 0, 0, 0]
    stats.to_csv(f"{args.results_pth}/sub{args.subj_ID}_pupil_ddm.csv")
    m_reg.save(f"{args.model_pth}/sub{args.subj_ID}_pupil_ddm.hddm")
