import scipy.stats as stats


# T-Test independence test with p-value
def t_test_ind(
    params,
    param_name="param_name",
    param_value="param_value",
    condition_name="condition",
    conditions=["equalRange", "equalIndifference"],
):
    p_values, t_stats = [], []
    for param in params[param_name].unique():
        ttest = stats.ttest_ind(
            params.query(
                f"{param_name} == '{param}' & {condition_name} == '{conditions[0]}'"
            )[param_value],
            params.query(
                f"{param_name} == '{param}' & {condition_name} == '{conditions[1]}'"
            )[param_value],
        )
        p_values.append(ttest.pvalue)
        t_stats.append(ttest.statistic)

    p_values = stats.false_discovery_control(p_values, method="by")
    for i, param in enumerate(params[f"{param_name}"].unique()):
        if p_values[i] < 0.05:
            print("Significant difference for", param)
            print("t_stats: ", t_stats[i], "p_values: ", p_values[i])
            print()
        else:
            print("No significant difference for", param)
            print("t_stats: ", t_stats[i], "p_values: ", p_values[i])
            print()
