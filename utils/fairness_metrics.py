from .plot import plot_bar


def statistical_parity_difference(data, target_column, protected_column):
    """
    Statistical parity difference calculation and plot
    :param data: Pandas Dataframe
    :param protected_column: column corresponding to protected variable
    :param target_column: Output variable
    :return:
    """

    # protected column values
    column_values = data[protected_column].unique()

    privileged_group = data[protected_column].value_counts().keys()[0]
    print(f"Privileged Group: {privileged_group}")

    # compute statistical parity for each value of the protected variable
    parities = {}
    for val in column_values:
        parity = data[target_column].mean() - data[data[protected_column] == val][target_column].mean()
        print(f"Statistical parity P(Y_hat = 1) - P(Y_hat= 1| {protected_column}={val}): {parity}")
        parities[val] = parity

    # compute statistical parity difference which is privileged_group - other_group
    diff_parities  = {}
    for key, val in parities.items():

        if key != privileged_group:
            parity = parities[key] - parities[privileged_group]
            diff_parities[f"{privileged_group} -> {key}"] = parity
            print(f"Statistical parity P(Y_hat = 1|{protected_column}={privileged_group}) "
                  f"- P(Y_hat= 1|{protected_column}={key}): {parity}")

    plot_bar(diff_parities.keys(), diff_parities.values(), "Statistical Parity Difference")


def equality_of_opportunity(data, predicted_column, label_column, protected_column):
    """
    Equality of opportunnity difference calculation and plot
    :param data: Pandas Dataframe
    :param protected_column: column corresponding to protected variable
    :param predicted_column: Estimated output variable
    :param label_column: Output variable
    :return:
    """

    # protected column values
    column_values = data[protected_column].unique()
    pop_opp = data[data[label_column] == 1][predicted_column].mean()

    privileged_group = data[protected_column].value_counts().keys()[0]
    print(f"Privileged Group: {privileged_group}")

    # compute opportunity for each value of the protected variable
    opps = {}
    for val in column_values:
        prot_opp = data[data[protected_column] == val]
        prot_opp = prot_opp[prot_opp[label_column] == 1][predicted_column].mean()
        opp = pop_opp - prot_opp
        print(f"Opportunity P(Y_hat = 1| Y=1) - P(Y_hat = 1| Y=1, {protected_column}={val}): {opp}")
        opps[val] = opp

    # compute opportunity difference which is privileged_group - other_group
    diff_opps = {}
    for key, val in opps.items():

        if key != privileged_group:
            opp = opps[key] - opps[privileged_group]
            diff_opps[f"{privileged_group} -> {key}"] = opp
            print(f"Statistical parity P(Y_hat = 1| Y=1, {protected_column}={privileged_group}) "
                  f"- P(Y_hat= 1| Y=1, {protected_column}={key}): {opp}")

    plot_bar(diff_opps.keys(), diff_opps.values(), "Equal Opportunity Difference")


