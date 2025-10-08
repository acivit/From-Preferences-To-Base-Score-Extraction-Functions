import matplotlib.pyplot as plt
import plotly.graph_objs as go
import streamlit.components.v1 as components
import json
import itertools
import random
import pandas as pd
import seaborn as sns
import os 

# Make plots look nicer
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

import numpy as np
from sklearn.metrics import cohen_kappa_score

# from statsmodels.stats.inter_rater import fleiss_kappa

from src.argumentation_framework import ArgumentationFramework


def generate_preference_orders(values, relations):
    perm = random.sample(values, len(values))
    expression = ""
    expression += f"{perm[0][0]}={perm[0][1]}"
    first_change = random.choice(relations)
    second_change = random.choice(relations)
    if first_change == ">>" and second_change == ">>":
        first_change = ">"
        second_change = ">"
    elif first_change == ">>" and second_change == "=":
        first_change = ">"
        second_change = "="
    elif first_change == "=" and second_change == ">>":
        first_change = "="
        second_change = ">"
    elif first_change == "=" and second_change == "=":
        return "a=b=c=d=e=f"
    expression += first_change
    expression += f"{perm[1][0]}={perm[1][1]}"
    expression += second_change
    expression += f"{perm[2][0]}={perm[2][1]}"
    return expression


if __name__ == "__main__":
    random.seed(42)
    arg_file = """dec(fast)
dec(slow)
arg(a)
arg(b)
arg(c)
arg(d)
arg(e)
arg(f)
att(a, slow)
sup(b, slow)
sup(c, b)
att(d, b)
sup(e, d)
sup(d, fast)
att(f, fast)"""

    possible_ratios = [0.20, 0.25, 0.33, 0.5]
    arg_file =  arg_file.splitlines()
    int_a = 0
    int_b = 0
    tmax = 1
    tmin = 0
    centralisation = True
    outputs = {
        "ordering": [],
        "tmax": [],
        "tmin": [],
        "ratio": [],
        "QE": [],
        "EB": [],
        "DF": [],
        "EBT": [],
    }

    old_orders = []

    results_for_orderings = {}
    full_results = {}
    df_results = pd.DataFrame(
        columns=["ordering", "tmax", "tmin", "ratio", "output_QE", "output_EB", "output_EBT", "output_DF"]
    )
    list_results = []

    for i in range(30000):

        ordering = generate_preference_orders(
            [("c", "f"), ("b", "e"), ("a", "d")], ["=", ">", ">>"]
        )

        if ordering not in results_for_orderings:
            results_for_orderings[ordering] = {
                "QE": [],
                "EB": [],
                "EBT": [],
                "DF": [],
            }

            full_results[ordering] = {
                "QE": [],
                "EB": [],
                "EBT": [],
                "DF": [],
            }

        if centralisation:
            tmin = round(random.uniform(0, 0.5),2)
            tmax = round(1 - tmin,2)
        else:
            tmin = round(random.uniform(0, 1.0),2)
            tmax = round(random.uniform(tmin, 1.0),2)

        if tmax>=1:
            tmax=1
        if tmin<0:
            tmin=0

        ratio = random.choice(possible_ratios)

        af = ArgumentationFramework(
            scoring="from_rank_edged",
            small_gap=1,
            large_gap=1 / ratio,
        )
        af.generate_af_from_file(arg_file)
        values = af.get_values_base_scores_from_order(
            ordering, a=int_a, b=int_b, tmax=tmax, tmin=tmin
        )

        # scoring = "from_rank_edged_a_b"

        af.modify_arguments_base_scores_without_modifying_file(args_list=values.items())

        labels = list(values.keys())
        initial_positions = list(values.values())

        semantics = [
            "QuadraticEnergy_Model",
            "EulerBased_model",
            "DFQuAD_model",
            "EulerBasedTop_model",
            # "SquaredDFQuAD_model",
            # "EulerBasedTop_model",
        ]

        outputs["ordering"].append(ordering)
        outputs["tmax"].append(tmax)
        outputs["tmin"].append(tmin)
        outputs["ratio"].append(ratio)
        partial_results = []
        outputs_local = {"QE": [], "EB": [], "DF": [], "EBT": []}
        for sem in semantics:
            final_strengths = af.get_final_strengths(semantics=sem).final_strengths
            decision = [None, 0]
            for arg, strength in final_strengths.items():
                if arg not in af.decisions:
                    continue
                if round(strength, 4) > decision[1]:
                    decision = [arg, round(strength, 4)]
                elif round(strength, 4) == decision[1]:
                    decision = ["Tie", round(strength, 4)]
                else:
                    pass

            if sem == "QuadraticEnergy_Model":
                outputs["QE"].append(decision)
                outputs_local["QE"] = decision
                results_for_orderings[ordering]["QE"].append(decision[0])
                full_results[ordering]["QE"].append((tmin, tmax, ratio, decision[0]))
            elif sem == "EulerBased_model":
                outputs["EB"].append(decision)
                outputs_local["EB"] = decision
                results_for_orderings[ordering]["EB"].append(decision[0])
                full_results[ordering]["EB"].append((tmin, tmax, ratio, decision[0]))
            elif sem == "EulerBasedTop_model":
                outputs["EBT"].append(decision)
                outputs_local["EBT"] = decision
                results_for_orderings[ordering]["EBT"].append(decision[0])
                full_results[ordering]["EBT"].append((tmin, tmax, ratio, decision[0]))
            elif sem == "DFQuAD_model":
                outputs["DF"].append(decision)
                outputs_local["DF"] = decision
                results_for_orderings[ordering]["DF"].append(decision[0])
                full_results[ordering]["DF"].append((tmin, tmax, ratio, decision[0]))

            # df_results = pd.concat([df_results, pd.DataFrame({
            #         "ordering": [ordering],
            #         "tmax": [tmax],
            #         "tmin": [tmin],
            #         "ratio": [ratio],
            #         "gradual_semantics": [sem],
            #         "gradual_semantics": [decision[0]]
            #     })], ignore_index=True)

            
            
            partial_results.append([ordering, tmax, tmin, ratio, sem, decision[0]])
        
        list_results.append([ordering, tmax, tmin, ratio, outputs_local["QE"][0], outputs_local["EB"][0], outputs_local["EBT"][0], outputs_local["DF"][0]])

        if partial_results[0][-1] != partial_results[-1][-1]:
            print(f"Different decisions in one ordering! ")
            for result in partial_results:
                print(result)
            print("Ue")

        # if ordering == "b=e>>c=f>a=d" and sem == "DFQuAD_model" and decision[0] == "slow":
        #     print(
        #         f"Found one! {i} -- tmin: {tmin}, tmax: {tmax}, ratio: {ratio}"
        #     )
        #     print("Check")

        # if ordering == "b=e>c=f>a=d" and sem == "DFQuAD_model" and decision[0] == "slow":
        #     print(
        #         f"Found one! {i} -- tmin: {tmin}, tmax: {tmax}"
        #     )
        #     print("Check")

    df_results = pd.DataFrame(
        list_results,
        columns=["ordering", "top", "bottom", "ratio", "output_QE", "output_EB", "output_EBT", "output_DF"],
    )

    # Easier process...

    # Example predictions
    y1 = np.array(outputs["QE"])[:, 0]  # predictions from model 1
    y2 = np.array(outputs["EB"])[:, 0]  # predictions from model 2
    y3 = np.array(outputs["DF"])[:, 0]  # predictions from model 3
    y4 = np.array(outputs["EBT"])[:, 0]  # predictions from model 4

    # Pairwise agreement
    pairwise_agreement_12 = np.mean(y1 == y2)
    pairwise_agreement_13 = np.mean(y1 == y3)
    pairwise_agreement_23 = np.mean(y2 == y3)
    pairwise_agreement_14 = np.mean(y1 == y4)
    pairwise_agreement_24 = np.mean(y2 == y4)
    pairwise_agreement_34 = np.mean(y3 == y4)

    print(f"Pairwise agreement QE-EB: {pairwise_agreement_12:.2f}")
    print(f"Pairwise agreement QE-DF: {pairwise_agreement_13:.2f}")
    print(f"Pairwise agreement EB-DF: {pairwise_agreement_23:.2f}")
    print(f"Pairwise agreement QE-EBT: {pairwise_agreement_14:.2f}")
    print(f"Pairwise agreement EB-EBT: {pairwise_agreement_24:.2f}")
    print(f"Pairwise agreement DF-EBT: {pairwise_agreement_34:.2f}")

    # Cohen's Kappa for pairs
    print(f"Cohen's Kappa QE-EB: {cohen_kappa_score(y1, y2):.2f}")
    print(f"Cohen's Kappa QE-DF: {cohen_kappa_score(y1, y3):.2f}")
    print(f"Cohen's Kappa EB-DF: {cohen_kappa_score(y2, y3):.2f}")
    print(f"Cohen's Kappa QE-EBT: {cohen_kappa_score(y1, y4):.2f}")
    print(f"Cohen's Kappa EB-EBT: {cohen_kappa_score(y2, y4):.2f}")
    print(f"Cohen's Kappa DF-EBT: {cohen_kappa_score(y3, y4):.2f}")

    # Create distribution plots for each design choice
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Distribution of Design Choices by Decision (Slow/Fast)")

    # Process data for each semantics

    for ordering, results in results_for_orderings.items():
        print(f"Ordering: {ordering}")
        qe_fast = results["QE"].count("fast")
        qe_slow = results["QE"].count("slow")
        percentage_fast = (
            qe_fast / (qe_fast + qe_slow) * 100 if (qe_fast + qe_slow) > 0 else 0
        )
        percentage_slow = (
            qe_slow / (qe_fast + qe_slow) * 100 if (qe_fast + qe_slow) > 0 else 0
        )
        print(
            f"\tQE fast: {qe_fast} ({percentage_fast:.2f}%), QE slow: {qe_slow} ({percentage_slow:.2f}%)"
        )
        if qe_fast > 10 and qe_slow > 10:
            for tmin, tmax, ratio, decision in full_results[ordering]["QE"]:
                tmin_slow = []
                tmax_slow = []
                ratio_slow = []
                tmin_fast = []
                tmax_fast = []
                ratio_fast = []
                if decision == "slow":
                    tmin_slow.append(tmin)
                    tmax_slow.append(tmax)
                    ratio_slow.append(ratio)
                elif decision == "fast":
                    tmin_fast.append(tmin)
                    tmax_fast.append(tmax)
                    ratio_fast.append(ratio)

            # Plot distributions
            params = [
                (tmin_slow, tmin_fast, "bottom"),
                (tmax_slow, tmax_fast, "top"),
                (ratio_slow, ratio_fast, "ratio"),
            ]

            for param_idx, (slow_data, fast_data, param_name) in enumerate(params):
                ax = axes[param_idx, 0]  # Slow distribution
                ax.hist(slow_data, bins=20, alpha=0.7, color="blue", density=True)
                ax.set_title(f"QE - {param_name} (Slow)")
                ax.set_xlabel(param_name)
                ax.set_ylabel("Density")

                ax = axes[param_idx, 1]  # Fast distribution
                ax.hist(fast_data, bins=20, alpha=0.7, color="red", density=True)
                ax.set_title(f"QE - {param_name} (Fast)")
                ax.set_xlabel(param_name)
                ax.set_ylabel("Density")

            plt.tight_layout()
            plt.savefig(f"results/design_choices_distributions_{ordering}.png")

        eb_fast = results["EB"].count("fast")
        eb_slow = results["EB"].count("slow")
        percentage_fast = (
            eb_fast / (eb_fast + eb_slow) * 100 if (eb_fast + eb_slow) > 0 else 0
        )
        percentage_slow = (
            eb_slow / (eb_fast + eb_slow) * 100 if (eb_fast + eb_slow) > 0 else 0
        )
        print(
            f"\tEB fast: {eb_fast} ({percentage_fast:.2f}%), EB slow: {eb_slow} ({percentage_slow:.2f}%)"
        )

        ebt_fast = results["EBT"].count("fast")
        ebt_slow = results["EBT"].count("slow")
        percentage_fast = (
            ebt_fast / (ebt_fast + ebt_slow) * 100 if (ebt_fast + ebt_slow) > 0 else 0
        )
        percentage_slow = (
            ebt_slow / (ebt_fast + ebt_slow) * 100 if (ebt_fast + ebt_slow) > 0 else 0
        )
        print(
            f"\tEBT fast: {ebt_fast} ({percentage_fast:.2f}%), EBT slow: {ebt_slow} ({percentage_slow:.2f}%)"
        )

        df_fast = results["DF"].count("fast")
        df_slow = results["DF"].count("slow")
        percentage_fast = (
            df_fast / (df_fast + df_slow) * 100 if (df_fast + df_slow) > 0 else 0
        )
        percentage_slow = (
            df_slow / (df_fast + df_slow) * 100 if (df_fast + df_slow) > 0 else 0
        )
        print(
            f"\tDF fast: {df_fast} ({percentage_fast:.2f}%), DF slow: {df_slow} ({percentage_slow:.2f}%)"
        )


    if not os.path.exists("results"):
        os.mkdir("results")
    df_results.to_csv("results/design_choices_full_results.csv")

    df_results_differences = df_results[
        (df_results["output_QE"] != df_results["output_EB"])
        | (df_results["output_QE"] != df_results["output_DF"])
        | (df_results["output_EB"] != df_results["output_DF"])
        | (df_results["output_QE"] != df_results["output_EBT"])
        | (df_results["output_EB"] != df_results["output_EBT"])
        | (df_results["output_DF"] != df_results["output_EBT"])
    ]

    df_results_differences.sort_values(by="ordering", inplace=True)
    df_results_differences.to_csv("results/design_choices_differences.csv", index=False)

    
    df_results_big_differences = df_results_differences[df_results_differences["top"]-df_results_differences["bottom"]>=0.4]
    df_results_big_differences = df_results_big_differences.reset_index(drop=True)
    df_results_big_differences.to_csv("results/design_choices_big_differences.csv", index=False)
    # Encode output as binary if needed
    # df_results["output_binary"] = df_results["output"].map({"slow": 0, "fast": 1})

    # df = df_results
    # # --- 3. Descriptive stats ---
    # print("\n=== Descriptive Statistics by Output ===")
    # print(df.groupby("output")[["top", "bottom", "ratio"]].describe())

    # print("\n=== Counts by Criteria and Output ===")
    # print(pd.crosstab(df["gradual_semantics"], df["output"]))

    # 4.1 Boxplots: variable distributions by output
    # for var in ["top", "bottom", "ratio"]:
    #     plt.figure(figsize=(6, 4))
    #     sns.boxplot(x="output", y=var, data=df)
    #     plt.title(f"{var} by Output")
    #     plt.show()

    #     # 4.2 Scatter plot: top vs bottom, colored by output, faceted by criteria
    #     plt.figure(figsize=(8, 6))
    #     sns.scatterplot(
    #         x="top", y="bottom", hue="output", style="gradual_semantics", data=df, s=100
    #     )
    #     plt.title("Top vs Bottom by Output and Gradual Semantics")
    #     plt.show()

    #     # 4.3 Bar chart: counts of fast/slow by criteria
    #     plt.figure(figsize=(6, 4))
    #     sns.countplot(x="gradual_semantics", hue="output", data=df)
    #     plt.title("Output Counts by Gradual Semantics")
    #     plt.show()

    #     # 4.4 Heatmap: correlation between numeric variables and output
    #     corr = df[["top", "bottom", "ratio", "output_binary"]].corr()
    #     plt.figure(figsize=(6, 4))
    #     sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    #     plt.title("Correlation Heatmap")
    #     plt.show()

    #     # 4.5 Pairplot: Explore all numeric variables, colored by output
    #     sns.pairplot(df, vars=["top", "bottom", "ratio"], hue="output", diag_kind="kde")
    #     plt.suptitle("Pairwise Relationships", y=1.02)
    #     plt.show()
