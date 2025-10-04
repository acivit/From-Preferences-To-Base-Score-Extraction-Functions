from qbaf import QBAFramework, QBAFARelations
from qbaf_visualizer.Visualizer import visualize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from src.argumentation_framework import ArgumentationFramework

import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from decimal import Decimal
import os


def plot_interpolated_grid(x, y, values, sem, grid_resolution=100):
    x = np.array(x)
    y = np.array(y)
    values = np.array(values)

    # Set up grid based on the range of x and y
    grid_x, grid_y = np.mgrid[
        x.min() : x.max() : complex(grid_resolution),
        y.min() : y.max() : complex(grid_resolution),
    ]

    # Interpolate
    grid_z = griddata((x, y), values, (grid_x, grid_y), method="cubic")

    # Plot
    plt.figure(figsize=(6, 5))
    sc = plt.imshow(
        grid_z.T,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    # plt.scatter(x, y, c='red', label='Original Points')
    plt.colorbar(sc, label="Differential Strength")
    plt.title("Interpolation of Aggregation Influence")
    plt.xlabel("Base Score")
    plt.ylabel("Final Strength")
    plt.legend()
    plt.savefig(f"compared_influences/aggregation_influence_{sem[:3]}.png")
    plt.close()


def plot_mesh(base_scores, strengths, aggregations, diff_strengths, sem):

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # ax[0].contourf(base_scores, strengths, diffs_list, 100)
    # ax[1].pcolormesh(base_scores, strengths, diffs_list, shading='gouraud')
    # plt.show()
    # Option 1: smoother filled contours
    c1 = ax[0].tricontourf(
        base_scores, strengths, diff_strengths, levels=100, cmap="viridis"
    )
    c2 = ax[1].tricontourf(
        base_scores, aggregations, diff_strengths, levels=100, cmap="viridis"
    )

    ax[0].set_ylabel("Final strength")
    ax[1].set_ylabel("Aggregations")
    ax[0].set_xlabel("Base Score")
    ax[1].set_xlabel("Base Score")

    # Title for the whole figure
    fig.suptitle(
        "Sensibility of the final strength for different base scores and aggregations"
    )

    # Shared colorbar on the right
    cbar = fig.colorbar(c1, ax=ax, orientation="vertical", fraction=0.025, pad=0.04)
    cbar.set_label("Differential final strength")

    plt.savefig(
        f"compared_influences/color_plot_aggregation_influence_{sem[:3]}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    c = plt.tricontourf(
        base_scores, aggregations, strengths, levels=100, cmap="viridis"
    )
    plt.xlabel("Base score")
    plt.ylabel("Aggregation")
    cbar = plt.colorbar(c, orientation="vertical", fraction=0.025, pad=0.04)
    cbar.set_label("Final strength")
    plt.savefig(
        f"compared_influences/color_plot_aggregation_strength_{sem[:3]}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


def compute_aggregation_and_final_strengths(semantics, af, n_args):

    print("Semantic: ", sem)
    base_score_to_strength = {}
    aggregation_to_strength = {}
    aggregation_to_strength_over_score = {}

    for base_score in base_scores:
        af.generate_af_from_file(arg_file)
        for agg in aggregations:
            if float(agg) <= 0.0005 and float(agg) >= -0.0005:
                af.activate_arguments(["e", "D1", "CR1"])
                weights = [["e", base_score], ["D1", 0], ["CR1", 0]]
            else:
                if agg > 0:
                    if n_args == 1:
                        number_of_arguments = int(agg) + 1
                        supports = supporting_arguments[:number_of_arguments]
                        af.activate_arguments(["e"] + supports)
                        cum_agg = agg.copy()
                        weights = [["e", base_score]]
                        for arg in supports:
                            if cum_agg > 1:
                                weights.append([arg, 1])
                                cum_agg -= 1
                            else:
                                weights.append([arg, cum_agg])
                                cum_agg = 0
                        
                    else:
                        supports = supporting_arguments[:2]
                        af.activate_arguments(["e"] + supports)
                        cum_agg = agg.copy()
                        weights = [["e", base_score]]
                        for arg in supports:
                            if cum_agg > 0.5:
                                weights.append([arg, 0.5])
                                cum_agg -= 0.5
                            else:
                                weights.append([arg, cum_agg])
                                cum_agg = 0
                else:
                    if n_args == 1:
                        number_of_arguments = int(-agg) + 1
                        attacks = attacking_arguments[:number_of_arguments]
                        af.activate_arguments(["e"] + attacks)
                        cum_agg = agg.copy()
                        weights = [["e", base_score]]
                        for arg in attacks:
                            if cum_agg < -1:
                                weights.append([arg, 1])
                                cum_agg += 1
                            else:
                                weights.append([arg, -cum_agg])
                                cum_agg = 0
                    else:
                        attacks = attacking_arguments[:2]
                        af.activate_arguments(["e"] + attacks)
                        cum_agg = agg.copy()
                        weights = [["e", base_score]]

                        for arg in attacks:
                            if cum_agg < -0.5:
                                weights.append([arg, 0.5])
                                cum_agg += 0.5
                            else:
                                weights.append([arg, -cum_agg])
                                cum_agg = 0

                    # weights = [["e", base_score], ["CR1", -agg]]

            af.modify_arguments_weights_without_modifying_file(weights)
            # af.change_argument_weight(arg_file, [("t_2_1", 1), ("t_1_1", 1)])
            # weight = 2.2
            # af.set_all_argument_weights_to_value(weight)
            qbaf = af.get_final_strengths(semantics=semantics)
            final_strengths_repeat = qbaf.final_strengths
        

            if base_score not in base_score_to_strength.keys():
                base_score_to_strength[base_score] = {
                    agg: final_strengths_repeat["e"]
                }
            else:
                base_score_to_strength[base_score][agg] = final_strengths_repeat[
                    "e"
                ]

            if agg not in aggregation_to_strength_over_score:
                aggregation_to_strength_over_score[agg] = {
                    float(
                        final_strengths_repeat["e"] / base_score
                    ): final_strengths_repeat["e"]
                }
                aggregation_to_strength[agg] = {
                    base_score: final_strengths_repeat["e"]
                }
            else:
                aggregation_to_strength_over_score[agg][
                    float(final_strengths_repeat["e"] / base_score)
                ] = final_strengths_repeat["e"]
                aggregation_to_strength[agg][base_score] = final_strengths_repeat[
                    "e"
                ]

    return base_score_to_strength, aggregation_to_strength, aggregation_to_strength_over_score

def compute_lines_to_plot(base_score_to_strength):

    base_scores = []
    base_scores_unique = []
    diffs = {}
    strengths = []
    diffs_list = []
    aggregations_list = []
    aggregations_unique = []

    aggregations_lines_dict = {}
    # Scatter plot

    for base_score, aggregations in base_score_to_strength.items():
        for i in range(len(aggregations.keys()) - 1):

            current_value = list(aggregations.keys())[i]
            next_value = list(aggregations.keys())[i + 1]

            if sem == "DFQuAD_model":

                if Decimal(str(round(current_value, 2))) % Decimal("0.2")== Decimal("0.0") and abs(current_value) < 2:

                    if round(current_value, 2) not in aggregations_lines_dict:
                        print("Adding: ", round(current_value, 2))
                        aggregations_lines_dict[round(current_value, 2)] = {
                            "base_score": [base_score],
                            "strength": [
                                base_score_to_strength[base_score][current_value]
                            ],
                        }
                    else:
                        aggregations_lines_dict[round(current_value, 2)][
                            "base_score"
                        ].append(base_score)
                        aggregations_lines_dict[round(current_value, 2)][
                            "strength"
                        ].append(base_score_to_strength[base_score][current_value])

            else:
                #if round(current_value, 2) == 1.0:
                #    print("Ue")
                if Decimal(str(round(current_value, 2))) % Decimal("0.2")== Decimal("0.0"):
                    if round(current_value, 2) not in aggregations_lines_dict:
                        print("Adding: ", round(current_value, 2))
                        aggregations_lines_dict[round(current_value, 2)] = {
                            "base_score": [base_score],
                            "strength": [
                                base_score_to_strength[base_score][current_value]
                            ],
                        }
                    else:
                        aggregations_lines_dict[round(current_value, 2)][
                            "base_score"
                        ].append(base_score)
                        aggregations_lines_dict[round(current_value, 2)][
                            "strength"
                        ].append(base_score_to_strength[base_score][current_value])
            diff = (
                base_score_to_strength[base_score][next_value]
                - base_score_to_strength[base_score][current_value]
            ) / (next_value - current_value)
            if base_score not in diffs:
                diffs[base_score] = []
            diffs[base_score].append(diff)
            diffs_list.append(diff)
            base_scores.append(base_score)
            strengths.append(base_score_to_strength[base_score][current_value])
            aggregations_list.append(current_value)

    return base_scores, strengths, aggregations_list, diffs_list, aggregations_lines_dict

# Function to crop colormap to avoid extremes
def crop_cmap(cmap, min_val=0.2, max_val=0.8, n=256):
    return ListedColormap(cmap(np.linspace(min_val, max_val, n)))


def plot_everything(base_scores, strengths, aggregations_list, diffs_list, aggregations_lines_dict, title_extra=""):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=base_scores,
                y=aggregations_list,
                z=strengths,
                mode="markers",
                marker=dict(
                    size=5,
                    color=diffs_list,  # Set color to values in diffs_list
                    colorscale="Viridis",  # Choose a colorscale
                    colorbar=dict(title="Diff"),  # Optional colorbar
                    opacity=0.8,
                ),
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Base Score",
            yaxis_title="Aggregation",
            zaxis_title=f"Final Strength {sem}",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="Interactive 3D Scatter Plot",
    )

    # Save interactive HTML
    if os.path.exists("interactive_web_plots") is False:
        os.mkdir("interactive_web_plots")
    fig.write_html(f"interactive_web_plots/3d_scatter_{sem}_{title_extra}.html")

    plt.figure(figsize=(8,3))
    plt.xlim(-0.15, 1.15)
    norm_pos = Normalize(vmin=0, vmax=1)   # for positive values
    norm_neg = Normalize(vmin=0, vmax=1)  # for negative values
    cmap_pos = crop_cmap(plt.cm.Blues, 0.3, 0.8)  # Blue in moderate range
    cmap_neg = crop_cmap(plt.cm.Reds, 0.3, 0.8)   # Red in moderate range
    for aggregation, aggregation_line in aggregations_lines_dict.items():
        
        if aggregation > 0:
            color = cmap_pos(norm_pos(abs(aggregation)))
        elif aggregation < 0:
            color = cmap_neg(norm_neg(abs(aggregation)))
        else:
            color = "grey"
            
        
        plt.plot(
            aggregation_line["base_score"],
            aggregation_line["strength"],
            color=color,
        )

        if sem == "QuadraticEnergy_model":
            limit = 3.5
            offset = 0
        elif sem == "DFQuAD_model":
            limit = 1.1
            offset = 0
        elif sem == "EulerBased_model":
            limit = 6
            offset = 0.25
        else:
            limit = 2
            offset = 0.05
        if abs(aggregation) < limit:
            if sem == "QuadraticEnergy_model" or sem == "DFQuAD_model":
                if aggregation > 0:
                    plt.text(
                        -0.03,
                        aggregation_line["strength"][0],  # + aggregation * offset,
                        f"$\Sigma$={aggregation}",
                        fontsize=8,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                    )

                elif aggregation < 0:
                    plt.text(
                        1.1,
                        aggregation_line["strength"][-1],  # + aggregation * offset,
                        f"$\Sigma$={aggregation}",
                        fontsize=8,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                    )
                else:
                    plt.text(
                        0.52,
                        0.53,
                        "$\Sigma$=0",
                        fontsize=8,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                    )

            elif sem == "EulerBased_model":
                if aggregation > 0:
                    plt.text(
                        -0.03,
                        aggregation_line["strength"][0] + aggregation * offset,
                        f"$\Sigma$={aggregation}",
                        fontsize=8,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                    )

                elif aggregation < 0:
                    plt.text(
                        1.1,
                        aggregation_line["strength"][-1] + aggregation * offset,
                        f"$\Sigma$={aggregation}",
                        fontsize=8,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                    )
                else:
                    plt.text(
                        0.52,
                        0.53,
                        "$\Sigma$=0",
                        fontsize=8,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                    )
            else:
                plt.text(
                    0.48,
                    0.48,
                    "$\Sigma$=0",
                    fontsize=8,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )
                plt.text(
                    0.34,
                    0.55,
                    "$\Sigma$=>1",
                    fontsize=8,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )
                plt.text(
                    0.42,
                    0.51,
                    "$\Sigma$=0.5",
                    fontsize=8,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )
                plt.text(
                    0.54,
                    0.43,
                    "$\Sigma$=-0.5",
                    fontsize=8,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )
                plt.text(
                    0.65,
                    0.39,
                    "$\Sigma$=<-1",
                    fontsize=8,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )

    

    plt.xlabel(r"Base Score $(\tau)$")
    mod = ""

    if sem == "QuadraticEnergy_model":
        mod = "Quadratic Energy"
    elif sem == "DFQuAD_model":
        mod = "DFQuAD"
    elif sem == "EulerBased_model":
        mod = "Euler Based"
    elif sem == "EulerBasedTop_model":
        mod = "Euler Based Top"
    else:
        pass
    plt.title(mod)
    plt.ylabel("Final Strength $(\sigma)$")  # +mod)
    if os.path.exists("compared_influences") is False:
        os.mkdir("compared_influences")
    plt.savefig(
        f"compared_influences/base_score_final_strength_{sem}_{title_extra}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

if __name__ == "__main__":

    #######################################################################
    af = ArgumentationFramework()
    af_multiple = ArgumentationFramework()

    arg_file = """arg(e, 0.5)
arg(D1, 0.5)
arg(D2, 0.5)
arg(D3, 0.5)
arg(D4, 0.5)
arg(D5, 0.5)
arg(D6, 0.5)
arg(D7, 0.5)
arg(D8, 0.5)
arg(CR1, 0.5)
arg(CR2, 0.5)
arg(CR3, 0.5)
arg(CR4, 0.5)
arg(CR5, 0.5)
arg(CR6, 0.5)
arg(CR7, 0.5)
arg(CR8, 0.5)

sup(D1, e)
sup(D2, e)
sup(D3, e)
sup(D4, e)
sup(D5, e)
sup(D6, e)
sup(D7, e)
sup(D8, e)
att(CR1, e)
att(CR2, e)
att(CR3, e)
att(CR4, e)
att(CR5, e)
att(CR6, e)
att(CR7, e)
att(CR8, e)"""

    arg_file =  arg_file.splitlines()

    arguments = ["e"]
    supporting_arguments = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
    attacking_arguments = ["CR1", "CR2", "CR3", "CR4", "CR5", "CR6", "CR7", "CR8"]

    semantics = [
        "QuadraticEnergy_model",
        "EulerBased_model",
        "EulerBasedTop_model",
        "DFQuAD_model",
    ]

    aggregations = np.arange(-1, 1.1, 0.05)
    base_scores = np.arange(0.00, 1.05, 0.05)

    for sem in semantics:

        base_score_to_strength, aggregation_to_strength, aggregation_to_strength_over_score = compute_aggregation_and_final_strengths(sem, af, 1)
        base_score_to_strength_multiple, aggregation_to_strength_multiple, aggregation_to_strength_over_score_multiple = compute_aggregation_and_final_strengths(sem, af_multiple, 2)

        
        base_scores, strengths, aggregations_list, diffs_list, aggregations_lines_dict = compute_lines_to_plot(base_score_to_strength)
        base_scores_multiple, strengths_multiple, aggregations_list_multiple, diffs_list_multiple, aggregations_lines_dict_multiple = compute_lines_to_plot(base_score_to_strength_multiple)

        plot_everything(base_scores, strengths, aggregations_list, diffs_list, aggregations_lines_dict, "single")
        plot_everything(base_scores_multiple, strengths_multiple, aggregations_list_multiple, diffs_list_multiple, aggregations_lines_dict_multiple, "multiple")


        
