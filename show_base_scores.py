import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import streamlit.components.v1 as components
import json

from src.argumentation_framework import ArgumentationFramework

# --- Plotting ---
def plot_horizontal_bar(dot_positions, labels):
    fig, ax = plt.subplots(figsize=(8, 2))

    # Draw base bar
    ax.hlines(y=0, xmin=0, xmax=1, color="gray", linewidth=4)

    ax.text(-0.05, 0, "1", color="black")
    ax.text(1.05, 0, "0", color="black")
    # Draw dots
    for i, x in enumerate(dot_positions):
        ax.plot(1 - x, 0, "o", color="blue")
        ax.text(1 - x, 0.1, labels[i], ha="center", fontsize=10)
        ax.text(1 - x, -0.1, round(dot_positions[i], 2), ha="center", fontsize=10)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 0.4)
    ax.axis("off")  # Hide axes

    return fig


# Configure the page to be wide
st.set_page_config(layout="wide")


# --- Streamlit Inputs ---
st.markdown(
    """
    <h1 style='text-align: center; color: #b3c6e0; font-size: 2.8em; font-family: "Segoe UI", Arial, sans-serif; margin-bottom: 0.2em;'>
        Base Score Extraction Function Visualisation
    </h1>
    <hr style='border: 1px solid #e0e6ef; margin-top: 0.2em; margin-bottom: 1.2em;'>
    """,
    unsafe_allow_html=True
)

# Create two main columns for the layout
left_col, right_col = st.columns(2)

with left_col:
    
    st.subheader("Framework Configuration")

    # Large text area for argumentation framework data
    framework_text = st.text_area(
        "Paste argumentation framework data here, please follow the provided template. Introduce the decision arguments as dec(name), the arguments as arg(name), the attacks as att(from,to), and the supports as sup(from,to):",
        """dec(o1)
dec(o2)
arg(a)
arg(b)
arg(c)
arg(d)
arg(e)
arg(f)
att(a, o2)
sup(b, o2)
sup(c, b)
att(d, b)
sup(e, d)
sup(d, o1)
att(f, o1)""",
        height=200
        )

    framework_lines = framework_text.splitlines()
    # Input string in the form a>b>c>>d>e=f

    st.subheader("Preferences Configuration")

    input_str = st.text_input("Enter preferences, being > a preferred relation, = a similar relation, and >> a much preferred relation. (format: a>b>c>>d>e=f):", "a>b>c>>d>e=f")

    arguments_text = []
    for line in framework_lines:
        if line[:3] == "arg":
            arguments_text.append(line[4:-1])

    
    arguments_preferences = []
    for item in input_str:
        if item == '>':
            continue
        elif item == '=':
            continue
        else:
            arguments_preferences.append(item)
        print("Line read from text area: ", line)

    print("arguments_text: ", arguments_text)
    print("arguments_preferences: ", arguments_preferences)
    assert(set(arguments_text) == set(arguments_preferences)), "All input arguments must be in the preference ordering."

    st.subheader("Design Choices for the Base Score Extraction Function")
    # Float input between 0 and 1
    st.latex(r"\text{Select a ratio for } \delta / \Delta")
    input_float = st.slider("δ/Δ ratio:", 0.0, 1.0, 0.5)

    methods = {
        "Play with edges": r" \nu = \tau_{min}+(\tau_{max}-\tau_{min})\cdot\frac{M-r}{M-1}",
        "Play with a b": r" \nu = \frac{M-r+a}{M-1+b}, \ \  a>0, \ \  b>0, \ \  a<b",
    }

    # Layout: two columns, left for radio, right for LaTeX
    col1, col2 = st.columns([1, 2])
    
    with col1:
    
        selected_method = st.radio(
            "Choose base score extraction function:", list(methods.keys())
        )

    with col2:
        # Show LaTeX for each method aligned with radio buttons
        for method_name in methods:
            if method_name == selected_method:
                st.latex(methods[method_name])
            else:
                st.markdown("<br>", unsafe_allow_html=True)  # Keeps spacing aligned

    int_a = 0
    int_b = 0
    tmax = 1
    tmin = 0

    if selected_method == "Play with edges":
        scoring = "from_rank_edged"
        tmax = st.number_input(
            "Max base score", min_value=0.0, max_value=1.0, value=1.0, step=0.01
        )
        tmin = st.number_input(
            "Min base score", min_value=0.0, max_value=1.0, value=0.0, step=0.01
        )

    elif selected_method == "Play with a b":
        scoring = "from_rank_edged_a_b"

        int_a = st.number_input("Enter a (0–10):", min_value=0, max_value=10, value=0)
        int_b = st.number_input("Enter b (0–10):", min_value=0, max_value=10, value=0)
    else:
        scoring = "from_rank"

    af_caregiver = ArgumentationFramework(
        scoring=scoring,
        small_gap=1,
        large_gap=1 / input_float,
    )
    
    af_caregiver.generate_af_from_file(framework_lines, order=input_str)
    values = af_caregiver.get_values_base_scores_from_order(
        input_str, a=int_a, b=int_b, tmax=tmax, tmin=tmin
    )

    af_caregiver.modify_arguments_base_scores_without_modifying_file(
        args_list=values.items()
    )

    labels = list(values.keys())
    initial_positions = list(values.values())

# Show bar with dots
with right_col:

    fig = plot_horizontal_bar(initial_positions, labels)
    st.pyplot(fig)

    semantics = [
        "QuadraticEnergy_model",
        "EulerBased_model",
        "DFQuAD_model",
        # "SquaredDFQuAD_model",
        # "EulerBasedTop_model",
    ]

    st.subheader("Final Strengths and Decisions")

    # Initialize data structures for the table


    print("Decisions in framework: ", af_caregiver.decisions)
    table_data = {
        "Semantics": [],
    }

    for decision in af_caregiver.decisions.keys():
        table_data[decision] = []

    # First pass to collect all unique arguments and initialize columns
    all_decisions = set()
    for sem in semantics:
        final_strengths = af_caregiver.get_final_strengths(
            semantics=sem
        ).final_strengths
        for arg in final_strengths:
            if arg in af_caregiver.decisions:
                all_decisions.add(arg)

    # Initialize columns for each argument
    for arg in sorted(all_decisions):
        table_data[arg] = []

    table_decisions = {sem: None for sem in semantics}
    # Fill the table data
    for sem in semantics:
        final_strengths = af_caregiver.get_final_strengths(
            semantics=sem
        ).final_strengths

        # Get decisions for this semantics
        decisions = {}
        for arg, strength in final_strengths.items():
            if arg not in af_caregiver.decisions:
                continue
            decisions[arg] = round(strength, 4)

        # Find highest strength for this semantics
        max_strength = max(decisions.values())

        # Add row data
        table_data["Semantics"].append(sem)
        for arg in all_decisions:
            strength = decisions.get(arg, 0)
            # Format with markdown for bold and color if it's the highest
            formatted_value = f"{strength}"  # f"**{arg}**: :green[**{strength}**]" if strength == max_strength else f"{arg}: {strength}"
            table_data[arg].append(formatted_value)
            table_decisions[sem] = [k for k, v in decisions.items() if v == max_strength]

    

    # Create and display the table
    st.write("Final strengths for each argument across different semantics:")
    st.dataframe(table_data, hide_index=True)

    # Create and display the table
    st.write("Decision for each gradual semantic model:")
    st.dataframe(table_decisions, hide_index=True)

    # Display rounded final strengths for each argument
    st.write("\nArguments Final Strengths for each gradual semantic model:")

    table_final_strengths = {sem: [] for sem in semantics}
    for sem in semantics:
        final_strengths = af_caregiver.get_final_strengths(
            semantics=sem
        ).final_strengths

        # Get only the decisions and round them
        final_strengths_rounded = {
            k: round(v, 3)
            for k, v in final_strengths.items()
            if k not in af_caregiver.decisions
        }
        table_final_strengths[sem] = final_strengths_rounded

        #st.markdown(f"**{sem}**: ")
        #st.write(final_strengths_rounded)


    print(table_final_strengths)
    
    st.dataframe(table_final_strengths)