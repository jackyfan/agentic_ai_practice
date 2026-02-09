import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def visualize_srl(verb, agent, patient, recipient=None, **kwargs):
    """
     Creates a semantic blueprint and visualizes it as a stemma.
     This is the main, user-facing function.
     """
    srl_roles = {
        "Agent (ARG0)": agent,
        "Patient (ARG1)": patient,
    }
    if recipient:
        srl_roles["Recipient (ARG2)"] = recipient
    # Add any extra modifier roles passed in kwargs
    for key, value in kwargs.items():
        # Format the key for display, e.g., "temporal" -> "Temporal (ARGM-TMP)"
        role_name = f"{key.capitalize()} (ARGM-{key[:3].upper()})"
        srl_roles[role_name] = value
    _plot_stemma(verb, srl_roles)


def _plot_stemma(verb, srl_roles):
    """Internal helper function to generate the stemma visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    verb_style = dict(boxstyle="round,pad=0.5", fc="lightblue", ec="b")
    role_style = dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="g")

    verb_pos = (5, 8.5)
    ax.text(verb_pos[0], verb_pos[1], verb, ha="center", va="center",
            bbox=verb_style, fontsize=12)

    srl_items = list(srl_roles.items())
    num_roles = len(srl_items)
    x_positions = [10 * (i + 1) / (num_roles + 1) for i in range(num_roles)]
    y_position = 4.5

    for i, (role, text) in enumerate(srl_items):
        child_pos = (x_positions[i], y_position)
        ax.text(child_pos[0], child_pos[1], text, ha="center", va="center",
                bbox=role_style, fontsize=10, wrap=True)

        arrow = FancyArrowPatch(
            verb_pos,
            child_pos,
            arrowstyle='->',
            mutation_scale=20,
            shrinkA=15,
            shrinkB=15,
            color='gray'
        )
        ax.add_patch(arrow)

        label_pos = ((verb_pos[0] + child_pos[0]) / 2, (verb_pos[1] + child_pos[1]) / 2 + 0.5)
        ax.text(label_pos[0], label_pos[1], role, ha="center", va="center",
                fontsize=9, color='black', bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

    fig.suptitle("The Semantic Blueprint (Stemma Visualization)", fontsize=16)
    plt.show()


def main():
    print("Example 1: A complete action with multiple roles.")
    visualize_srl(
        verb="pitch",
        agent="Sarah",
        patient="the new project",
        recipient="to the board",
        temporal="in the morning"
    )

    print("\nExample 2: An action with a location")
    visualize_srl(
        verb="resolved",
        agent="The backend team",
        patient="the critical bug",
        location="in the payment gateway"
    )

    print("\nExample 3: Describing how an action was performed")
    visualize_srl(
        verb="deployed",
        agent="Maria's team",
        patient="the new dashboard",
        manner="ahead of schedule"
    )

if __name__ == "__main__":
    main()