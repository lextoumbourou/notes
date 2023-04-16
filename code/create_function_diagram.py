import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from pathlib import Path

# Elements of Set A and Set B
set_a_elements = ["Clyde", "Sarah", "Geoff", "Betty"]
set_b_elements = [0, "...", 11, "...", 34, "...", 98, "...", 120]

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 4))

# Define positions for Set A and Set B elements
x_a = np.full(len(set_a_elements), 0.25)
y_a = np.linspace(0.8, 0.2, len(set_a_elements))

x_b = np.full(len(set_b_elements), 0.75)
y_b = np.linspace(0.8, 0.2, len(set_b_elements))

# Plot Set A and Set B elements
for i, element in enumerate(set_a_elements):
    ax.text(x_a[i], y_a[i], element, ha="center", va="center")

age_to_position = {}
for i, element in enumerate(set_b_elements):
    if isinstance(element, int):
        age_to_position[element] = (x_b[i], y_b[i])
        color = "darkred" if element in [11, 34, 98] else "black"
    else:
        color = "black"
    ax.text(x_b[i], y_b[i], element, ha="center", va="center", color=color)

# Define mappings from Set A to Set B
mappings = [
    (0, 11),  # Clyde -> 11
    (1, 34),  # Sarah -> 34
    (2, 34),  # Geoff -> 34
    (3, 98),  # Betty -> 98
]

# Draw arrows for the mappings
for start, end in mappings:
    ax.annotate(
        "",
        xy=(age_to_position[end][0] - 0.05, age_to_position[end][1]),
        xycoords="data",
        xytext=(x_a[start] + 0.05, y_a[start]),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )


# Draw ovals around Set A and Set B
set_a_oval = patches.Ellipse(
    (0.25, 0.5), 0.25, 0.8, fill=True, facecolor="lightblue", edgecolor=None
)
ax.add_patch(set_a_oval)
set_b_oval = patches.Ellipse(
    (0.75, 0.5), 0.25, 0.8, fill=True, facecolor="orange", edgecolor=None
)
ax.add_patch(set_b_oval)

# Set plot limits and remove axis
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Add Set A and Set B labels
ax.text(0.25, 1.0, "A", ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(0.75, 1.0, "B", ha="center", va="center", fontsize=14, fontweight="bold")

# Show the plot
plt.title("f : A → B")
output_file = Path(__file__).resolve().parent.parent.parent / Path(
    "notes/notes/_media/function-diagram.png"
)
plt.savefig(output_file, dpi=150)
