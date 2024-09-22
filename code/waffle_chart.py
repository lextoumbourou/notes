import matplotlib.pyplot as plt
from pywaffle import Waffle

data = {'Biden': 306, 'Trump': 232}

fig = plt.figure(
    FigureClass=Waffle,
    columns=40,
    values=data,
    figsize=(10, 10),
    colors=["#0015BC", "#E9141D"],
    title={"label": "2020 Presidential Electoral College Results", "loc": "left", "fontsize": 18, "pad": 20},
    legend={
        'labels': [f"{k} ({v})" for k, v in data.items()],
        'loc': 'lower left',
        'bbox_to_anchor': (0, -0.2),
        'ncol': len(data),
        'framealpha': 0,
        'fontsize': 15,

    }
)
fig.set_facecolor('#EEEEEE')
fig.patch.set_alpha(0.0)

# Save the figure as a PNG file
plt.savefig('notes/_media/waffle_example_2020_pres.png', dpi=100, bbox_inches='tight')
