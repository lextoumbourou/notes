import matplotlib.pyplot as plt
from pywaffle import Waffle

from manim.theme import Theme

fig = plt.figure(
    FigureClass=Waffle,
    rows=3,
    columns=3,
    values=[30, 16, 4],
    colors=[Theme.COLOR_1, Theme.COLOR_2, Theme.COLOR_3]
)
plt.savefig('notes/_media/waffle_cover.png', dpi=50, bbox_inches='tight')
