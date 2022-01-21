from numpy import array
from pathlib import Path

from manim import *

from theme import Theme


class EigenVectors(LinearTransformationScene):
    def __init__(self):
        super().__init__(
            include_background_plane=True,
            include_foreground_plane=True,
            show_basis_vectors=False
        )

    def construct(self):
        matrix = [[2, 1], [0, 2]]

        ihat = self.add_vector((1, 0), animate=False, color=Theme.COLOR_1)
        self.add_transformable_label(
            ihat, r'\hat{i}',
            new_label=r'\mathbf{A} \hat{j}',
            animate=False, color=Theme.COLOR_1)

        jhat = self.add_vector((0, 1), animate=False, color=Theme.COLOR_2)
        self.add_transformable_label(
            jhat, r'\hat{j}',
            new_label=r'\mathbf{A} \hat{j}',
            animate=False, color=Theme.COLOR_2)

        self.add_background_mobject(
            Line(start=array([0, 0, 0]), end=array([1, 0, 0]) * 10, stroke_color=Theme.COLOR_1))
        self.add_background_mobject(
            Line(start=array([0, 0, 0]), end=array([0, 1, 0]) * 10, stroke_color=Theme.COLOR_2))

        self.apply_matrix(matrix)
        self.wait()


if __name__ == '__main__':
    config.background_color = WHITE
    config.format = 'gif'

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/eigenvector')

    scene = EigenVectors()
    scene.render()
