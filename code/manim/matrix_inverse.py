from pathlib import Path
import math

import numpy as np
from manim import Scene, config, Line, NumberPlane, ReplacementTransform, MathTex
from manim import UP, ORIGIN, DOWN

from theme import Theme


ANGLE_IN_DEGREES = -90
angle = math.radians(ANGLE_IN_DEGREES)

trans_matrix = np.array([
    [math.cos(angle), math.sin(angle)],
    [-math.sin(angle), math.cos(angle)]
])

def convert_to_3d(vect):
    return np.array([*list(vect), 0])


class InverseTransformation(Scene):
    def construct(self):
        origin = np.array([0, -2])

        vect1 = np.array([2, 2])

        trans_vect = trans_matrix @ vect1

        vect1 = vect1 + origin
        trans_vect = trans_vect + origin

        line1 = Line(start=convert_to_3d(origin), end=convert_to_3d(vect1), stroke_color=Theme.COLOR_1).add_tip()
        line1_orig = line1.copy()
        line2 = Line(start=convert_to_3d(origin), end=convert_to_3d(trans_vect), stroke_color=Theme.COLOR_2, stroke_width=7).add_tip()

        # numberplane = NumberPlane()

        equation = MathTex(
            "A \\vec{x} = \\vec{v}", font_size=105
        ).set_color(Theme.TEXT_COLOR)
        equation.move_to(ORIGIN + UP * 2.5)

        equation[0][1].set_color(Theme.COLOR_1)
        equation[0][2].set_color(Theme.COLOR_1)
        equation[0][4].set_color(Theme.COLOR_2)
        equation[0][5].set_color(Theme.COLOR_2)

        equation_orig = equation.copy()

        equation2 = MathTex(
            "A^{-1} \\vec{v} = \\vec{x}", font_size=105
        ).set_color(Theme.TEXT_COLOR)
        equation2.move_to(ORIGIN + UP * 2.5)

        equation2[0][3].set_color(Theme.COLOR_2)
        equation2[0][4].set_color(Theme.COLOR_2)
        equation2[0][6].set_color(Theme.COLOR_1)
        equation2[0][7].set_color(Theme.COLOR_1)

        self.add(line1, equation)

        self.play(ReplacementTransform(line1, line2))
        self.wait()

        self.play(ReplacementTransform(equation, equation2))
        self.wait()

        self.play(ReplacementTransform(line2, line1_orig))
        self.wait()

        self.play(ReplacementTransform(equation2, equation_orig))
        self.wait()


if __name__ == '__main__':
    config.background_color = Theme.BG_COLOR

    config.pixel_width = 640
    config.pixel_height = 360

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/inverse-matrix-transformation')

    config.format = 'gif'

    scene = InverseTransformation()
    scene.render()
