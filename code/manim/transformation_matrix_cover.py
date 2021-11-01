import math

from pathlib import Path
from numpy import array
import numpy as np

from manim import *

ANGLE_IN_DEGREES = -90
angle = math.radians(ANGLE_IN_DEGREES)

TRANSFORMATION_MATRIX = np.array([
    [math.cos(angle), math.sin(angle)],
    [-math.sin(angle), math.cos(angle)]
])


def convert_to_3d(vect):
    return np.array([*list(vect), 0])


def get_transform_matrix_text(angle, color):
    return MathTex(
        f"\\begin{{bmatrix}}\\cos({angle}) && \\sin({angle}) \\\\ -\\sin({angle}) && \\cos({angle})\\end{{bmatrix}}").set_color(color)


class TransformationMatrix(Scene):
    def construct(self):
        VECT1 = np.array([3, 2])
        VECT1_COLOR = "#b9b28b"

        TRANS_VEC = TRANSFORMATION_MATRIX @ VECT1
        VECT2_COLOR = "#b98b99"

        VECT3_COLOR = "#8ba7b9"

        vect1 = DashedLine(start=ORIGIN, end=convert_to_3d(VECT1), stroke_color=VECT1_COLOR, stroke_width=10).add_tip()
        # vect1_name = MathTex(r"\begin{bmatrix}2 \\ 2\end{bmatrix}").next_to(vect1.get_end(), UP + RIGHT * 2, buff=0.1).set_color(VECT1_COLOR)

        trans_vect1 = Line(start=ORIGIN, end=convert_to_3d(TRANS_VEC), stroke_color=VECT1_COLOR, stroke_width=10).add_tip()

        self.camera.frame_center = np.array([0, 1, 0])

        self.add(vect1, trans_vect1)


if __name__ == '__main__':
    # Generate animated gif.
    config.background_color = WHITE

    config.frame_height = 8
    config.frame_width = 8

    config.pixel_width = 300
    config.pixel_height = 300

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/transformation-matrix-cover')
    config.save_last_frame = True

    scene = TransformationMatrix()
    scene.render()
