import math

from pathlib import Path
from numpy import array
import numpy as np

from manim import *

ANGLE_IN_DEGREES = 90
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
        VECT1 = np.array([2, 2])
        VECT1_COLOR = "#b9b28b"

        TRANS_VEC = TRANSFORMATION_MATRIX @ VECT1
        VECT2_COLOR = "#b98b99"

        VECT3_COLOR = "#8ba7b9"

        vect1 = Line(start=ORIGIN, end=convert_to_3d(VECT1), stroke_color=VECT1_COLOR).add_tip()
        # vect1_name = MathTex(r"\begin{bmatrix}2 \\ 2\end{bmatrix}").next_to(vect1.get_end(), UP + RIGHT * 2, buff=0.1).set_color(VECT1_COLOR)

        angle = math.radians(45)

        matrix_part = get_transform_matrix_text(90, color=VECT3_COLOR)

        vector_part = MathTex(f"\\begin{{bmatrix}}{VECT1[0]} \\\\ {VECT1[1]}\end{{bmatrix}}").set_color(VECT1_COLOR)
        equals_part = MathTex(" = ").set_color(VECT1_COLOR)

        transformed_vector_part = MathTex(f"\\begin{{bmatrix}}{round(TRANS_VEC[0], 1)} \\\\ {round(TRANS_VEC[1], 1)}\end{{bmatrix}}").set_color(VECT2_COLOR)

        expression = VGroup(matrix_part, vector_part, equals_part, transformed_vector_part).arrange(RIGHT)

        expression.shift(UP * 3 + LEFT * 2)

        # self.add(expression)

        self.play(GrowFromPoint(vect1, point=vect1.get_start()), FadeIn(vector_part), FadeIn(matrix_part))

        self.wait()

        self.play(FadeIn(equals_part))

        trans_vect1 = vect1.copy().set_color(VECT2_COLOR)
        trans_vect1.put_start_and_end_on(ORIGIN, np.array([2, 2.00001, 0]))

        angle = Angle(vect1, trans_vect1, other_angle=True)
        self.add(angle)

        angle.add_updater(
            lambda x: x.become(Angle(vect1, trans_vect1, color=VECT2_COLOR, other_angle=True))
        )

        self.add(trans_vect1)
        self.play(ApplyMethod(trans_vect1.put_start_and_end_on, ORIGIN, np.array([*list(TRANS_VEC), 0])), FadeIn(transformed_vector_part))

        self.wait(3)


if __name__ == '__main__':
    # Generate animated gif.
    config.background_color = WHITE

    config.pixel_width = 640
    config.pixel_height = 360

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/transformation-matrix-example')

    config.format = 'gif'

    scene = TransformationMatrix()
    scene.render()
