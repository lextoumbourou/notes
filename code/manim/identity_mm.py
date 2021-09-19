from pathlib import Path
from copy import deepcopy

import numpy as np

from manim import (
    Scene, SMALL_BUFF, Group, SurroundingRectangle, Create,
    ReplacementTransform, Matrix, MathTex, DOWN, Write, BLACK, Uncreate,
    WHITE, ApplyMethod, BLUE_D, GREEN_D, VGroup, Text, config, RED_D, UP)

config.background_color = WHITE
config.format = 'gif'
config.save_as_gif = True
config.pixel_height = 300
config.pixel_width = 600
config.low_quality = True

DEFAULT_COLOR = BLACK
ONES_COLOR = RED_D
M1_COLOR = "#7187A2"
M2_COLOR = RED_D
CALCULATED_COLOR = "#655b28"
RECT_COLOR = "#b9b28b"


class IdentityMatrixMultiplication(Scene):
    def construct(self):
        matrix1 = Matrix(
            m1,
            v_buff=1.3,
            h_buff=0.8,
            bracket_h_buff=SMALL_BUFF,
            bracket_v_buff=SMALL_BUFF)

        matrix1.set_color(DEFAULT_COLOR)

        matrix1.get_entries()[0].set_color(RECT_COLOR)
        matrix1.get_entries()[-1].set_color(RECT_COLOR)

        self.add(matrix1)

        matrix2 = Matrix(
            m2,
            v_buff=1.3,
            h_buff=0.8,
            bracket_h_buff=SMALL_BUFF,
            bracket_v_buff=SMALL_BUFF)
        matrix2.set_color(DEFAULT_COLOR)
        self.add(matrix2)

        equals = MathTex(r'=')
        self.add(equals)

        m3 = (m1@m2)
        matrix3 = Matrix(
            m3,
            v_buff=1.3,
            h_buff=0.8,
            bracket_h_buff=SMALL_BUFF,
            bracket_v_buff=SMALL_BUFF)
        matrix3.set_color(DEFAULT_COLOR)

        # args = [BLACK]*m3.shape[1]
        # matrix3.set_column_colors(*args)

        for i in matrix3.get_entries():
            i.set_opacity(0)

        self.add(matrix3)

        g = Group(matrix1, matrix2, equals, matrix3).arrange(buff=1).shift(UP)
        g.set_color(DEFAULT_COLOR)
        self.add(g)

        i = 0
        for m1_row in matrix1.get_rows():
            for m2_column in matrix2.get_columns():
                rect_1 = SurroundingRectangle(m1_row).set_stroke(color=RECT_COLOR)
                rect_2 = SurroundingRectangle(m2_column).set_stroke(color=RECT_COLOR)

                self.play(Create(rect_1))
                self.wait()

                self.play(ReplacementTransform(rect_1, rect_2))

                math_strings = []
                values_to_sum = []
                for m1_val, m2_val in zip(m1_row, m2_column):
                    val_left = int(m1_val.get_tex_string())
                    val_right = int(m2_val.get_tex_string())
                    values_to_sum.append(val_left * val_right)
                    text_left = m1_val.get_tex_string()
                    text_right = m2_val.get_tex_string()

                    if val_left < 0:
                        text_left = f'({text_left})'

                    if val_right < 0:
                        text_right = f'({text_right})'

                    math_strings.extend([text_left, " \\times ", text_right, " + "])

                math_strings = math_strings[:-1]
                math_strings.extend([" = ", sum(values_to_sum)])

                ds_m = MathTex(*math_strings)
                ds_m.set_color(BLACK)

                offset = 0
                for j in range(m1.shape[0]):
                    pos = j + offset
                    # ds_m[pos].set_color(M1_COLOR)
                    # ds_m[pos + 2].set_color(M2_COLOR)

                    offset += 3

                ds_m[-1].set_opacity(0)
                # ds_m[-1].set_color(CALCULATED_COLOR)

                ds_m.shift(2 * DOWN)

                self.play(ReplacementTransform(rect_1, rect_2), Write(ds_m))

                self.wait()

                rect_3 = SurroundingRectangle(matrix3.get_entries()[i]).set_color(RECT_COLOR)
                # matrix3.get_entries()[i].set_color(CALCULATED_COLOR)
                self.play(
                    ReplacementTransform(rect_2, rect_3),
                    ApplyMethod(matrix3.get_entries()[i].set_opacity, 1),
                    ApplyMethod(ds_m[-1].set_opacity, 1)
                )

                self.wait()

                self.remove(ds_m)
                self.remove(rect_3)

                m1_row.set_color(BLACK)
                m2_column.set_color(BLACK)

                i += 1
        self.wait()

if __name__ == '__main__':
    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/identity-matrix-1')

    m1 = np.array([[1, 0], [0, 1]])
    m2 = np.array([[2, 4, 5], [1, 3, -2]])

    scene = IdentityMatrixMultiplication()
    scene.render()
