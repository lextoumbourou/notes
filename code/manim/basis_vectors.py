from pathlib import Path

import numpy as np

from manim import *


config.background_color = WHITE
config.save_as_gif = True
config.pixel_height = 200
config.pixel_width = 200
config.medium_quality = True
config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/basis-vector-cover')

config.frame_width = 4
config.frame_height = 4


class BasisVectors(Scene):
    def construct(self):
        numberplane = NumberPlane(
            x_range=(-2, 2),
            y_range=(-2, 2), 
            background_line_style={
                "stroke_opacity": 0.4
            }
        )

        basis_1 = np.array([1, 0, 0])
        basis_2 = np.array([0, 1, 0])

        vect1_color = "#b9b28b"
        vect2_color = "#b98b99"

        vect1 = Line(start=ORIGIN, end=basis_1, stroke_color=vect1_color).add_tip()
        vect1_name = MathTex("\\hat{i}").next_to(vect1.get_end(), RIGHT * 2, buff=0.1).set_color(vect1_color)

        vect2 = Line(start=ORIGIN, end=basis_2, stroke_color=vect2_color).add_tip()
        vect2_name = MathTex("\\hat{j}").next_to(vect2.get_end(), UP * 2, buff=0.1).set_color(vect2_color)

        self.add(numberplane, vect1, vect1_name, vect2, vect2_name)


if __name__ == '__main__':
    scene = BasisVectors()
    scene.render()
