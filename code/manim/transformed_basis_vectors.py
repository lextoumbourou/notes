from pathlib import Path

import numpy as np

from manim import *


config.background_color = WHITE
config.pixel_height = 200
config.pixel_width = 200
config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/trans-basis')
config.format = 'gif'

config.frame_width = 4
config.frame_height = 4


class BasisVectors(Scene):
    def construct(self):
        basis_1 = np.array([1, 0, 0])
        basis_2 = np.array([0, 1, 0])

        trans_vect = np.array([1, 2, 0])

        vect1_color = "#b9b28b"
        vect2_color = "#b98b99"

        vect1 = Line(start=ORIGIN, end=basis_1, stroke_color=vect1_color).add_tip()
        vect1_name = MathTex("\\hat{i}").next_to(vect1.get_end(), RIGHT * 2, buff=0.1).set_color(vect1_color)

        vect2 = Line(start=ORIGIN, end=basis_2, stroke_color=vect2_color).add_tip()
        vect2_name = MathTex("\\hat{j}").next_to(vect2.get_end(), UP * 2, buff=0.1).set_color(vect2_color)

        trans_vect = Line(start=ORIGIN, end=trans_vect, stroke_color=vect2_color).add_tip()

        self.camera.frame_center = np.array([0, 1, 0])

        self.add(vect1, vect1_name, vect2, vect2_name)

        self.play(Transform(vect2, trans_vect))
        self.wait()


if __name__ == '__main__':
    scene = BasisVectors()
    scene.render()
