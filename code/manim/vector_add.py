from pathlib import Path
from numpy import array

from manim import *


class VectorAddition(Scene):
    def construct(self):
        VECT1 = np.array([3, 2, 0])
        VECT2 = np.array([2, -1, 0])

        VECT1_COLOR = "#b9b28b"
        VECT2_COLOR = "#b98b99"
        VECT3_COLOR = "#8ba7b9"

        vect1 = Line(start=ORIGIN, end=VECT1, stroke_color=VECT1_COLOR).add_tip()
        vect1_name = MathTex("\\vec{a}").next_to(vect1.get_center(), UP + LEFT * 2, buff=0.1).set_color(VECT1_COLOR)

        vect2 = Line(start=VECT1, end=VECT1 + VECT2, stroke_color=VECT2_COLOR).add_tip()
        vect2_name = MathTex("\\vec{b}").next_to(vect2.get_center(), UP * 2 + RIGHT, buff=0.1).set_color(VECT2_COLOR)

        vect3 = Line(start=ORIGIN, end=VECT1 + VECT2, stroke_color=VECT3_COLOR, stroke_width=8).add_tip()
        vect3_name = MathTex("\\vec{a} + \\vec{b}").next_to(vect3.get_center(), DOWN * 1.5, buff=0.1).set_color(VECT3_COLOR)

        self.camera.frame_center = np.array([2.5, 1, 0])

        self.play(GrowFromPoint(vect1, point=vect1.get_start()), Write(vect1_name), run_time=2)
        self.wait()

        self.play(GrowFromPoint(vect2, point=vect2.get_start()), Write(vect2_name), run_time=2)
        self.wait()

        self.play(LaggedStart(GrowFromPoint(vect3, point=vect3.get_start())), Write(vect3_name), run_time=3, lag_ratio=1)
        self.wait(4)


if __name__ == '__main__':
    # Generate animated gif.
    config.background_color = WHITE
    config.pixel_height = 300
    config.pixel_width = 600

    config.frame_width = 6
    config.frame_height = 5

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/vector-add-example')

    config.format = 'gif'

    scene = VectorAddition()
    scene.render()

    # Generate cover png.
    config.save_last_frame = True

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/vector-add-cover')

    scene = VectorAddition()
    scene.render()
