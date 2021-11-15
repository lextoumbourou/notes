from pathlib import Path

from manim import *
from numpy import array

from theme import Theme

config.background_color = WHITE
config.save_as_gif = True
config.pixel_height = 300
config.pixel_width = 300
config.medium_quality = True
config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/law-of-cosines-cover')

config.frame_width = 6
config.frame_height = 6

LINE_COLOR = "#b9b28b"
TEXT_COLOR = "#7187A2"

line_args = {}

class LawOfCosinesTriange(Scene):
    def construct(self):
        line_1 = Line(start=array([-2, -2, 0]), end=array([0, 2, 0]), **line_args)
        line_1.set_color(Theme.TEXT_COLOR)

        line_2 = Line(start=array([-2, -2, 0]), end=array([2, -1, 0]), **line_args)
        line_2.set_color(Theme.TEXT_COLOR)

        line_3 = Line(start=line_2.end, end=line_1.end, **line_args)
        line_3.set_color(Theme.TEXT_COLOR)

        angle = Angle(line_1, line_2, color=Theme.TEXT_COLOR, other_angle=True, radius=1)
        self.add(line_1, line_2, line_3, angle)

        equation = MathTex(r"\textbf{c}^2 = \textbf{a}^2 + \textbf{b}^2 - 2\textbf{ab}\cos\theta", font_size=50)
        equation.set_color(Theme.TEXT_COLOR)
        equation.move_to(UP * 2.5)

        equation[0][0].set_color(Theme.COLOR_1)
        equation[0][3].set_color(Theme.COLOR_2)
        equation[0][6].set_color(Theme.COLOR_3)
        equation[0][10].set_color(Theme.COLOR_2)
        equation[0][11].set_color(Theme.COLOR_3)
        equation[0][-1].set_color(Theme.COLOR_4)

        self.add(equation)

        line_text = Text("a")
        line_text.set_color(Theme.COLOR_2)
        line_text.move_to(line_1.get_center() + UP/2 + LEFT/2)
        self.add(line_text)

        line_text = Text("b")
        line_text.move_to(line_2.get_center() + DOWN/2 + RIGHT/2)
        line_text.set_color(Theme.COLOR_3)
        self.add(line_text)

        line_text = Text("c")
        line_text.set_color(Theme.COLOR_1)
        line_text.next_to(line_3.get_center() + RIGHT/4 + UP/4)
        self.add(line_text)

        line_text = Text("θ")
        line_text.set_color(Theme.COLOR_4)
        line_text.move_to(angle.get_center() + 0.4 * RIGHT + 0.4 * UP)
        self.add(line_text)


if __name__ == '__main__':
    scene = LawOfCosinesTriange()
    scene.render()
