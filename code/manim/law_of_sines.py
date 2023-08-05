from pathlib import Path

from manim import *
from numpy import array

from theme import Theme

config.background_color = WHITE
config.save_as_gif = True
config.pixel_height = 300
config.pixel_width = 300
config.medium_quality = True
config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/law-of-sines-cover')

config.frame_width = 6
config.frame_height = 6

LINE_COLOR = "#b9b28b"
TEXT_COLOR = "#7187A2"

line_args = {}

class LawOfSinesTriangle(Scene):
    def construct(self):
        line_1 = Line(start=array([-2, -2.5, 0]), end=array([1, 1.5, 0]), **line_args)
        line_1.set_color(Theme.TEXT_COLOR)

        line_2 = Line(start=array([-2, -2.5, 0]), end=array([2, -1, 0]), **line_args)
        line_2.set_color(Theme.TEXT_COLOR)

        line_3 = Line(start=line_2.end, end=line_1.end, **line_args)
        line_3.set_color(Theme.TEXT_COLOR)

        angle_A = Angle(line_2, line_3, color=Theme.TEXT_COLOR, radius=0.7, other_angle=True, quadrant=(-1, 1))
        angle_B = Angle(line_3, line_1, color=Theme.TEXT_COLOR, radius=0.7, other_angle=True, quadrant=(-1, -1))
        angle_C = Angle(line_1, line_2, color=Theme.TEXT_COLOR, other_angle=True, radius=0.7)

        self.add(line_1, line_2, line_3, angle_A, angle_B, angle_C)

        equation = MathTex(r"\frac{\sin(A)}{a} = \frac{\sin(B)}{b} = \frac{\sin(C)}{c}", font_size=50)
        equation.set_color(Theme.TEXT_COLOR)

        equation[0][4].set_color(Theme.COLOR_2_a)
        equation[0][7].set_color(Theme.COLOR_2)

        equation[0][13].set_color(Theme.COLOR_3_a)
        equation[0][16].set_color(Theme.COLOR_3)

        equation[0][22].set_color(Theme.COLOR_1_a)
        equation[0][25].set_color(Theme.COLOR_1)

        equation.move_to(UP * 2.25)

        self.add(equation)

        self.add_label(line_1, 'a', Theme.COLOR_2, UP + LEFT)
        self.add_label(line_2, 'b', Theme.COLOR_3, DOWN + RIGHT)
        self.add_label(line_3, 'c', Theme.COLOR_1, UP + RIGHT)

        self.add_angle_label(angle_A, 'A', Theme.COLOR_2_a, 0.2 * UP + 0.5 * LEFT)
        self.add_angle_label(angle_B, 'B', Theme.COLOR_3_a, 0.4 * DOWN + 0 * LEFT)
        self.add_angle_label(angle_C, 'C', Theme.COLOR_1_a, 0.35 * UP + 0.4 * RIGHT)

    def add_label(self, line, text, color, direction):
        label = Text(text)
        label.set_color(color)
        label.next_to(line.get_center(), direction=direction)
        self.add(label)

    def add_angle_label(self, angle, text, color, offset):
        label = Text(text, font_size=36)
        label.set_color(color)
        label.move_to(angle.get_center() + offset)
        self.add(label)


if __name__ == '__main__':
    scene = LawOfSinesTriangle()
    scene.render()
