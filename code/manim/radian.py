import math

from pathlib import Path

from manim import *
from manim import Scene

config.background_color = WHITE
config.save_as_gif = True
config.pixel_height = 300
config.pixel_width = 400
config.medium_quality = True
config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/radian-cover')

config.frame_width = 10
config.frame_height = 4

DEFAULT_COLOR = BLACK
TEXT_COLOR = "#7187A2"
LINE_COLOR = "#b9b28b"


class Radian(Scene):
    def construct(self):
        center_point = ORIGIN

        line1 = Line(center_point, center_point + RIGHT * 3, color=LINE_COLOR, stroke_width=7)

        line_moving = Line(center_point, center_point + RIGHT * 3, color=DEFAULT_COLOR)
        line_ref = line_moving.copy()
        line_moving.rotate(1, about_point=center_point)

        angle = Angle(line1, line_moving, radius=0.5, other_angle=False, color=TEXT_COLOR)

        tex = MathTex(r"\theta").move_to(
            Angle(
                line1, line_moving, radius=0.5 + 3 * SMALL_BUFF, other_angle=False
            ).point_from_proportion(0.5)
        )
        tex.set_color(TEXT_COLOR)

        b1 = Brace(line1)
        b1.set_color(TEXT_COLOR)

        b1text = b1.get_text("radius")
        b1text.set_color(TEXT_COLOR)

        circle = Circle(color=DEFAULT_COLOR, radius=3)

        arc_between_points = ArcBetweenPoints(line1.get_end(), line_moving.get_end(), color=LINE_COLOR, stroke_width=7, angle=1)

        arc_text = Text("radius", font_size=38.4)
        arc_text.set_color(TEXT_COLOR)
        arc_text.move_to(arc_between_points.get_center() + RIGHT * 1.5)

        bottom_text = MathTex(r"\theta = 1 \ \text{radian}", font_size=60).move_to(UP * 3.5)
        bottom_text.set_color(TEXT_COLOR)

        self.add(line1, line_moving, angle, tex, b1text, circle, b1, arc_between_points, arc_text, bottom_text)


if __name__ == '__main__':
    scene = Radian()
    scene.render()
