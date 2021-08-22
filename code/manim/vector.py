from manim import *

config.background_color = WHITE
config.save_as_gif = True
config.pixel_height = 300
config.pixel_width = 300

config.frame_width = 6
config.frame_height = 6

DEFAULT_COLOR = BLACK

class VectorArrow(Scene):
    def construct(self):
        dot = Dot(ORIGIN).set_color(BLACK)
        arrow = Arrow(ORIGIN, [1, 2, 0], buff=0).set_color(BLACK)
        numberplane = NumberPlane(x_range=(-4, 4), y_range=(-4, 4))
        origin_text = Text('Origin').next_to(dot, DOWN).set_color(BLACK)
        tip_text = Text('(1, 2)').next_to(arrow.get_end(), RIGHT).set_color(BLACK)
        self.add(numberplane, dot, arrow, origin_text, tip_text)
