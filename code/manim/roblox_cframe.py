from pathlib import Path

from manim import *

config.background_color = WHITE
config.save_as_gif = True
config.pixel_height = 300
config.pixel_width = 300

config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/roblox-cframes-cover')

config.frame_width = 3
config.frame_height = 4

LINE_COLOR = "#b9b28b"
TEXT_COLOR = "#7187A2"

RIGHT_COLOR = '#A92C21'
UP_COLOR = '#89CC4C'
BACK_COLOR = '#1220CB'


class CFrame(ThreeDScene):
    def construct(self):
        #axes = ThreeDAxes(x_range=(0, 6, 1), y_range=(0, 5, 1), z_range=(0, 4, 1))

        dot = Dot(point=ORIGIN)
        dot.set_color(TEXT_COLOR)

        right = Arrow(start=[0., 0., 0.], end=[2., 0., 0.])
        right.set_color(RIGHT_COLOR)

        right_text = Text("right vector")
        right_text.move_to(right.get_center())
        right_text.set_color(TEXT_COLOR)

        up = Arrow(start=[0., 0., 0.], end=[0., -2., 0.])
        up.set_color(UP_COLOR)

        back = Arrow(start=[0., 0., 0.], end=[0., 0., 2.])
        back.set_color(BACK_COLOR)

        self.add(right, up, back, dot, right_text)

        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)


if __name__ == '__main__':
    scene = CFrame()
    scene.render()
