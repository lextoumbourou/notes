from pathlib import Path
from numpy import array

from manim import *

class DottedLine(Line):

    """A dotted :class:`Line`.
    Parameters
    ----------
    args : Any
        Arguments to be passed to :class:`Line`
    dot_spacing : Optional[:class:`float`]
        Minimal spacing of the dots. The spacing is scaled up to fit the start and end of the line.
    dot_kwargs : Any
        Arguments to be passed to ::class::`Dot`
    kwargs : Any
        Additional arguments to be passed to :class:`Line`
    Examples
    --------
    .. manim:: DottedLineExample
        :save_last_frame:
        class DottedLineExample(Scene):
            def construct(self):
                # default dotted line
                dotted_1 = DottedLine(LEFT, RIGHT))
                # reduced spacing
                dotted_2 = DottedLine(LEFT, RIGHT, dot_spacing=.3).shift(.5*DOWN))
                # smaller and colored dots
                dotted_3 = DottedLine(LEFT, RIGHT, dot_kwargs=dict(radius=.04, color=YELLOW)).shift(DOWN))

                self.add(dotted_1, dotted_2, dotted_3)

    """

    def __init__(
        self,
        *args,
        dot_spacing=0.1,
        dot_kwargs={},
        **kwargs
    ):
        Line.__init__(self, *args, **kwargs)
        n_dots = int(self.get_length() / dot_spacing) + 1
        dot_spacing = self.get_length() / (n_dots - 1)
        unit_vector = self.get_unit_vector()
        start = self.start

        self.dot_points = [start + unit_vector * dot_spacing * x for x in range(n_dots)]
        self.dots = [Dot(point, **dot_kwargs) for point in self.dot_points]

        self.clear_points()

        self.add(*self.dots)

        self.get_start = lambda: self.dot_points[0]
        self.get_end = lambda: self.dot_points[-1]

    def get_first_handle(self):
        return self.dot_points[-1]

    def get_last_handle(self):
        return self.dot_points[-2]


class VectorAddition(Scene):
    def construct(self):
        VECT1 = np.array([3, 2, 0])
        VECT2 = np.array([2, -1, 0])

        VECT1_COLOR = "#b9b28b"
        VECT2_COLOR = "#b98b99"
        VECT3_COLOR = "#8ba7b9"

        vect1 = Line(start=ORIGIN, end=VECT1, stroke_color=VECT1_COLOR).add_tip()
        vect1_name = MathTex("\\vec{a}").next_to(vect1.get_center(), DOWN + RIGHT * 2, buff=0.1).set_color(VECT1_COLOR)

        vect2 = Line(start=VECT1, end=VECT1 + VECT2, stroke_color=VECT2_COLOR).add_tip()
        vect2_name = MathTex("\\vec{b}").next_to(vect2.get_center(), UP * 2 + RIGHT, buff=0.1).set_color(VECT2_COLOR)

        vect2_negative = DashedLine(start=VECT1, end=VECT1 - VECT2, stroke_color=VECT2_COLOR).add_tip()
        vect2_negative_name = MathTex("-\\vec{b}").next_to(vect2_negative.get_center(), UP * 2 + RIGHT, buff=0.1).set_color(VECT2_COLOR)

        vect3 = Line(start=ORIGIN, end=VECT1 - VECT2, stroke_color=VECT3_COLOR, stroke_width=8).add_tip()
        vect3_name = MathTex("\\vec{a} - \\vec{b}").next_to(vect3.get_center(), LEFT * 2, buff=0.1).set_color(VECT3_COLOR)

        self.camera.frame_center = np.array([2.5, 1.5, 0])

        self.play(GrowFromPoint(vect1, point=vect1.get_start()), Write(vect1_name), run_time=2)
        self.wait()

        self.play(GrowFromPoint(vect2, point=vect2.get_start()), Write(vect2_name), run_time=2)
        self.wait()

        self.play(GrowFromPoint(vect2_negative, point=vect2_negative.get_start()), Write(vect2_negative_name), run_time=2)
        self.wait()

        self.play(LaggedStart(GrowFromPoint(vect3, point=vect3.get_start())), Write(vect3_name), run_time=3, lag_ratio=1)
        self.wait(4)


if __name__ == '__main__':
    # Generate animated gif.
    config.background_color = WHITE
    config.pixel_height = 300
    config.pixel_width = 600

    config.frame_width = 8
    config.frame_height = 10

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/vector-subtract-example')

    config.format = 'gif'

    scene = VectorAddition()
    scene.render()

    # Generate cover png.
    config.save_last_frame = True

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/vector-subtract-cover')

    scene = VectorAddition()
    scene.render()
