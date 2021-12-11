import math
from pathlib import Path

from manim import *

from theme import Theme

def cross_product(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2] , a[0] * b[1] - a[1] * b[0]])


class CrossProduct(ThreeDScene):
    phi = 45 * DEGREES
    theta = -75 * DEGREES
    gamma = 0 * DEGREES

    def construct(self):
        a0 = ValueTracker(2)
        a1 = ValueTracker(0)
        a2 = ValueTracker(0)
        b2 = ValueTracker(0)

        axes = ThreeDAxes(z_axis_config={'stroke_width': 1, 'stroke_color': BLACK})

        def _get_vects():
            a = np.array([a0.get_value(), a1.get_value(), a2.get_value()])
            b = np.array([0, 2, b2.get_value()])
            return a, b

        def _draw_line_a():
            a, b = _get_vects()
            line_a = Line(start=ORIGIN, end=a, stroke_color=Theme.COLOR_1, stroke_width=10).add_tip()
            return line_a

        def _draw_line_b():
            a, b = _get_vects()
            line_b = Line(start=ORIGIN, end=b, stroke_color=Theme.COLOR_2, stroke_width=10).add_tip()
            return line_b

        def _draw_polygon():
            a, b = _get_vects()
            polygon = Polygon(b, b + a, a, ORIGIN, color=Theme.COLOR_3, fill_opacity=0.6, stroke_width=0)
            return polygon

        def _draw_cross_product():
            a, b = _get_vects()
            c = cross_product(a, b)
            magnitude = np.linalg.norm(c)
            stroke_width=10
            if abs(magnitude) < 0.1:
                stroke_width=0
            line = Line(start=ORIGIN, end=c, stroke_color=Theme.COLOR_3, stroke_width=stroke_width)
            if magnitude > 1:
                line.add_tip()
            return line

        line_a = always_redraw(_draw_line_a)
        line_b = always_redraw(_draw_line_b)
        line_a.z_index = 1
        line_b.z_index = 1
        polygon = always_redraw(_draw_polygon)
        output = always_redraw(_draw_cross_product)

        numberplane = NumberPlane(
            x_range=(-14, 14, 1), y_range=(-8, 8, 1),
            background_line_style={
                "stroke_opacity": 0.4
            },
            axis_config={
                "stroke_opacity": 1,
                "stroke_color": BLUE_D,
                "stroke_opacity": 0.7
            }
        )

        equation = MathTex(r"\vec{c}", r" = ", r" \vec{a} ", r" \times ", r" \vec{b}", font_size=96).set_color(Theme.TEXT_COLOR)
        equation.to_edge(UR)

        equation.set_color_by_tex("{a}", Theme.COLOR_1)
        equation.set_color_by_tex("{b}", Theme.COLOR_2)
        equation.set_color_by_tex("{c}", Theme.COLOR_3)

        self.set_camera_orientation(phi=self.phi, theta=self.theta, gamma=self.gamma, distance=0.1)

        self.begin_ambient_camera_rotation(rate=PI / 30)
        self.add_fixed_in_frame_mobjects(equation)

        self.add(line_a, line_b, polygon, output, numberplane, equation)

        self.play(a0.animate.set_value(0), a1.animate.set_value(2), run_time=3)
        self.play(a0.animate.set_value(-2), a1.animate.set_value(0), run_time=3)
        self.play(a2.animate.set_value(1), b2.animate.set_value(-1), run_time=3)


if __name__ == '__main__':
    config.background_color = Theme.BG_COLOR

    config.pixel_width = 640
    config.pixel_height = 360

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/cross-product')

    scene = CrossProduct()
    scene.render()
