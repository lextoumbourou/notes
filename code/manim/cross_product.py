from pathlib import Path

from manim import *

from theme import Theme

def cross_product(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2] , a[0] * b[1] - b[0] * a[1]])


class CrossProduct(ThreeDScene):
    def construct(self):
        a = np.array([0, 2, 1])
        b = np.array([0, 0, 2])

        c = cross_product(a, b)

        line_a = Line(start=ORIGIN, end=a, stroke_color=Theme.COLOR_1, stroke_width=10).add_tip()
        line_b = Line(start=ORIGIN, end=b, stroke_color=Theme.COLOR_2, stroke_width=10).add_tip()

        cross = Line(start=ORIGIN, end=c, stroke_color=Theme.COLOR_3, stroke_width=10).add_tip()

        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        polygon = Polygon(b, b + a, a, ORIGIN, color=Theme.COLOR_1, fill_opacity=0.2, stroke_width=0)

        self.add(line_a, line_b, cross, polygon)


if __name__ == '__main__':
    config.background_color = Theme.BG_COLOR

    config.pixel_width = 640
    config.pixel_height = 360

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/cross-product')

    config.format = 'gif'

    scene = CrossProduct()
    scene.render()
