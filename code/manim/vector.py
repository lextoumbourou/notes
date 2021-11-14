from pathlib import Path

from manim import *

from theme import Theme

class VectorArrow(Scene):
    def construct(self):
        dot = Dot(ORIGIN).set_color(Theme.COLOR_1)
        arrow = Arrow(ORIGIN, [1, 2, 0], buff=0).set_color(Theme.COLOR_1)
        numberplane = NumberPlane(
            background_line_style={
                "stroke_opacity": 0.4
            }
        )
        origin_text = Text('Origin', font_size=38).next_to(dot, DOWN).set_color('#333')
        tip_text = MathTex('\\begin{bmatrix}1 \\\\ 2\\end{bmatrix}', font_size=60).next_to(arrow.get_end(), RIGHT).set_color(Theme.COLOR_1)
        self.add(numberplane, dot, arrow, origin_text, tip_text)


if __name__ == '__main__':
    # Generate animated gif.
    config.background_color = WHITE

    config.pixel_width = 300
    config.pixel_height = 300

    config.frame_width = 6
    config.frame_height = 6

    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/vector-cover')

    config.format = 'gif'

    scene = VectorArrow()
    scene.render()
