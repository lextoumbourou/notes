from pathlib import Path

from manim import *

class MatrixInverseCover(Scene):
    def construct(self):
        inverse = MathTex("A^{-1}", font_size=500).set_color("#333").move_to(ORIGIN)
        self.add(inverse)


if __name__ == '__main__':
    config.background_color = WHITE
    config.format = 'gif'
    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/matrix-inverse-cover')
    config.pixel_width = 400
    config.pixel_height = 225

    scene = MatrixInverseCover()
    scene.render()
