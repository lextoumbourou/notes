from pathlib import Path

from manim import *

from theme import Theme

class LinearFuncCover(Scene):
    def construct(self):
        cross_product = MathTex(r"f(", r"\mathbf{x}", r")", r" = ", r" \mathbf{a}", r"{\mathbf{x}}", r"^2", r" + ", r" \mathbf{b}", r"\mathbf{x}", "+", r"\mathbf{c}", font_size=130).set_color(Theme.TEXT_COLOR)
        cross_product.set_color_by_tex("{a}", Theme.COLOR_1)
        cross_product.set_color_by_tex("{b}", Theme.COLOR_2)
        cross_product.set_color_by_tex("{c}", Theme.COLOR_3)

        self.add(cross_product)


if __name__ == '__main__':
    config.background_color = WHITE
    config.format = 'gif'
    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/quad-func-cover.png')
    config.pixel_width = 400
    config.pixel_height = 225

    scene = LinearFuncCover()
    scene.render()
