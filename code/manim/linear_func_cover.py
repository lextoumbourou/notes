from pathlib import Path

from manim import *

from theme import Theme

class LinearFuncCover(Scene):
    def construct(self):
        cross_product = MathTex(r"f(", r"\mathbf{x}", r")", r" = ", r" \mathbf{m}", r"\mathbf{x}", r" + ", r" \mathbf{b}", font_size=200).set_color(Theme.TEXT_COLOR)
        cross_product.set_color_by_tex("{m}", Theme.COLOR_1)
        cross_product.set_color_by_tex("{x}", Theme.COLOR_2)
        cross_product.set_color_by_tex("{b}", Theme.COLOR_3)

        self.add(cross_product)


if __name__ == '__main__':
    config.background_color = WHITE
    config.format = 'gif'
    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/linear-func-cover.png')
    config.pixel_width = 400
    config.pixel_height = 225

    scene = LinearFuncCover()
    scene.render()
