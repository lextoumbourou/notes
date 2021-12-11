from pathlib import Path

from manim import *

from theme import Theme

class CrossProductCover(Scene):
    def construct(self):
        cross_product = MathTex(r"\vec{c}", r" = ", r" \vec{a} ", r" \times ", r" \vec{b}", font_size=300).set_color(Theme.TEXT_COLOR)
        cross_product.set_color_by_tex("{a}", Theme.COLOR_1)
        cross_product.set_color_by_tex("{b}", Theme.COLOR_2)
        cross_product.set_color_by_tex("{c}", Theme.COLOR_3)

        self.add(cross_product)


if __name__ == '__main__':
    config.background_color = WHITE
    config.format = 'gif'
    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/cross-product-cover')
    config.pixel_width = 400
    config.pixel_height = 225

    scene = CrossProductCover()
    scene.render()
