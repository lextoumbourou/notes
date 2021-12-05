from pathlib import Path

from manim import *

class CrossProductCover(Scene):
    def construct(self):
        cross_product = MathTex(r"\vec{a} \times \vec{b} = \vec{c}", font_size=300).set_color("#333").move_to(ORIGIN)
        self.add(cross_product)


if __name__ == '__main__':
    config.background_color = WHITE
    config.format = 'gif'
    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/cross-product-cover')
    config.pixel_width = 400
    config.pixel_height = 225

    scene = CrossProductCover()
    scene.render()
