from pathlib import Path

from manim import *

class DotProductCover(Scene):
    def construct(self):
        dot_product = MathTex("\mathbf{a} \cdot \mathbf{b} = a_1 \cdot b_1 + ... + a_N \cdot b_N", font_size=100).set_color("#333").move_to(ORIGIN)
        self.add(dot_product)


if __name__ == '__main__':
    config.background_color = WHITE
    config.format = 'gif'
    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/dot-product-cover')
    config.pixel_width = 400
    config.pixel_height = 225

    scene = DotProductCover()
    scene.render()
