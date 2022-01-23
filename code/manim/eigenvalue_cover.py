from pathlib import Path

from manim import *

from theme import Theme

class EigenVectorCover(Scene):
    def construct(self):
        cross_product = MathTex(r"A", r"\vec{x}", r" = ", r"\lambda", r"\vec{x}", font_size=300).set_color(Theme.TEXT_COLOR)
        cross_product.set_color_by_tex(r"\lambda", Theme.COLOR_3)

        self.add(cross_product)


if __name__ == '__main__':
    config.background_color = WHITE
    config.format = 'gif'
    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/eigenvalue-cover')
    config.pixel_width = 400
    config.pixel_height = 225

    scene = EigenVectorCover()
    scene.render()
