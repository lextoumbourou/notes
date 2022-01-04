from pathlib import Path

from manim import *

from theme import Theme

class ChangingBasisCover(Scene):
    def construct(self):
        changing_basis = MathTex(r"{A^{-1}}", "{M}", "{A}", " \ ", r"\vec{v}", font_size=250).set_color(Theme.TEXT_COLOR).move_to(ORIGIN)
        changing_basis.set_color_by_tex("{A^{-1}}", Theme.COLOR_1)
        changing_basis.set_color_by_tex("{M}", Theme.COLOR_3)
        changing_basis.set_color_by_tex("{A}", Theme.COLOR_2)
        changing_basis.set_color_by_tex("{v}", Theme.COLOR_4)
        self.add(changing_basis)


if __name__ == '__main__':
    config.background_color = WHITE
    config.format = 'gif'
    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/changing-basis-cover')
    config.pixel_width = 400
    config.pixel_height = 225

    scene = ChangingBasisCover()
    scene.render()
