from pathlib import Path

from manim import *

class Determinant(Scene):
    def construct(self):
        text_color = "#333"
        vect1_color = "#b98b99"
        vect2_color = "#b9b28b"

        numberplane = NumberPlane(
            background_line_style={
                "stroke_opacity": 0.4
            }
        )

        determinant = MathTex(
            "\\det\\left( \\begin{bmatrix}a && b \\\\ c && d \\end{bmatrix}\\right) = ad - bc", font_size=105
        ).set_color(text_color)
        determinant[0][5].set_color(vect1_color)
        determinant[0][6].set_color(vect2_color)

        determinant[0][7].set_color(vect1_color)
        determinant[0][8].set_color(vect2_color)

        determinant[0][12].set_color(vect1_color)
        determinant[0][13].set_color(vect2_color)

        determinant[0][15].set_color(vect2_color)
        determinant[0][16].set_color(vect1_color)

        determinant.move_to(ORIGIN + UP * 2.25)

        origin = np.array([-6, -3, 0])

        vect_1 = np.array([12, 0, 0])
        vect_2 = np.array([0, 3, 0])

        grid_1 = vect_1 + vect_2
        grid_2 = vect_2 + vect_1

        vect1 = Line(start=origin, end=origin + vect_1, stroke_color=vect1_color, stroke_width=10).add_tip()
        dashed_line1 = DashedLine(start=origin + vect_1, end=origin + grid_1, stroke_color="#ccc", stroke_width=10)

        vect2 = Line(start=origin, end=origin + vect_2, stroke_color=vect2_color, stroke_width=10).add_tip()
        dashed_line2 = DashedLine(start=origin + vect_2, end=origin + grid_2, stroke_color="#ccc", stroke_width=10)

        center_point = origin + ((vect_1 + vect_2) * 0.5)

        area_text = MathTex("ad - bc", font_size=150).set_color("#333").move_to(center_point)
        area_text[0][0].set_color(vect1_color)
        area_text[0][1].set_color(vect2_color)

        area_text[0][3].set_color(vect2_color)
        area_text[0][4].set_color(vect1_color)

        rectangle = Rectangle(width=vect_1[0], height=vect_2[1], color="#ccc").move_to(center_point)

        self.add(numberplane, rectangle, determinant, vect1, vect2, dashed_line1, dashed_line2, area_text)


if __name__ == '__main__':
    config.background_color = WHITE
    config.format = 'gif'
    config.output_file = Path(__file__).resolve().parent.parent.parent / Path('notes/_media/determinant')
    config.pixel_width = 400
    config.pixel_height = 225

    scene = Determinant()
    scene.render()

