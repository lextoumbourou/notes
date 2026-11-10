import unittest

import markdown


class LazyImagesTests(unittest.TestCase):
    def render(self, text):
        return markdown.markdown(text, extensions=['markdown_lazy_images'])

    def test_first_image_stays_eager_and_later_ones_are_lazy(self):
        html = self.render('![one](a.jpg)\n\n![two](b.jpg)\n\n![three](c.jpg)')
        first, second, third = html.split('<img')[1:]
        self.assertNotIn('loading=', first)
        self.assertIn('loading="lazy"', second)
        self.assertIn('loading="lazy"', third)
        self.assertIn('decoding="async"', third)

    def test_single_image_is_untouched(self):
        self.assertNotIn('loading=', self.render('![only](a.jpg)'))


if __name__ == '__main__':
    unittest.main()
