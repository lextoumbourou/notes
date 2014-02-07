import unittest
from i_saw_you import *

class TestISeeYou(unittest.TestCase):
    def test_input_negative_fails(self):
        self.assertFalse(check_input(-1))

    def test_input_string_fails(self):
        self.assertFalse(check_input("foo"))

    def test_input_positive_succeeds(self):
        self.assertTrue(check_input(123))

    def test_graph_scale_example_1(self):
        graph = Graph([100, 0, 0, 0])
        self.assertEquals(graph._scale_numbers(), [20, 0, 0, 0])

    def test_graph_scale_example_2(self):
        graph = Graph([5, 5, 0, 0])
        self.assertEquals(graph._scale_numbers(), [10, 10, 0, 0])

    def test_graph_scape_example_3(self):
        graph = Graph([2, 2, 2, 2])
        self.assertEquals(graph._scale_numbers(), [5, 5, 5, 5])

if __name__ == '__main__':
    unittest.main()

