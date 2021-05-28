import unittest
from pennies import *

class TestPennies(unittest.TestCase):
    def setUp(self):
        self.exp = Exponenter(1, 31)

    def test_day_fails_if_less_than_28(self):
        self.assertFalse(is_valid_days(27))

    def test_day_succeeds_if_in_range(self):
        self.assertEquals(is_valid_days(29), 29)

    def test_cents_fails_if_string(self):
        self.assertFalse(is_valid_cents('Foo'))

    def test_exponent_generates_value_correctly(self):
        self.assertEquals(self.exp._exponent(), 1073741824)

    def test_exponent_outputs_string_correctly(self):
        self.assertEquals(unicode(self.exp), "$10,737,418.24")

if __name__ == '__main__':
    unittest.main()
