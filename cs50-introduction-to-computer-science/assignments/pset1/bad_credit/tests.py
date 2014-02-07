import unittest
from bad_credit import *

class TestBadCredit(unittest.TestCase):
	def test_valid_credit_card(self):
            self.assertTrue(is_valid_credit_card('378282246310005'))

        def test_another_valid_credit_card(self):
            self.assertTrue(is_valid_credit_card('5555555555554444'))
		
        def test_invalid_card_returns_false(self):
            self.assertFalse(is_valid_credit_card('12335345345'))

        def test_amex_returns_true(self):
            self.assertTrue(is_amex('378282246310005'))
		
        def test_visa_returns_true(self):
            self.assertTrue(is_visa('4940521306376179'))
        
        def test_master_card_returns_true(self):
            self.assertTrue(is_mastercard('5105105105105100'))

if __name__ == '__main__':
	unittest.main()

