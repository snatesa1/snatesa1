import unittest
from Binary_Sequence import convert_to_binary, find_longest_sequence

class TestBinarySequence(unittest.TestCase):

    def test_convert_to_binary_valid(self):
        """Tests valid decimal to binary conversions."""
        test_cases = [(0, '0'), (1, '1'), (15, '1111'), (156, '10011100')]
        for decimal, expected_binary in test_cases:
            with self.subTest(decimal=decimal):
                result = convert_to_binary(decimal)
                self.assertEqual(result, expected_binary)

    def test_convert_to_binary_invalid(self):
        """Tests for invalid input (non-integer)."""
        invalid_input = "abc"
        result = convert_to_binary(invalid_input)
        self.assertEqual(result, "Invalid input. Please enter a valid integer.")

    def test_find_longest_sequence(self):
        """Tests finding the starting position of the longest 1's sequence."""
        test_cases = [
            ('1101110', 3),    # Sequence in the middle
            ('001111', 2),     # Sequence at the end
            ('111100', 0),     # Sequence at the beginning
            ('101010', 0),     # No continuous sequence
            ('1', 0),          # Single '1'
            ('0', None)        # All '0's
        ]
        for binary_str, expected_start in test_cases:
            with self.subTest(binary_str=binary_str):
                result = find_longest_sequence(binary_str)
                self.assertEqual(result, expected_start)

if __name__ == '__main__':
    unittest.main()
