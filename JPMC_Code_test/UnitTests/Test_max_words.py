import unittest
from Max_words import find_words_with_max_char


class TestMaxWords(unittest.TestCase):

    def test_invalid_char(self):
        sentence = ["apple", "banana"]
        char = "123"  # Non-alphabetic
        result = find_words_with_max_char(sentence, char)
        self.assertEqual(result, (0, []))

    def test_case_insensitive(self):
        sentence = ["Apple", "banana"]
        char = "A"
        result = find_words_with_max_char(sentence, char)
        self.assertEqual(result, (1, ["Apple"]))

    def test_alpha_char(self):
        sentence = ["apple", "bananaaaaaaa"]
        char = "a"  # alphabetic
        result = find_words_with_max_char(sentence, char)
        self.assertEqual(result, (9, ["bananaaaaaaa"]))        

    def test_first_alpha_char(self):
        sentence = ["apple", "bananaaaaaaa", "rananaaaaaaa"]
        char = "a"  # alphabetic
        result = find_words_with_max_char(sentence, char)
        self.assertEqual(result, (9, ["bananaaaaaaa"]))   

if __name__ == "__main__":
    unittest.main()
