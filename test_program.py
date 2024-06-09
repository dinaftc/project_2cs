class MidWrongIf3:
    @staticmethod
    def mid(a, b, c):
        m = c

        if b < c:
            if a < b:
                m = b
            elif a < c:
                m = a
        else:
            if a < b: # (a > b)
                m = b
            elif a > c:
                m = a

        return m


import unittest

class TestProgram(unittest.TestCase):
    def test_case_0(self):
        result = MidWrongIf3.mid(5, 10, 7)
        self.assertEqual(result, 7)

    def test_case_1(self):
        result = MidWrongIf3.mid(10, 5, 7)
        self.assertEqual(result, 7)

    def test_case_2(self):
        result = MidWrongIf3.mid(5, 7, 10)
        self.assertEqual(result, 7)

    def test_case_3(self):
        result = MidWrongIf3.mid(2, 3, 10)
        self.assertEqual(result, 3)

    def test_case_4(self):
        result = MidWrongIf3.mid(2, 10, 3)
        self.assertEqual(result, 3)

    def test_case_5(self):
        result = MidWrongIf3.mid(10, 2, 3)
        self.assertEqual(result, 3)

    def test_case_6(self):
        result = MidWrongIf3.mid(5, 5, 5)
        self.assertEqual(result, 5)

    def test_case_7(self):
        result = MidWrongIf3.mid(1, 5, 7)
        self.assertEqual(result, 5)

    def test_case_8(self):
        result = MidWrongIf3.mid(7, 5, 1)
        self.assertEqual(result, 5)


if __name__ == "__main__":
    unittest.main()
