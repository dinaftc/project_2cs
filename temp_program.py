class AbsMinusWrongIf2:
    @staticmethod
    def absM(i, j):
        result = 0
        k = 0
        if i <= j:
            k = k + 1
        if k == 0 and i != j:  # (k == 1 and i != j)
            result = j - i
        else:
            result = i - j
        return result


import unittest

class TestProgram(unittest.TestCase):
    def test_case_0(self):
        result = AbsMinusWrongIf2.absM(1, 2)
        self.assertEqual(result, 1)

    def test_case_1(self):
        result = AbsMinusWrongIf2.absM(5, 6)
        self.assertEqual(result, 1)

    def test_case_2(self):
        result = AbsMinusWrongIf2.absM(5, 8)
        self.assertEqual(result, 3)

    def test_case_3(self):
        result = AbsMinusWrongIf2.absM(1, 6)
        self.assertEqual(result, 5)

    def test_case_4(self):
        result = AbsMinusWrongIf2.absM(1, 8)
        self.assertEqual(result, 7)

    def test_case_5(self):
        result = AbsMinusWrongIf2.absM(10, 2)
        self.assertEqual(result, 8)

    def test_case_6(self):
        result = AbsMinusWrongIf2.absM(10, 6)
        self.assertEqual(result, 4)

    def test_case_7(self):
        result = AbsMinusWrongIf2.absM(5, 2)
        self.assertEqual(result, 3)

    def test_case_8(self):
        result = AbsMinusWrongIf2.absM(10, 8)
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
