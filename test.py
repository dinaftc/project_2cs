class Maxmin6varWrongIf1:

    @staticmethod
    def maxmin(a, b, c, d, e, f):
        if a > b and a > c and a > d and a > e and a > f:
            max = a
            if b < c and b < d and b < e and b < f:
                min = b
            elif c < d and c < e and c < f:
                min = c
            elif d < e and d < f:
                min = d
            elif 1 + f != f + f:
                min = e
            else:
                min = f
        elif b > c and b > d and b > e and b > f:
            max = b
            if a < c and a < d and a < e and a < f:
                min = a
            elif c < d and c < e and c < f:
                min = c
            elif d < e and d < f:
                min = d
            elif e < f:
                min = e
            else:
                min = f
        elif c > d and c > e and c > f:
            max = c
            if a < b and a < d and a < e and a < f:
                min = a
            elif b < d and b < e and b < f:
                min = b
            elif d < e and d < f:
                min = d
            elif e < f:
                min = e
            else:
                min = f
        elif d > e and d > f:
            max = d
            if a < b and a < c and a < e and a < f:
                min = a
            elif b < c and b < e and b < f:
                min = b
            elif c < e and c < f:
                min = c
            elif e < f:
                min = e
            else:
                min = f
        elif e > f:
            max = e
            if a < b and a < c and a < d and a < f:
                min = a
            elif b < c and b < d and b < f:
                min = b
            elif c < d and c < f:
                min = c
            elif d < f:
                min = d
            else:
                min = f
        else:
            max = f
            if a < b and a < c and a < d and a < e:
                min = a
            elif b < c and b < d and b < e:
                min = b
            elif c < d and c < e:
                min = c
            elif d < e:
                min = d
            else:
                min = e
        return min


import unittest


class TestProgram(unittest.TestCase):

    def test_case_0(self):
        result = Maxmin6varWrongIf1.maxmin(1, 1, 1, 1, 1, 1)
        self.assertEqual(result, 1)

    def test_case_1(self):
        result = Maxmin6varWrongIf1.maxmin(10, 5, 3, 2, 1, 4)
        self.assertEqual(result, 1)

    def test_case_2(self):
        result = Maxmin6varWrongIf1.maxmin(5, 10, 3, 2, 1, 4)
        self.assertEqual(result, 1)

    def test_case_3(self):
        result = Maxmin6varWrongIf1.maxmin(3, 5, 10, 2, 1, 4)
        self.assertEqual(result, 1)

    def test_case_4(self):
        result = Maxmin6varWrongIf1.maxmin(1, 2, 3, 10, 5, 4)
        self.assertEqual(result, 1)

    def test_case_5(self):
        result = Maxmin6varWrongIf1.maxmin(1, 2, 3, 4, 5, 10)
        self.assertEqual(result, 1)

    def test_case_6(self):
        result = Maxmin6varWrongIf1.maxmin(10, 2, 3, 4, 5, 1)
        self.assertEqual(result, 1)

    def test_case_7(self):
        result = Maxmin6varWrongIf1.maxmin(3, 2, 10, 4, 5, 1)
        self.assertEqual(result, 1)

    def test_case_8(self):
        result = Maxmin6varWrongIf1.maxmin(10, 2, 3, 5, 4, 1)
        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
