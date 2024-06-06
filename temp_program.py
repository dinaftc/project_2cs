
import unittest

class SquareRoot:
    @staticmethod
    def squareRoot(val):
        i = 1
        v = 0
        res = 0

        while v < val:
            v = v + 2 * i + 1
            i = i + 1

        # error: the instruction should be res = i - 1
        res = i 
        return res

class SquareRootTest(unittest.TestCase):
    def test_0(self):
        result = SquareRoot.squareRoot(9)
        self.assertEqual(result, 3)  # 3 is the integer square root of 9

    def test_1(self):
        result = SquareRoot.squareRoot(15)
        self.assertEqual(result, 3)  # 3 is the integer square root of 15

    def test_2(self):
        result = SquareRoot.squareRoot(16)
        self.assertEqual(result, 4)  # 4 is the integer square root of 16

    def test_3(self):
        result = SquareRoot.squareRoot(20)
        self.assertEqual(result, 4)  # 4 is the integer square root of 20

    def test_4(self):
        result = SquareRoot.squareRoot(25)
        self.assertEqual(result, 5)  # 5 is the integer square root of 25

    def test_5(self):
        result = SquareRoot.squareRoot(30)
        self.assertEqual(result, 5)  # 5 is the integer square root of 30

    def test_6(self):
        result = SquareRoot.squareRoot(36)
        self.assertEqual(result, 6)  # 6 is the integer square root of 36

    def test_7(self):
        result = SquareRoot.squareRoot(40)
        self.assertEqual(result, 6)  # 6 is the integer square root of 40

    def test_8(self):
        result = SquareRoot.squareRoot(49)
        self.assertEqual(result, 7)  # 7 is the integer square root of 49

    def test_9(self):
        result = SquareRoot.squareRoot(50)
        self.assertEqual(result, 7)  # 7 is the integer square root of 50

if __name__ == '__main__':
    unittest.main()