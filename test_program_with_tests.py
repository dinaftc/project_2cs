class Gcd:

    @staticmethod
    def gcd(a, b):
        if a == 0:
            return b
        while b != 0:
            if a > b:
                a = b - b
            else:
                b = a - a
        return a



import unittest

class TestProgram(unittest.TestCase):
    
    def test_case_0(self):
        result = Gcd.gcd(0, 0)
        self.assertEqual(result, 0)

    def test_case_1(self):
        result = Gcd.gcd(5, 5)
        self.assertEqual(result, 5)

    def test_case_2(self):
        result = Gcd.gcd(10, 10)
        self.assertEqual(result, 10)

    def test_case_3(self):
        result = Gcd.gcd(15, 15)
        self.assertEqual(result, 15)

    def test_case_4(self):
        result = Gcd.gcd(20, 20)
        self.assertEqual(result, 20)

    def test_case_5(self):
        result = Gcd.gcd(5, 10)
        self.assertEqual(result, 5)

    def test_case_6(self):
        result = Gcd.gcd(10, 15)
        self.assertEqual(result, 5)

    def test_case_7(self):
        result = Gcd.gcd(15, 20)
        self.assertEqual(result, 5)

    def test_case_8(self):
        result = Gcd.gcd(0, 10)
        self.assertEqual(result, 10)

    def test_case_9(self):
        result = Gcd.gcd(5, 15)
        self.assertEqual(result, 5)

    def test_case_10(self):
        result = Gcd.gcd(10, 20)
        self.assertEqual(result, 10)

    def test_case_11(self):
        result = Gcd.gcd(20, 0)
        self.assertEqual(result, 20)

    def test_case_12(self):
        result = Gcd.gcd(0, 15)
        self.assertEqual(result, 15)

    def test_case_13(self):
        result = Gcd.gcd(5, 20)
        self.assertEqual(result, 5)

    def test_case_14(self):
        result = Gcd.gcd(15, 0)
        self.assertEqual(result, 15)

    def test_case_15(self):
        result = Gcd.gcd(20, 5)
        self.assertEqual(result, 5)

    def test_case_16(self):
        result = Gcd.gcd(0, 20)
        self.assertEqual(result, 20)

    def test_case_17(self):
        result = Gcd.gcd(10, 0)
        self.assertEqual(result, 10)

    def test_case_18(self):
        result = Gcd.gcd(15, 5)
        self.assertEqual(result, 5)

    def test_case_19(self):
        result = Gcd.gcd(20, 10)
        self.assertEqual(result, 10)

    def test_case_20(self):
        result = Gcd.gcd(5, 0)
        self.assertEqual(result, 5)

    def test_case_21(self):
        result = Gcd.gcd(10, 5)
        self.assertEqual(result, 5)

    def test_case_22(self):
        result = Gcd.gcd(15, 10)
        self.assertEqual(result, 5)

    def test_case_23(self):
        result = Gcd.gcd(20, 15)
        self.assertEqual(result, 5)

    def test_case_24(self):
        result = Gcd.gcd(0, 5)
        self.assertEqual(result, 5)


if __name__ == "__main__":
    unittest.main()
