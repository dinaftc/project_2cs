import unittest
class Gcd:
    @staticmethod
    def gcd(a, b):
        if a == 0:
            return b
        while b != 0:
            if a > b:
                a = b - b
            else:
                b = b - a
        return a
    


    class GeneratedTest(unittest.TestCase):
        def test_0(self):
            result = Gcd.gcd(0, 0)
            self.assertEqual(0, result)
        def test_1(self):
            result = Gcd.gcd(5, 5)
            self.assertEqual(5, result)
        def test_2(self):
            result = Gcd.gcd(10, 10)
            self.assertEqual(10, result)
        def test_3(self):
            result = Gcd.gcd(15, 15)
            self.assertEqual(15, result)
        def test_4(self):
            result = Gcd.gcd(20, 20)
            self.assertEqual(20, result)
        def test_5(self):
            result = Gcd.gcd(5, 10)
            self.assertEqual(5, result)
        def test_6(self):
            result = Gcd.gcd(10, 15)
            self.assertEqual(5, result)
        def test_7(self):
            result = Gcd.gcd(15, 20)
            self.assertEqual(5, result)
        def test_8(self):
            result = Gcd.gcd(0, 10)
            self.assertEqual(10, result)
        def test_9(self):
            result = Gcd.gcd(5, 15)
            self.assertEqual(5, result)
        def test_10(self):
            result = Gcd.gcd(10, 20)
            self.assertEqual(10, result)
        def test_11(self):
            result = Gcd.gcd(20, 0)
            self.assertEqual(20, result)
        def test_12(self):
            result = Gcd.gcd(0, 15)
            self.assertEqual(15, result)
        def test_13(self):
            result = Gcd.gcd(5, 20)
            self.assertEqual(5, result)
        def test_14(self):
            result = Gcd.gcd(15, 0)
            self.assertEqual(15, result)
        def test_15(self):
            result = Gcd.gcd(20, 5)
            self.assertEqual(5, result)
        def test_16(self):
            result = Gcd.gcd(0, 20)
            self.assertEqual(20, result)
        def test_17(self):
            result = Gcd.gcd(10, 0)
            self.assertEqual(10, result)
        def test_18(self):
            result = Gcd.gcd(15, 5)
            self.assertEqual(5, result)
        def test_19(self):
            result = Gcd.gcd(20, 10)
            self.assertEqual(10, result)
        def test_20(self):
            result = Gcd.gcd(5, 0)
            self.assertEqual(5, result)
        def test_21(self):
            result = Gcd.gcd(10, 5)
            self.assertEqual(5, result)
        def test_22(self):
            result = Gcd.gcd(15, 10)
            self.assertEqual(5, result)
        def test_23(self):
            result = Gcd.gcd(20, 15)
            self.assertEqual(5, result)
        def test_24(self):
            result = Gcd.gcd(0, 5)
            self.assertEqual(5, result)

    if __name__ == "__main__":
        unittest.main()