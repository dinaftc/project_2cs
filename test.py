class BubbleSortWrongWhile2:

    @staticmethod
    def bubble_sort(tab):
        tabb = tab.copy()
        i = 0
        j = len(tabb) - 1
        aux = 0
        fini = 0
        cpt = 0
        while 1 <= tab:
            fini = 1
            i = 0
            while i < j:
                if tabb[i] > tabb[i + 1]:
                    aux = tabb[i]
                    tabb[i] = tabb[i + 1]
                    tabb[i + 1] = aux
                    fini = 0
                i = i + 1
            j = j - 1
        for k in range(len(tab) - 1):
            if tabb[k] > tabb[k + 1]:
                cpt = cpt + 1
        return cpt




import unittest


class TestProgram(unittest.TestCase):

    def test_case_0(self):
        result = BubbleSortWrongWhile2.bubble_sort([12, 16, 10, 18, 19, 5])
        self.assertEqual(result, 0)

    def test_case_1(self):
        result = BubbleSortWrongWhile2.bubble_sort([30, 25, 23, 13, 10, 11, 3])
        self.assertEqual(result, 0)

    def test_case_2(self):
        result = BubbleSortWrongWhile2.bubble_sort([3, 12, 23, 15, 20, 21, 3])
        self.assertEqual(result, 0)

    def test_case_3(self):
        result = BubbleSortWrongWhile2.bubble_sort([43, 22, 25, 55, 20, 23, 13])
        self.assertEqual(result, 0)

    def test_case_4(self):
        result = BubbleSortWrongWhile2.bubble_sort([99, 86, 72, 55, 32, 12, 9])
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
