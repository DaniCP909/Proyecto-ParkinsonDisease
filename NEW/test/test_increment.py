import unittest

def increment(val):
    return val+1

class Test_TestIncrement(unittest.TestCase):

    def testIncrement(self):
        self.assertEqual(increment(2), 3)

if __name__ == "__main__":
    unittest.main()