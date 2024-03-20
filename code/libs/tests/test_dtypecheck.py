import unittest
import dtypecheck as dc


class Test_strlistChecks(unittest.TestCase):
    def test_is_strlist(self):
        self.assertTrue(dc.is_strlist(['bla']))
        self.assertTrue(dc.is_strlist(['bla', 'bla']))
        self.assertFalse(dc.is_strlist([['bla'], ['bla']]))
        self.assertFalse(dc.is_strlist([]))
        self.assertFalse(dc.is_strlist('bla'))
        self.assertFalse(dc.is_strlist(None))

    def test_is_str_or_strlist(self):
        self.assertTrue(dc.is_str_or_strlist('bla'))
        self.assertTrue(dc.is_str_or_strlist(['bla']))
        self.assertTrue(dc.is_str_or_strlist(['bla', 'bla']))
        self.assertFalse(dc.is_str_or_strlist([['bla'], ['bla']]))
        self.assertFalse(dc.is_str_or_strlist([]))
        self.assertFalse(dc.is_str_or_strlist(None))


if __name__ == '__main__':
    unittest.main()
