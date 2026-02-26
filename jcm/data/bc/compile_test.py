import unittest
from jcm.data.bc.compile import main

class TestCompileUnit(unittest.TestCase):

    def test_compile(self):
        # Just test that the compile main function runs without error
        exit_code = main([])
        self.assertEqual(exit_code, 0)