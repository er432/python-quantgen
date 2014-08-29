import unittest
from quantgen.core.locus import locus

debug = False

class locusTest(unittest.TestCase):
    """ Unit tests for locus.py """
    def setUp(self):
        self.theLocus = locus(0.59, 0.17)
    def test_get_a_d_values(self):
        if debug: print("Testing get_a_d_values")
        a,d=self.theLocus.get_a_d_values()
        self.assertEqual(a,0.59/2)
        self.assertEqual(d, (1+0.17)*0.59)
    def test_get_alpha(self):
        if debug: print("Testing get_alpha")
        alpha1,alpha2 = self.theLocus.get_alpha_vals(0.5)
        self.assertAlmostEqual(alpha1, 0.295, places=3)
        self.assertAlmostEqual(alpha2, -0.295, places=3)
    def test_get_raw_G(self):
        if debug: print("Testing get_raw_G")
        self.assertEqual(self.theLocus.get_raw_G(0,0),0.)
        self.assertAlmostEqual(self.theLocus.get_raw_G(0,1),0.59*(1+0.17))
        self.assertAlmostEqual(self.theLocus.get_raw_G(1,0),0.59*(1+0.17))
        self.assertAlmostEqual(self.theLocus.get_raw_G(1,1),0.59*2)
        

if __name__ == "__main__":
    debug = True
    unittest.main(exit=False)
