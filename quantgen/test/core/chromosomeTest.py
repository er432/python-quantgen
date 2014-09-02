import unittest
import numpy as np
from quantgen.core.locus import locus
from quantgen.core.chromosome import chromosome

debug = False

class chromosomeTest(unittest.TestCase):
    """ Unit tests for chromosome.py """
    def test_add_locus(self):
        if debug: print("Test add locus")
        a = chromosome()
        a.add_locus(locus(0.1,0.))
        a.add_locus(locus(0.009,0.01),pos=21.8)
        a.add_locus(locus(0.004,0.02),pos=5.8)
        a.add_locus(locus(0.4,0.03),pos=5.)
        self.assertEqual(a[1],locus(0.4,0.03,5.))
        self.assertEqual(a[2],locus(0.004,0.02,5.8))
        self.assertEqual(a[3],locus(0.009,0.01,21.8))
    def test_get_locus(self):
        if debug: print("Test get_locus")
        a = chromosome()
        a.add_locus(locus(0.1,0.))
        a.add_locus(locus(0.009,0.01),pos=21.8)
        a.add_locus(locus(0.004,0.02),pos=5.8)
        a.add_locus(locus(0.4,0.03),pos=5.)
        self.assertEqual(a.get_locus(1),locus(0.4,0.03,5.))
    def test_get_locus_by_name(self):
        if debug: print("Test get_locus_by_name")
        a = chromosome()
        a.add_locus(locus(0.1,0.))
        a.add_locus(locus(0.009,0.01),pos=21.8)
        a.add_locus(locus(0.004,0.02),pos=5.8)
        a.add_locus(locus(0.4,0.03),pos=5.)
        self.assertEqual(a.get_locus_by_name(3),locus(0.4,0.03,5.))            
    def test_get_bordering_loci(self):
        if debug: print("Test get_bordering_loci")
        a = chromosome()
        a.add_locus(locus(0.1,0.))
        a.add_locus(locus(0.009,0.01),pos=21.8)
        a.add_locus(locus(0.004,0.02),pos=5.8)
        a.add_locus(locus(0.4,0.03),pos=5.)
        locus1,locus2=a.get_bordering_loci(2.)
        self.assertEqual(locus1, locus(0.1,0.,0.))
        self.assertEqual(locus2, locus(0.4,0.03,5.))
    def test_get_bordering_loci_inds(self):
        if debug: print("Test get_bordering_loci_inds")
        a = chromosome()
        a.add_locus(locus(0.1,0.))
        a.add_locus(locus(0.009,0.01),pos=21.8)
        a.add_locus(locus(0.004,0.02),pos=5.8)
        a.add_locus(locus(0.4,0.03),pos=5.)
        locus1,locus2=a.get_bordering_loci_inds(2.)
        self.assertEqual(locus1, 0)
        self.assertEqual(locus2, 1)
    def test_get_chrom_length(self):
        if debug: print("Test get_chrom_length")
        a = chromosome()
        a.add_locus(locus(0.1,0.))
        a.add_locus(locus(0.009,0.01),pos=21.8)
        a.add_locus(locus(0.004,0.02),pos=5.8)
        a.add_locus(locus(0.4,0.03),pos=5.)
        self.assertEqual(21.8, a.get_chrom_length())        
    def test_recomb_points(self):
        if debug: print("Test get_recomb_points")
        a = chromosome()
        a.add_locus(locus(0.1,0.))
        a.add_locus(locus(0.4,0.03),pos=5.)
        a.add_locus(locus(0.004,0.02),pos=5.8)
        a.add_locus(locus(0.009,0.01),pos=21.8)
        a.add_locus(locus(0.9,0.01),pos=25.8)
        a.add_locus(locus(0.9,-0.01),pos=31.2)
        a.add_locus(locus(0.23,-0.01),pos=50.2)
        a.add_locus(locus(0.002,-0.05),pos=70.2)
        np.random.seed(12345)
        recomb_points = a.get_recomb_points()
        self.assertEqual(len(recomb_points),1)
        self.assertAlmostEqual(65.25904971576378, recomb_points[0])
if __name__ == "__main__":
    debug = True
    unittest.main(exit=False)












