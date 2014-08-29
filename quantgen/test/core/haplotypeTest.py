import unittest
import numpy as np
from quantgen.core.chromosome import chromosome
from quantgen.core.haplotype import haplotype
from quantgen.core.locus import locus

debug = False

class haplotypeTest(unittest.TestCase):
    """ Unit tests for haplotype.py """
    def setUp(self):
        self.chrom = chromosome()
        self.chrom.add_locus(locus(0.1,0.))
        self.chrom.add_locus(locus(0.4,0.03),pos=5.)
        self.chrom.add_locus(locus(0.004,0.02),pos=5.8)
        self.chrom.add_locus(locus(0.009,0.01), pos=21.8)
        self.haplo1 = haplotype(self.chrom,genos=[0,0,0,0])
        self.haplo2 = haplotype(self.chrom,genos=[1,1,1,1])
        self.assertEqual(self.haplo1[0],0)
        self.assertEqual(self.haplo2[0],1)
    def test_recombine2(self):
        if debug: print("Testing recombine2")
        np.random.seed(12345)
        recomb_haplo1, recomb_haplo2 = haplotype.recombine2(self.haplo1, self.haplo2)
        self.assertEqual(recomb_haplo1.genos,[0,0,0,1])
        self.assertEqual(recomb_haplo2.genos,[1,1,1,0])
if __name__ == "__main__":
    debug = True
    unittest.main(exit=False)
