import unittest
from quantgen.core.chromosome import chromosome
from quantgen.core.locus import locus
from quantgen.core.diploidIndividual import diploidIndividual
from quantgen.core.haplotype import haplotype

debug = False

class diploidIndividualTest(unittest.TestCase):
    """ Unit tests for diploidIndividual.py """
    def setUp(self):
        self.chrom = chromosome()
        self.chrom.add_locus(locus(0.1,0.))
        self.chrom.add_locus(locus(0.4,0.03),pos=5.)
        self.chrom.add_locus(locus(0.004,0.02),pos=5.8)
        self.chrom.add_locus(locus(0.009,0.01), pos=21.8)
        self.haplo1 = haplotype(self.chrom,genos=[0,0,0,0])
        self.haplo2 = haplotype(self.chrom,genos=[1,1,1,1])
        self.individual = diploidIndividual(haplotype_dict={
            1:(self.haplo1,self.haplo2)
        })
    def test_get_raw_nonepistatic_G(self):
        if debug: print("Testing get_raw_nonepistatic_G")
        val = self.individual.get_raw_nonepistatic_G()
        self.assertAlmostEqual(val,0.1+0.4*1.03+0.004*1.02+0.009*1.01)
if __name__ == "__main__":
    debug = True
    unittest.main(exit = False)
