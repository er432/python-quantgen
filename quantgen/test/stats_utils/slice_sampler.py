import unittest
import numpy as np
from scipy.stats import norm
from quantgen.stats_utils.slice_sampler import univariate_slice_sampler

debug = False

class slice_samplerTest(unittest.TestCase):
    """ Unit tests for slice_sampler.py"""
    def setUp(self):
        func = lambda x: norm.logpdf(x, loc=0, scale=5.)
        self.sampler = univariate_slice_sampler(func)
    def test_run_sampler(self):
        if debug: print("Testing run_sampler")
        np.random.seed(12345)
        samples = [x for x in self.sampler.run_sampler(n_samp=1000, w=2.5)]
        self.assertAlmostEqual(np.mean(samples),0.066295876612242136)
        self.assertAlmostEqual(np.std(samples,ddof=1),4.9739808606298874)
if __name__ == "__main__":
    debug = True
    unittest.main(exit=False)
