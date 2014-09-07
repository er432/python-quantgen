import unittest
import numpy as np
from scipy.stats import norm, pearsonr, multivariate_normal
from quantgen.stats_utils.slice_sampler import univariate_slice_sampler, multivariate_slice_sampler

debug = False

class slice_samplerTest(unittest.TestCase):
    """ Unit tests for slice_sampler.py"""
    def setUp(self):
        func = lambda x: norm.logpdf(x, loc=0, scale=5.)
        self.sampler = univariate_slice_sampler(func)
    def test_run_sampler(self):
        if debug: print("Testing univariate run_sampler")
        np.random.seed(12345)
        samples = [x for x in self.sampler.run_sampler(n_samp=1000, w=2.5)]
        self.assertAlmostEqual(np.mean(samples),0.066295876612242136)
        self.assertAlmostEqual(np.std(samples,ddof=1),4.9739808606298874)
    def test_run_sampler_univariates(self):
        if debug: print("Testing multivariate run_sampler_univariates")
        np.random.seed(12345)
        cov = np.eye(2)
        cov[0,1] = 0.9
        cov[1,0] = 0.9
        rv = multivariate_normal(np.zeros(2),cov)
        sampler = multivariate_slice_sampler(rv.logpdf,2)
        samples = np.array([x for x in sampler.run_sampler_univariates(np.zeros(2),
                                                                       n_samp=1000)])
        corr,pval=pearsonr(samples[:,0],samples[:,1])
        self.assertAlmostEqual(corr,0.89004316524381255)
if __name__ == "__main__":
    debug = True
    unittest.main(exit=False)
