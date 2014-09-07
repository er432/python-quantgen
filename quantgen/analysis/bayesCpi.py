from quantgen.analysis.linearGenomicSelection import linearGenomicSelection
import numpy as np
import sys

if sys.version_info[0] > 2:
    xrange = range

class bayesCpi(linearGenomicSelection):
    """ Runs the Bayes Cpi analysis
    """
    def __init__(self, markers, phenotypes):
        """ Instantiates a Bayes Cpi analysis

        Parameters
        ----------
        markers : numpy matrix
           Matrix in which each row corresponds to a sample and each column to
           a marker. Entries refer to the count of the alternative allele
        phenotypes : numpy array
           Array of phenotypes, of the same length as the number of rows in markers

        Raises
        ------
        ValueError
            If the number of samples does not match the number of phenotypes        
        """
        super(bayesCpi, self).__init__(markers, phenotypes)
        # Hold estimate of pi
        self._pi = 1.
    def get_pi(self):
        """ Gets the current estimate of pi

        Returns
        -------
        The current estimate of pi, the probability that a marker does NOT contribute
        to the additive variance
        """
        return self._pi
    def run_gibbs(self, n_iter=10000, burn=100, thin=10,
                   pi_init=0.95, var_a = 0.05, nu_alpha=4., return_traces=False,
                   verbosity = 100):
        """ Runs the Bayes Cpi Gibbs sampler (based on Rohan Fernando's R implementation
        at http://www.ans.iastate.edu/stud/courses/short/2009/B-Day3/BayesCPi.R)

        This will set the betas for the analysis to the mean of the retained iterations
        
        Algorithm explained in http://www.ans.iastate.edu/stud/courses/short/2009/B-Day2-3.pdf
        Parameters
        ----------
        n_iter : int
            The number of iterations for sampling
        burn : int
            The number of iterations for burning in (before taking mean of betas)
        thin : int
            The number for thinning (before taking mean of betas)
        pi_init : float
            The initial value for Pi, the probability that a marker does NOT contribute to
            the additive variance
        var_a : float
            The prior variance for additive effects
        nu_alpha : float
            nu Parameter for scaled inverse chi-2 distribution parameterizing
            additive variance prior
        return_traces : boolean
            Whether to return traces for betas and pi
        verbosity : int
            How often to print to screen

        Returns
        -------
        If return_traces is true, the it returns the traces as:
        pi_trace, var_alpha_trace, beta_trace
        """
        # Get alternative allele frequencies
        q = np.array(np.sum(self._markers, axis=0).flat)/(2.*self._markers.shape[0])        
        ## Initiate parameters
        n_markers = self._markers.shape[1]
        n_records = len(self._phenotypes)
        log_pi = np.log(pi_init)
        log_pi_comp = np.log(1.-pi_init)
        pi = float(pi_init)
        x = np.concatenate((np.array([np.ones(n_records)]).T,self._markers),axis=1)
        mean_2pq = 2*np.mean(q*(1-q))
        var_effects = var_a/(n_markers*(1-pi_init)*mean_2pq)
        scale_c = var_effects*(nu_alpha-2.)/nu_alpha        
        b = np.matrix(np.zeros(n_markers+1)).T
        mean_b = np.array(b.flat)
        b[0,0] = np.mean(self._phenotypes)
        var = np.zeros(n_markers)
        piMean = 0.
        meanVar = 0.
        # Keep track of number of iterations to keep
        n_keep = 0        
        ## Make storage parameters
        store_pi = None
        store_betas = None
        store_vars = None
        if return_traces:
            store_pi = np.zeros(n_iter)
            store_betas = []
            store_vars = np.zeros(n_iter)
        # Adjust y
        y_corr = np.matrix([self._phenotypes - np.array(np.dot(x,b).flat)]).T
        ## Run sampling
        for i in xrange(n_iter):
            # Sample error variance
            var_e = np.dot(y_corr.T, y_corr)[0,0]/np.random.chisquare(n_records+3)
            # Sample intercept
            y_corr = y_corr + x[:,0]*b[0,0]
            rhs = np.sum(y_corr)/var_e
            invLhs = 1.0/(n_records/var_e)
            mean = rhs*invLhs
            b[0,0] = np.random.normal(loc=mean, scale=np.sqrt(invLhs))
            y_corr = y_corr - x[:,0]*b[0,0]
            mean_b[0] = mean_b[0] + b[0,0]
            # Sample delta ad effect for each locus
            n_loci = 0
            for loc in xrange(n_markers):
                y_corr = y_corr + x[:,1+loc]*b[1+loc,0]
                rhs = np.dot(x[:,1+loc].T,y_corr)
                xpx = np.dot(x[:,1+loc].T,x[:,1+loc])
                v0 = xpx*var_e
                v1 = ((xpx**2)*var_effects+xpx*var_e)
                log_delta0 = -0.5*(np.log(v0)+(rhs**2)/v0)+log_pi
                log_delta1 = -0.5*(np.log(v1)+(rhs**2)/v1)+log_pi_comp
                prob_delta1 = 1./(1.+np.exp(log_delta0-log_delta1))
                u = np.random.random()
                if u < prob_delta1:
                    n_loci += 1
                    lhs = xpx/var_e + 1./var_effects
                    invLhs = 1./lhs
                    mean = invLhs*rhs/var_e
                    b[1+loc,0] = np.random.normal(loc=mean, scale=np.sqrt(invLhs))
                    y_corr = y_corr - x[:,1+loc]*b[1+loc,0]
                    if i>= burn and (i % thin == 0):
                        mean_b[1+loc] = mean_b[1+loc] + b[1+loc,0]
                    var[loc] = var_effects
                else:
                    b[1+loc,0] = 0.
                    var[loc] = 0.
            if i % verbosity == 0:
                print("Iteration %d, number of loci in model = %d" % (i, n_loci))
            # Sample common variance
            count_loci = 0
            sum_b_sq = 0.
            for loc in xrange(n_markers):
                if var[loc] > 0.:
                    count_loci += 1
                    sum_b_sq += b[1+loc,0]**2
            var_effects = (scale_c*nu_alpha+sum_b_sq)/np.random.chisquare(nu_alpha+count_loci)
            if i >= burn and (i % thin == 0):
                n_keep += 1
                meanVar += var_effects
            # Sample pi
            aa = n_markers - count_loci+1
            bb = count_loci + 1
            pi = np.random.beta(aa,bb)
            if return_traces:
                store_pi[i] = pi
                store_betas.append(np.array(b.flat))
                store_vars[i] = var_effects
            if i >= burn and (i % thin == 0):
                piMean = piMean+pi
            scale_c = ((nu_alpha-2.)/nu_alpha)*(var_a/((1-pi)*n_markers*mean_2pq))
            log_pi = np.log(pi)
            log_pi_comp = np.log(1-pi)
        self._betas = mean_b / n_keep
        self._pi = piMean / n_keep
        self._var_alpha = meanVar / n_keep
        if return_traces:
            return store_pi, store_vars, store_betas
        
        
