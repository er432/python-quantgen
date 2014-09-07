import numpy as np

class linearGenomicSelection(object):
    """ Class that represents a genomic selection in which where are coefficients
    for markers, and the prediction is based on the dot product of the number of
    alleles at each locus and the coefficients for markers
    """
    def __init__(self, markers, phenotypes):
        """ Instantiates a linearGenomicSelection analysis

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
        if not isinstance(markers, np.ndarray):
            raise TypeError("Markers is not a numpy matrix")
        if not isinstance(phenotypes, np.ndarray):
            raise TypeError("Phenotypes is not a numpy array")
        if len(phenotypes) != markers.shape[0]:
            raise ValueError("Number of samples is not the same in markers and phenotypes")
        # Hold the markers
        self._markers = np.matrix(markers)
        # Hold the phenotypes
        self._phenotypes = phenotypes
        # Hold the best values for each marker
        self._betas = np.zeros(self._markers.shape[1]+1)
        # Hold the variance of the effects
        self._var_alpha = 0.
    def get_betas(self):
        """ Gets the current beta coefficients for the markers

        Returns
        -------
        array of beta coefficients for the markers. Note that the first entry is the
        intercept (mean)
        """
        return self._betas
    def get_var_alpha(self):
        """ Gets the variance of the additive effects at sites contributing to the
        phenotype (sigma^2_alpha)

        Returns
        -------
        The alpha variance
        """
        return self._var_alpha
    def get_marker_matrix(self):
        """ Returns the marker matrix

        Returns
        -------
        Matrix in which each row corresponds to a sample and each column to a marker.
        Entries refer to the count of the alternative allele
        """
        return self._markers
    def get_phenotypes(self):
        """ Returns the phenotypes

        Returns
        -------
        Array of phenotypes
        """
        return self._phenotypes
    def get_predictions(self, markers):
        """ Gets genomic predictions on a new set of markers

        Parameters
        ----------
        markers : numpy matrix
           Matrix in which each row corresponds to a sample and each column to
           a marker. Entries refer to the count of the alternative allele

        Returns
        -------
        Array of predictions        
        """
        markers = np.concatenate((np.array([np.ones(markers.shape[0])]).T,
                                  markers),axis=1)
        pred = np.array(np.dot(markers, np.matrix([self._betas]).T).flat)
        return pred
