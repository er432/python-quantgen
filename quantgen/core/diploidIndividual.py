import sys

if sys.version_info[0] > 2:
    xrange = range

class diploidIndividual(object):
    """ Represents an individual composed of 1 or more chromosomes,
    each of which has 2 haplotypes
    """
    def __init__(self, haplotype_dict = None, environment = 0.):
        """ Instantiates an individual

        Parameters
        ----------
        haplotype_dict : dict
            Dictionary of 'chromosome_name' -> (haplotype1, haplotype2),
            where each haplotype is a haplotype object
        environment : float
            The environmental value to add (everything not encompassed by genotype)
        """
        if haplotype_dict:
            self._haplotype_dict = haplotype_dict
        else:
            self._haplotype_dict = {}
    def get_raw_nonepistatic_G(self):
        """ Gets the value of G for the individual, without taking into
        account any epistatic interactions, by adding up the contributions
        from a and k in each locus

        Returns
        -------
        The value of G, taken by adding up the contributions from a and k
        at each locus
        """
        G = 0.
        for chrom, haplotypes in self._haplotype_dict.items():
            for i in xrange(len(haplotypes[0])):
                locus = haplotypes[0].get_locus(i)
                G += locus.get_raw_G(haplotypes[0][i],haplotypes[1][i])
        return G
