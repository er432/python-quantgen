
class locus(object):
    """ Represents a diallelic locus on a chromosome.

    Note that the comparison functions only compare positions.
    """
    def __init__(self, a, k, genetic_loc=0.):
        """ Instantiates the locus. See Lynch/Walsh pg. 62

        Parameters
        ----------
        a : float
             Half the genotypic value of the homozygote for the second
             allele
        k : float
             The dominance value, where k=0 is no dominance, k=1 is dominance
             of second allele, and k=-1 is dominance of first allele
        """
        self.a = a
        self.k = k
        # The centimorgan location of the locus
        self.genetic_loc = genetic_loc
    def __repr__(self):
        return "Locus(%0.5g cM; a=%0.3g, d=%0.3g)" % (self.genetic_loc,
                                                      self.a, self.k)
    def __hash__(self):
        return super(locus, self).__hash__()
    def __eq__(self, other):
        if not isinstance(other, locus):
            return False
        else:
            return self.genetic_loc == other.genetic_loc
    def __ne__(self, other):
        return not self.__eq__(other)
    def __lt__(self, other):
        if type(other) == int:
            return self.genetic_loc < other
        elif type(other) == float:
            return self.genetic_loc < other
        elif not isinstance(other, locus):
            raise TypeError("Must be locus type")
        return (self.genetic_loc < other.genetic_loc)
    def __gt__(self, other):
        if type(other) == int:
            return self.genetic_loc > other
        elif type(other) == float:
            return self.genetic_loc > other
        elif not isinstance(other, locus):
            raise TypeError("Must be locus type")
        return (self.genetic_loc > other.genetic_loc)
    def __le__(self, other):
        if type(other) == int:
            return self.genetic_loc <= other
        elif type(other) == float:
            return self.genetic_loc <= other
        elif not isinstance(other, locus):
            raise TypeError("Must be locus type")
        return (self.genetic_loc <= other.genetic_loc)
    def __ge__(self, other):
        if type(other) == int:
            return self.genetic_loc >= other
        elif type(other) == float:
            return self.genetic_loc >= other
        elif not isinstance(other, locus):
            raise TypeError("Must be locus type")
        return (self.genetic_loc >= other.genetic_loc)    
    def get_a_d_values(self):
        """ Gets the values of a and d under that allele configuration
        See Lynch/Walsh pg. 62

        Returns
        -------
        a, d
        """
        return (self.a/2., (1+self.k)*self.a)
    def get_alpha(self, p1):
        """ Gets the average effect of allelic substitution, alpha.
        See Lynch/Walsh ph. 68. Under the assumption of random mating,
        this represents the average change in genotypic value that
        results when a B2 allele is randomly substituted for a B1 allele.

        Parameters
        ----------
        p1 : float
            The frequency of allele 1 in the population

        Returns
        -------
        alpha, the average effect of allelic substitution
        """
        return self.a*(1+self.k*(2*p1-1))
    def get_raw_G(self, allele1, allele2):
        """ Gets the value of G for this locus, given a genotype

        Parameters
        ----------
        allele1 : int
            0 or 1 for the first allele
        allele2 : int
            0 or 1 for the second allele

        Returns
        -------
        The value of G
        """
        if allele1 != allele2:
            return(self.a*(1+self.k))
        elif allele1 == 0:
            return 0.
        elif allele1 == 1:
            return 2*self.a
    def get_mu_G(self, p1):
        """ Gets the theoretical average phenotype assuming the locus
        is the only locus affecting the phenotype

        Parameters
        ----------
        p1 : float
            The frequency of allele 1 in the population

        Returns
        -------
        mu_G for the locus
        """
        return 2*(1-p1)*self.a*(1+p1*self.k)
    def get_alpha_vals(self, p1):
        """ Gets the slope of the number of alleles on the phenotypic values
        (the average effects)

        Parameters
        ----------
        p1 : float
            The frequency of allele 1 in the population

        Returns
        -------
        alpha1, alpha2
        """
        alpha = self.get_alpha(p1)
        return p1*alpha,(p1-1)*alpha
    def set_a(self, a):
        """ Sets the value of a for the locus

        Parameters
        ----------
        a : float
             Half the genotypic value of the homozygote for the second
             allele
        """
        self.a = a
    def set_cM(self, location):
        """ Sets the genetic location of the locus

        Parameters
        ----------
        location : float
            The location of the locus in centimorgans
        """
        self.genetic_loc = location
    def set_k(self, k):
        """ Sets the value of k for the locus

        Parameters
        ----------
        k : float
             The dominance value, where k=0 is no dominance, k=1 is dominance
             of second allele, and k=-1 is dominance of first allele        
        """
        self.k = k
