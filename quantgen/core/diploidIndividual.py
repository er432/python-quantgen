from quantgen.core.haplotype import haplotype
import sys
import numpy as np
from abc import ABCMeta

if sys.version_info[0] > 2:
    xrange = range

class individual:
    __metaclass__ = ABCMeta
    @classmethod
    def __subclasshook__(cls, C):
        if issubclass(C, diploidIndividual):
            return True
        else:
            return False

class diploidIndividual(object):
    """ Represents an individual composed of 1 or more chromosomes,
    each of which has 2 haplotypes
    """
    def __init__(self, haplotype_dict = None):
        """ Instantiates an individual

        Parameters
        ----------
        haplotype_dict : dict
            Dictionary of 'chromosome_name' -> (haplotype1, haplotype2),
            where each haplotype is a haplotype object
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
    def mate(self, other):
        """ Mates 2 diploidIndividuals to produce a new diploidIndividual

        Parameters
        ----------
        other: diploidIndividual
            The individual to mate with

        Returns
        -------
        The new diploidIndividual resulting from the mating
        """
        new_dict = {}
        for chrom_name, haplotypes in self._haplotype_dict.items():
            # Get gamete chromosomes from first individual
            new_haplotypes1 = haplotype.recombine2(*haplotypes)
            # Get gamete chromosome from second individual
            new_haplotypes2 = haplotype.recombine2(*other._haplotype_dict[chrom_name])
            # Select which recombinant chromosomes go into individual
            new_dict[chrom_name] = []
            rands = np.random.rand(2)
            if rands[0] < 0.5:
                new_dict[chrom_name].append(new_haplotypes1[0])
            else:
                new_dict[chrom_name].append(new_haplotypes1[1])
            if rands[1] < 0.5:
                new_dict[chrom_name].append(new_haplotypes2[0])
            else:
                new_dict[chrom_name].append(new_haplotypes2[1])
        return diploidIndividual(haplotype_dict=new_dict)
