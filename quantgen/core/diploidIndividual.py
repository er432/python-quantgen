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
    def get_alleles(self, chrom_name, locus):
        """ Gets the alleles at a locus

        Parameters
        ----------
        chrom_name : hashable
            The name of the chromosome
        locus : hashable
            The id of a locus

        Returns
        -------
        allele in haplotype 1, allele in haplotype 2
        """
        locus_ind = self._haplotype_dict[chrom_name][0].get_locus_ind(locus)
        return self._haplotype_dict[chrom_name][0][locus_ind], self._haplotype_dict[chrom_name][1][locus_ind]
    def get_chromosome_names(self):
        """ Gets the names of the chromsomes in this individual

        Returns
        -------
        List of chromosome names
        """
        return self._haplotype_dict.keys()
    def get_locus_by_name(self, chrom_name, locus_name):
        """ Gets a locus object using its name

        Parameters
        ----------
        chrom_name : hashable
            The name of the chromosome
        locus_name : hashable
            The name of the locus

        Returns
        -------
        The locus object
        """
        return self._haplotype_dict[chrom_name][0].get_locus_by_name(locus_name)
    def get_locus_names(self, chrom_name):
        """ Gets the ordered locus names on a chromosome

        Parameters
        ----------
        chrom_name : hashable
            The name of the chromosome

        Returns
        -------
        The ordered list of locus name on that chromosome
        """
        return self._haplotype_dict[chrom_name][0].get_locus_names()
    def get_theoretical_breeding_value(self, allele_freqs):
        """ Gets the breeding value of the individual under the assumption of random mating
        and the assumption that all locus a parameters are set

        Parameters
        ----------
        allele_freqs : dict
            Dictionary of (chrom_name, locus_name) -> reference (0-allele) frequency.
            Only the loci present in the keys of this dictionary will be used to calculate
            the breeding value

        Returns
        -------
        Theoretical breeding value for the individual
        """
        breeding_value = 0.
        for (chrom, locus), freq in allele_freqs.iteritems():
            alleles = self.get_alleles(chrom, locus)
            locus = self._haplotype_dict[chrom][0].get_locus_by_name(locus)
            alpha_vals = locus.get_alpha_vals(freq)
            breeding_value += alleles.count(0)*alpha_vals[0]+alleles.count(1)*alpha_vals[1]
        return breeding_value
                    
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
    def mate(self, other, obligate = True):
        """ Mates 2 diploidIndividuals to produce a new diploidIndividual

        Parameters
        ----------
        other: diploidIndividual
            The individual to mate with
        obligate : boolean, optional
            If true, requires at least 1 crossover to take place on each chromosome

        Returns
        -------
        The new diploidIndividual resulting from the mating
        """
        new_dict = {}
        for chrom_name, haplotypes in self._haplotype_dict.items():
            # Get gamete chromosomes from first individual
            new_haplotypes1 = haplotype.recombine2(*haplotypes, obligate=obligate)
            # Get gamete chromosome from second individual
            new_haplotypes2 = haplotype.recombine2(*other._haplotype_dict[chrom_name], obligate=obligate)
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
