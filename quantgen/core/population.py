from quantgen.core.diploidIndividual import individual, diploidIndividual
from quantgen.core.haplotype import haplotype
from quantgen.core.locus import locus
from quantgen.core.chromosome import chromosome
import numpy as np
import sys

if sys.version_info[0] > 2:
    xrange = range

class population(object):
    """ Represents a collection of individuals
    """
    def __init__(self, individuals = None):
        """ Instantiates a population

        Parameters
        ----------
        individuals : iterabler
            A list of individual objects
        """
        self.individuals = []
        for ind in individuals:
            self.add_individual(ind)
    def add_individual(self, ind):
        """ Adds an individual to the population

        Parameters
        ----------
        ind : individual
            An individual object to add to the population

        Raises
        ------
        TypeError
            If the variable is not an individual
        """
        if not isinstance(ind, individual):
            raise TypeError("%s is not an individual" % type(ind))
        else:
            self.individuals.append(ind)
    def random_mate(self, n, selfing=False):
        """ Randomly selects pairs to be mated until a population of a certain
        size exists

        Parameters
        ----------
        n : int
            The size of the resulting population
        selfing : boolean
            Whether selfing is allowed

        Returns
        -------
        A new population object with the resulting individuals
        """
        individuals = []
        for i in xrange(n):
            if selfing:
                # Pull 2 random indices
                sire,dam = np.random.randint(low=0,high=len(self.individuals),
                                             size=2)
                individuals.append(self.individuals[sire].mate(self.individuals[dam]))
            else:
                sire,dam = np.random.randint(low=0,high=len(self.individuals),
                                             size=2)
                if dam == sire:
                    if sire == (len(self.individuals-1)):
                        sire -=1
                    else:
                        sire += 1
                individuals.append(self.individuals[sire].mate(self.individuals[dam]))
        return population(individuals=individuals)
    def mate_2(self, ind1, ind2, n):
        """ Mates 2 individuals in the population

        Parameters
        ----------
        ind1 : int
            The index for the first individual to mate (sire)
        ind2 : int
            The index for the second individual to mate (dam)
        n : int
            The number of offspring to generate

        Returns
        -------
        A new population object with the resulting offspring
        """
        individuals = []
        for i in xrange(n):
            individuals.append(self.individuals[ind1].mate(self.individuals[ind2]))
        return population(individuals = individuals)
    def set_a(self, a, chrom_name, locus_name):
        """ Sets the value of a for a locus

        Parameters
        ----------
        a : float
            Half the genotypic value of the homozygote for the second
            allele
        """
        self.individuals[0]._haplotype_dict[chrom_name][0].\
          get_locus_by_name(locus_name).set_a(a)
    def set_k(self, k, chrom_name, locus_name):
        """ Sets the vlaue of k for a locus

        Parameters
        ----------
        k : float
             The dominance value, where k=0 is no dominance, k=1 is dominance
             of second allele, and k=-1 is dominance of first allele                
        """
        self.individuals[0]._haplotype_dict[chrom_name][0].\
          get_locus_by_name(locus_name).set_k(k)
    def get_normal_phenotype(self, env=0., sigma_e = 1.,
                             interactions = False):
        """ Gets a phenotype for each individual in the population

        Parameters
        ----------
        env : float
            A systematic environmental effect
        sigma_e : float
            The error standard deviation (assuming Guassian errors with
            mean of 0)
        interactions : boolean
            Whether epistasis should be taken into account (not yet
            implemented)

        Returns
        -------
        Vector of phenotypes, in same order as individuals
        """
        y = np.ones(len(self.individuals))*env
        # Get the systematic genotype effects
        if not interactions:
            for i,ind in enumerate(self.individuals):
                y[i] = ind.get_raw_nonepistatic_G()
        else:
            raise NotImplemented("Interactions not yet implemented")
        # Add on error
        y += np.random.normal(scale=sigma_e,size=len(y))
        return y
class population_builder(object):
    """ Builder for a population of individuals
    """
    def __init__(self):
        """ Instantiates the builder
        """
        ## Create dictionary of chrom_name -> chromosome object
        self.chromosomes = {}
        ## Create list of individuals
        self.individuals = []
        ## Create dictionary of locus_name -> chrom_name
        self.locus_chrom_dict = {}
    def add_chromosome(self, chrom_name):
        """ Adds a chromosome for individual genomes in the population
        being built

        Parameters
        ----------
        chrom_name : hashable
            Name for the chromosome being added

        Returns
        -------
        Instance of builder with chromosome added
        """
        self.chromosomes[chrom_name] = chromosome()
        return self
    def add_locus(self, chrom_name, locus_id, genetic_position, a=0, k=0):
        """ Adds a locus for individual genomes to the population being
        built

        See Lynch/Walsh pg. 62 for genotypic value definitions
        
        Parameters
        ----------
        chrom_name : hashable
            Name for the chromosome being added. If the chromosome
            hasn't already been added, a new one will be added with
            that name
        locus_id : hashable
            A name for the locus being added
        genetic_position : float
            The position of the locus in cM
        a : float
            Half the genotypic value of the homozygote for the second allele
        k : float
            The dominance value, where k=0 is no dominance, k=1 is dominance
            of second allele, and k=-1 is dominance of first allele

        Returns
        -------
        Instance of the builder with the locus added to the chromosome
        """
        if chrom_name not in self.chromosomes:
            self.add_chromosome(chrom_name)
        self.chromosomes[chrom_name].add_locus(locus(a=a,k=k,
                                                     genetic_loc=genetic_position),
                                                     pos=genetic_position,
                                                     locus_id=locus_id)
        self.locus_chrom_dict[locus_id] = chrom_name
        return self
    def add_diploid_individual(self, mutations1, mutations2):
        """ Adds a diploid individual to the population

        Parameters
        ----------
        mutations1 : list
            List of loci_ids, in which the first haplotype carries the
            alternative allele
        mutations2 : list
            List of loci_ids, in which the second haplotype carries the
            alternative allele

        Returns
        -------
        Instance of the builder with the individual added
        """
        ## Make initial haplotypes for the individual, consiting only of the
        # reference alleles
        haplo_lists1 = dict((chrom_name, [0]*len(chrom)) for chrom_name,chrom \
                            in self.chromosomes.items())
        haplo_lists2 = dict((chrom_name, [0]*len(chrom)) for chrom_name,chrom \
                            in self.chromosomes.items())
        for mut in mutations1:
            mut_chrom = self.locus_chrom_dict[mut]
            haplo_lists1[mut_chrom][self.chromosomes[mut_chrom].get_locus_ind(mut)] = 1
        for mut in mutations2:
            mut_chrom = self.locus_chrom_dict[mut]
            haplo_lists2[mut_chrom][self.chromosomes[mut_chrom].get_locus_ind(mut)] = 1
        # Make haplotype dictionary and instantiate individual
        haplotype_dict = {}
        for chrom_name in haplo_lists1:
            haplotype_dict[chrom_name] = (
                haplotype(self.chromosomes[chrom_name],
                          genos=haplo_lists1[chrom_name]),
                haplotype(self.chromosomes[chrom_name],
                          genos=haplo_lists2[chrom_name])
            )
        self.individuals.append(diploidIndividual(haplotype_dict=haplotype_dict))
        return self
    def build(self):
        """ Instantiates a new population consisting of the individuals
        added

        Returns
        -------
        A population of the individuals added
        """
        return population(individuals=self.individuals)
