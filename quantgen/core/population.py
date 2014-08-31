from quantgen.core.diploidIndividual import individual
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
                
