from quantgen.core.chromosome import chromosome
from collections import deque

class haplotype(chromosome):
    """ Represents the alleles along a single phased haplotype
    """
    def __init__(self, chromosome, genos = None):
        """ Instantiates the haplotype

        Parameters
        ----------
        chromosome : chromosome
            The chromosome giving the structure of the haplotype
        genos : list
            List of the alleles, with one for each locus. 0 represents allele 1
            and 1 represents allele 2. If None, all alleles instantiated to 0
        
        Raises
        ------
        ValueError
            If the number of loci doesn't match the number of genotypes
        """
        self.lociList = chromosome.lociList
        self.adj = chromosome.adj
        if genos is None:
            self.genos = [0]*len(self)
        elif len(genos) != len(self):
            raise ValueError("Number of loci doesn't match number of genotypes")
        else:
            self.genos = genos
    def __getitem__(self, ind):
        return self.genos[ind]
    def __setitem__(self, ind, geno):
        self.genos[ind] = geno
    @staticmethod
    def recombine2(haplo1, haplo2):
        """ Recombines 2 haplotypes

        Parameters
        ----------
        haplo1 : haplotype
            The first haplotype
        haplo2 : haplotype
            The second haplotype

        Raises
        ------
        ValueError
            If the haplotype chromosomes are not compatible
        
        Returns
        -------
        first recombined haplotype, second recombined haplotype    
        """
        if len(haplo1) != len(haplo2):
            raise ValueError("Chromosomes differ in length")
        else:
            # Get recombination breakpoints
            breakpoints = deque(haplo1.get_recomb_points())
            # Create new haplotypes
            new_haplo1 = haplotype(haplo1)
            new_haplo2 = haplotype(haplo2)
            start_ind = 0
            flipped = False
            while len(breakpoints) > 0:
                breakpt = breakpoints.popleft()
                border_ind1, border_ind2 = haplo1.get_bordering_loci_inds(breakpt)
                if flipped:
                    new_haplo1[start_ind:border_ind2] = haplo2[start_ind:border_ind2]
                    new_haplo2[start_ind:border_ind2] = haplo1[start_ind:border_ind2]
                else:
                    new_haplo1[start_ind:border_ind2] = haplo1[start_ind:border_ind2]
                    new_haplo2[start_ind:border_ind2] = haplo2[start_ind:border_ind2]
                flipped = not flipped
                start_ind = border_ind2
            if flipped:
                new_haplo1[start_ind:] = haplo2[start_ind:]
                new_haplo2[start_ind:] = haplo1[start_ind:]
            else:
                new_haplo1[start_ind:] = haplo1[start_ind:]
                new_haplo2[start_ind:] = haplo2[start_ind:]            
        return new_haplo1,new_haplo2
