from bisect import bisect_left
from scipy.stats import expon
import numpy as np
import sys

if sys.version_info[0] > 2:
    xrange = range

class chromosome(object):
    """ Represents a chromosome (linkage group).
    Nodes in the chromosome are loci, with edges in between
    representing some type of distance (genetic or physical)
    """
    def __init__(self):
        """ Initializes chromosome object
        """
        ## Create graph dictionary of locus -> locus
        self.adj = {}
        ## Ordered list of loci
        self.lociList = []
        ## Dictionary of locus_id -> index in chromosome
        self.locus_id_dict = {}
        ## Locus ids in same order as loci
        self.locus_id_list = []
        ## Keep track of the current id
        self.current_default_id = 0
    def __len__(self):
        return len(self.lociList)
    def __getitem__(self, ind):
        return self.lociList[ind]
    def add_locus(self, locus, pos=0, locus_id = None):
        """ Adds the locus at a given genetic position

        Parameters
        ----------
        locus : locus object
            A locus object
        pos : float
            The location of the locus in cM
        locus_id : hashable, optional
            The name of the locus. If left as None, will be given an integer
            id

        Raises
        ------
        ValueError
            If trying to put in a position > 0 when there are no loci
            in the chromosome yet or if trying to add a locus at the exact
            position of another locus
        """
        if len(self) == 0 and pos != 0:
            raise ValueError("The first added locus must be 0")
        if len(self) == 0:
            self.lociList.append(locus)
            if locus_id:
                self.locus_id_dict[locus_id] = 0
                self.locus_id_list.append(locus_id)
            else:
                # Ensure ID is unique, and then add it
                while self.current_default_id in self.locus_id_dict:
                    self.current_default_id += 1
                self.locus_id_list.append(self.current_default_id)
                self.locus_id_dict[self.current_default_id] = 0
                self.current_default_id += 1
        else:
            locus.set_cM(pos)
            # Figure out where to insert
            insertion_pt = bisect_left(self.lociList, locus)
            # Raise valueerror if loci exactly overlap
            if insertion_pt != 0:
                if locus == self.lociList[insertion_pt-1]:
                    raise ValueError("Cannot add locus position that already exists")
            if insertion_pt != len(self):
                if locus == self.lociList[insertion_pt]:
                    raise ValueError("Cannot add locus position that already exists")
            if insertion_pt == len(self):
                self.lociList.append(locus)
                self.adj[self.lociList[insertion_pt-1]] = locus
                if locus_id:
                    self.locus_id_dict[locus_id] = len(self.lociList)-1
                    self.locus_id_list.append(locus_id)
                else:
                    # Ensure ID is unique, and then add it
                    while self.current_default_id in self.locus_id_dict:
                        self.current_default_id += 1
                    self.locus_id_list.append(self.current_default_id)
                    self.locus_id_dict[self.current_default_id] = len(self.lociList)-1
                    self.current_default_id += 1
            elif insertion_pt != 0:
                old_val = self.adj[self.lociList[insertion_pt-1]]
                self.adj[self.lociList[insertion_pt-1]] = locus
                self.adj[locus] = self.lociList[insertion_pt]
                self.lociList.insert(insertion_pt, locus)
                # Update indices >= current one
                for i in xrange(insertion_pt, len(self.locus_id_list)):
                    self.locus_id_dict[self.locus_id_list[i]] += 1
                if locus_id:
                    self.locus_id_dict[locus_id] = insertion_pt
                    self.locus_id_list.insert(insertion_pt, locus_id)
                else:
                    while self.current_default_id in self.locus_id_dict:
                        self.current_default_id += 1
                    self.locus_id_dict[self.current_default_id] = insertion_pt
                    self.locus_id_list.insert(insertion_pt, self.current_default_id)
                    self.current_default_id += 1
                    
            else:
                self.adj[locus] = self.lociList[insertion_pt]
                self.lociList.insert(insertion_pt, locus)
                # Update indices >= current one
                for i in xrange(insertion_pt, len(self.locus_id_list)):
                    self.locus_id_dict[self.locus_id_list[i]] += 1
                if locus_id:
                    self.locus_id_dict[locus_id] = insertion_pt
                    self.locus_id_list.insert(insertion_pt, locus_id)
                else:
                    while self.current_default_id in self.locus_id_dict:
                        self.current_default_id += 1
                    self.locus_id_dict[self.current_default_id] = insertion_pt
                    self.locus_id_list.insert(insertion_pt, self.current_default_id)
                    self.current_default_id += 1
    def get_bordering_loci(self, pos):
        """ Gets the loci that surround a given genetic position

        Parameters
        ----------
        pos : float
            The position to query, in centimorgans

        Raises
        ------
        ValueError
            If the position is past the end of the chromosome
        
        Returns
        -------
        locus1, locus2
        """
        if pos < 0:
            raise ValueError("Value must be within the chromosome")
        elif pos > self.lociList[-1].genetic_loc:
            raise ValueError("Value must be within the chromosome")
        # Find the greatest locus <= the position
        lower_locus_ind = max(bisect_left(self.lociList, pos)-1,0)
        if lower_locus_ind == 0:
            return self.lociList[0], self.lociList[1]
        elif lower_locus_ind == (len(self)-1):
            return self.lociList[-2],self.lociList[-1]
        else:
            return self.lociList[lower_locus_ind],self.lociList[lower_locus_ind+1]
    def get_bordering_loci_inds(self, pos):
        """ Gets the indices of the loci that surround a given genetic position

        Parameters
        ----------
        pos : float
            The position to query, in centimorgans

        Raises
        ------
        ValueError
            If the position is past the end of the chromosome
        
        Returns
        -------
        locus ind 1, locus2 ind 2
        """
        if pos < 0:
            raise ValueError("Value must be within the chromosome")
        elif pos > self.lociList[-1].genetic_loc:
            raise ValueError("Value must be within the chromosome")
        # Find the greatest locus <= the position
        lower_locus_ind = max(bisect_left(self.lociList, pos)-1,0)
        if lower_locus_ind == 0:
            return 0,1
        elif lower_locus_ind == (len(self)-1):
            return (len(self)-2,len(self)-1)
        else:
            return lower_locus_ind, (lower_locus_ind+1)
    def get_locus(self, locus_ind):
        """ Gets a locus object by index

        Parameters
        ----------
        locus_ind : int
            The index of the locus

        Returns
        -------

        The locus object at the index
        """
        return self.lociList[locus_ind]
    def get_locus_by_name(self, locus_name):
        """ Gets a locus object by name

        Parameters
        ----------
        locus_name : hashable
            The name of the locus

        Returns
        -------

        The locus object for the name
        """
        return self.lociList[self.locus_id_dict[locus_name]]
    def get_locus_ind(self, locus_name):
        """ Gets the index of a locus object by name

        Parameters
        ----------
        locus_name : hashable
            The name of the locus

        Returns
        -------
        The index of the locus
        """
        return self.locus_id_dict[locus_name]
    def get_chrom_length(self):
        """ Gets the length of the chromosome in centimorgans

        Returns
        -------
        The length of the chromosome, in centimorgans
        """
        return self.lociList[-1].genetic_loc
    def get_recomb_points(self,obligate=True):
        """ Gets recombination point(s) in centimorgans using the exponential
        distribution

        Parameters
        ----------
        obligate : boolean
            If true, requires at least 1 crossover to take place
        
        Returns
        -------
        List of recombination breakpoints in centimorgans
        """
        breakpoints = []
        start_pt = 0
        chrom_length = self.get_chrom_length()
        if obligate:
            start_pt = np.random.rand()*chrom_length
            breakpoints.append(start_pt)
        while 1:
            dist_to_breakpoint = expon.rvs(scale=1./((chrom_length-start_pt)\
                                               /100.))
            if dist_to_breakpoint > 1:
                break
            else:
                breakpoints.append(start_pt+dist_to_breakpoint*(chrom_length-start_pt))
                start_pt = float(breakpoints[-1])
        return breakpoints
            
