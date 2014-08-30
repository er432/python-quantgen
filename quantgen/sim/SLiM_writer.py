from Ranger import RangeMap
from Ranger import Range
import re
import os

## Initializes a generator for a SLiM file
class SLiM_generator:
    """ Class for generating a SLiM input file
    """
    ## The probability of a mutation per nucleotide per generation
    mu = 1.e-8
    ## The number of generations to run the simulation
    generations = 1e6
    ## The length of the chromosome in bp
    chrom_length = 1e7
    ## The mutation types allowed
    mutation_types = {'neutral':('m1',0.,'f',0.)}
    ## The recombination rate in intervals over the chromosome
    # as an instance of Ranger.RangeMap
    recomb_rates = None
    ## The types of genomic elements allowed
    element_types = {}
    ## Elements along the chromosome
    elements = []
    ## The populations to use in the simulation
    populations = {}
    ## Optional output settings to use in the simulation
    output_settings = []
    ## Optional gene conversion settings to use in the simulation
    gene_conversion_settings = []
    ## Optional predetermined mutations to introduce during the simulation
    predetermined_mutations = []
    ## An optional file of genomes to use to initialize the population
    initialization_file = None
    ## An optional random seed to use for simulation initialization
    seed = None
    
    ## Instantiates an object capable of generating a SLiM file
    # Also adds a single neutral mutation type as m1
    # @param mu The mutation probability per nucleotide per generation
    # @param generations The number of generations to run the simulation
    # @param chrom_length The length of the chromosome in bp
    # @param recomb_rate The basal recombination rate as probability
    # of a crossover event per bp
    # @param starting_pop_size The effective size of the intial population. Note
    # that you can add more if you want and/or change its parameters
    # @param starting_pop_nickname A nickname for the starting population
    def __init__(self, mu=1.e-8, generations=1e6, chrom_length=1e7, \
                 recomb_rate=1.e-7, starting_pop_size = 10000, \
                 starting_pop_nickname='start'):
        """ Instantiates an object capable of generating a SLiM file.

        Also adds a single neutral mutation type as m1 (nickname 'neutral')

        Parameters
        ----------
        mu : float
            The mutation probability per nucleotide per generation
        generations : int
            The number of generations to run the simulation
        chrom_length : int
            The length of the chromosome in bp
        recomb_rate : float
            The basal recombination rate as a probability of a crossover event
            per bp
        starting_pop_size : int
            The effective size of the initial population. Note that you can add
            more if you want and/or change its parameters
        starting_pop_nickname : str
            A nickname for the starting population
        """
        self.mu = mu
        self.generations = generations
        self.mutation_types = {'neutral':('m1',0,'f',0.)}
        self.recomb_rates = RangeMap(rangeDict={
            Range.closed(1,int(chrom_length)): recomb_rate
        })
        self.chrom_length = chrom_length
        self.populations[starting_pop_nickname] = [(1,'P','p1',starting_pop_size)]
        self.output_settings = []
        self.element_types = {}
        self.elements = []
        self.gene_conversion_settings = []
        self.predetermined_mutations = []
        self.initialization_file = None
        self.seed = None
                
    ## Adds a chromosomal element to the simulation
    # @param el_type The nickname for the element type for this element
    # @param start The start point for the element on the chromosome
    # @param end The end point for the element on the chromosome
    def add_chromosome_element(self, el_type, start, end):
        """
        Adds a chromosomal element to the simulation

        Parameters
        ----------
        el_type : str
            The nickname for the element type for this element
        start : int
            The start point for the element on the chromosome
        end : int
            The end point for the element on the chromosome
        """
        self.elements.append((self.element_types[el_type][0],start,end))

    ## Adds a genomic element type to the simulation
    # @param nickname A nickname for the genomic element type (doesn't affect the simulation file)
    # @param mut_types The mutation types you want to affect this element type, as a dictionary of
    # nickname -> relative proportion
    def add_element_type(self, nickname, mut_types):
        """
        Adds a genomic element type to the simulation

        Parameters
        ----------
        nickname : str
            A nickname for the genomic element type (doesn't
            affect the simulation file)
        mut_types : dict
            The mutation types you want to affect this element type,
            as a dictionary of nickname -> relative proportion
        """
        # First get the simulation name of the type
        if len(self.element_types) == 0: element_name = 'g1'
        else:
            element_type_vals = sorted(self.element_types.values(), \
                               key=lambda x: int(re.search('(?<=g)\d+',x[0]).group()))
            last_element_num = int(re.search(r'(?<=g)\d+',element_type_vals[-1][0]).group())
            element_name = "g%d" % (last_element_num+1)
        element_list = [element_name]
        for mut_type,prop in mut_types.iteritems():
            element_list.append(self.mutation_types[mut_type][0])
            element_list.append(prop)
        self.element_types[nickname] = tuple(element_list)
        return

    ## Adds an initialization file to start the population at a certain state
    # @param file The name of the file
    def add_initialization_file(self, file):
        """
        Adds an initialization file to start the population at a certain state

        Parameters
        ----------
        file : str
           The name of the file
        """
        self.initialization_file = file
        return
    
    ## Adds a specific recombination rate to a given interval.
    # Note that the remaining pieces of the chromosome, unspecified
    # will remain at the basal rate
    # @param start The start of the interval (inclusive)
    # @param end The end of the interval (inclusive)
    # @param recomb_rate The recombination rate in the interval
    # as the probability of a crossover event per bp
    def add_interval_recomb_rate(self,start,end,recomb_rate):
        """
        Adds a specific recombination rate to a given interval.
        Note that the remaining pieces of the chromosome, unspecified,
        will remain at the basal rate

        Parameters
        ----------
        start : int
            The start of the interval (inclusive)
        end : int
            The end of the interval (inclusive)
        recomb_rate : float
            The recombination rate in the interval as the
            probability of a crossover event in bp
        """
        self.recomb_rates[Range.closed(start,end)] = recomb_rate
    
    ## Adds a mutation type to the simulation
    # @param nickname A nickname for the mutation type (doesn't affect the simulation file)
    # @param h The dominance coefficient (0.5 would be codominant)
    # @param dfe_type 'f' for fixed, 'e' for exponential, and 'g' for gamma
    # @param mean_s The mean selection coefficent for the distribution.
    # Note that w = 1+hs
    # @param shape_alpha The shape (alpha) parameter for the gamma distribution with mean s=(alpha)(beta)
    def add_mutation_type(self, nickname, h, dfe_type, mean_s=0., shape_alpha=None):
        """ Adds a mutation type to the simulation

        Parameters
        ----------
        nickname : str
            A nickname for the mutation type (doesn't affect the simulation file)
        h : float
            The dominance coefficient (0.5 would be codominant)
        dfe_type : str
            'f' for fixed, 'e' for exponential, and 'g' for gamma
        mean_s : float
            The mean selection coefficient for the distribution. Note that w = 1+hs
        shape_alpha : float
            The shape (alpha) parameter for the gamma distribution with mean
            s=s=(alpha)(beta)
        """
        # Get the last mutation type number
        mutation_type_vals = sorted(self.mutation_types.values(), \
                                    key=lambda x: int(re.search('(?<=m)\d+',x[0]).group()))
        last_mutation_num = int(re.search(r'(?<=m)\d+',mutation_type_vals[-1][0]).group())
        # Write for gamma distribution
        if shape_alpha: 
            self.mutation_types[nickname] = ("m%d" % (last_mutation_num+1),h,"g",mean_s,shape_alpha)
        else:
            # Write for non-gamma ditribution
            self.mutation_types[nickname] = ("m%d" % (last_mutation_num+1),h,dfe_type,mean_s)
        return

    ## Adds an output option to the simulation
    # @param generation The generation that the output should take place
    # @param output_type One of 'A' for the entire state of the population,
    # 'R' for a random sample from the population, 'F' for list of all
    # fixed mutations, or 'T' to track mutations of a particular type
    # @param random_sample_pop The nickname of the population from
    # which to take a random sample if using the 'R' option
    # @param random_sample_size The size of the random sample from the
    # population if using the 'R' option
    # @param mutation_track_type The nickname of the mutation type to
    # track if using the 'T' option
    def add_output_option(self, generation, output_type, \
                          random_sample_pop=None, random_sample_size=None, \
                          mutation_track_type=None):
        """ Adds an output option the the simulation

        Parameters
        ----------
        generation : int
            The generation that the output should take place
        output_type : str
            One of 'A' for the entire state of the population,
            'R' for a random sample from the population, 'F' for list of all
            fixed mutations, or 'T' to track mutations of a particular type
        random_sample_pop : float
            The nickname of the population from which to
            take a random sample if using the 'R' option
        random_sample_size : int
            The size of the random sample from the
            population if using the 'R' option
        mutation_track_type : str
            The nickname of the mutation type to track
            if using the 'T' option
        """
        if output_type == 'A':
            self.output_settings.append((generation,'A'))
        elif output_type == 'R':
            assert random_sample_pop, "random_sample_pop must be set for \
              'R' option."
            assert random_sample_size, "random_sample_size must be set for \
              'R' option."
            self.output_settings.append((generation,'R', \
                       self.populations[random_sample_pop][0][2], \
                       random_sample_size))
        elif output_type == 'F':
            self.output_settings.append((generation, 'F'))
        elif output_type == 'T':
            assert mutation_track_type, "mutation_track_type must be set \
            for 'T' option."
            self.output_settings.append((generation, 'T', \
                           self.mutation_types[mutation_track_type][0]))
        else:
            raise ValueError("Unrecognized output option %s." % output_type)
        self.output_settings = sorted(self.output_settings, \
                        key = lambda x: x[0])
        return
    
    ## Adds a new population to the simulation
    # @param pop_nickname A nickname for the new population
    # @param generation_start The generation at which this population
    # should appear (where the first generation is 1)
    # @param pop_size The size of the population
    # @param source_pop An optional specification of the nickname
    # for the source population of this new population
    def add_population(self,pop_nickname,generation_start,pop_size, \
                       source_pop=None):
        """ Adds a new population to the simulation

        Parameters
        ----------
        pop_nickname : str
            A nickname for the new population
        generation_start : int
            The generation at which this population
            should appear (where the first generation is 1)
        pop_size : int
            The size of the population
        source_pop : str
            An optional specification of the nickname for
            the source population of this new population               
        """
        # Get the name of the new population
        population_vals = sorted(self.populations.values(), \
                                 key=lambda x: int(re.search(r'(?<=p)\d+', x).group()))
        pop_name = "p%d" % (int(re.search(r'(?<=p)\d+',population_vals[-1][2]).group())+1) 
        if source_pop:
            self.populations[pop_nickname] = [(generation_start, 'P', \
                                              pop_name, pop_size, \
                                              source_pop)]
        else:
            self.populations[pop_nickname] = [(generation_start, 'P', \
                                              pop_name, pop_size)]
        return

    ## Adds a predetermined mutation to a population at a given time
    # @param generation The generation at which the mutation should be added
    # @param mut_type The nickname for the mutation type to be added
    # @param pos The position of the added mutation
    # @param pop_nickname The nickname of the population to get the mutation
    # @param n_homo The number of homozygotes for the mutation to appear
    # @param n_hetero The number of heterozygotes for the mutation to appear
    # @param partial Should this be a partial sweep, where the selection
    # coefficient of the mutation type applies until it reaches a certain
    # frequency, at which point it drops to zero
    # @param partial_freq If a partial sweep, the frequency at which s
    # is set to zero
    def add_predetermined_mutation(self, generation, mut_type, pos, \
                                   pop_nickname, n_homo, n_hetero, \
                                   partial = False, partial_freq=0.5):
        """ Adds a predetermined mutation to a population at a given time

        Parameters
        ----------
        generation : int
            The generation at which the mutation should be added
        mut_type : str
            The nickname for the mutation type to be added
        pos : int
            The position of the added mutation
        pop_nickname : str
            The nickname of the population to get the mutation
        n_homo : int
            The number of homozygotes for the mutation to appear
        n_hetero : int
            The number of heterozygotes for the mutation to appear
        partial : boolean
            Should this be a partial sweep, where the selection
            coefficient of the mutation type applies until it reaches a certain
            frequency, at which point it drops to zero
            partial_freq If a partial sweep, the frequency at which s is set
            to zero
        """
        if partial:
            self.predetermined_mutations.append((generation, \
                 self.mutation_types[mut_type][0],pos, \
                 self.populations[pop_nickname][0][2],n_homo,n_hetero, \
                 'P',partial_freq))    
        else:
            self.predetermined_mutations.append((generation, \
                 self.mutation_types[mut_type][0],pos, \
                 self.populations[pop_nickname][0][2],n_homo,n_hetero))
        self.predetermined_mutations = sorted(self.predetermined_mutations, \
                      key = lambda x: x[0])
        return
                                   
    ## Changes the migration rate from a source to a target population
    # @param source_pop_nickname The nickname for the source population
    # @param target_pop_nickname The nickname for the target population
    # @param generation_start The generation at which the migration rate
    # should take effect
    # @param rate Fraction of target population made up of migrants from
    # source population each generation
    def change_pop_migration(self, source_pop_nickname, target_pop_nickname, \
                             generation_start, rate):
        """ Changes the migration rate from a source to a target population

        Parameters
        ----------
        source_pop_nickname : str
            The nickname for the source population
        target_pop_nickname : str
            The nickname for the target population
        generation_start : int
            The generation at which the migration rate
            should take effect
        rate : float
            Fraction of targeet population made up of migrants from
            source population each generation
        """
        # Add the new param to the target population queue
        self.populations[target_pop_nickname].append((generation_start, \
                        'M',self.populations[target_pop_nickname][0][2], \
                        self.populations[source_pop_nickname][0][2], \
                        rate))
        # Sort by generation, but ensure the population declaration is
        # always first
        self.populations[pop_nickname] = sorted(self.populations[target_pop_nickname], \
                                                key=lambda x: (x[0], \
                                                0 if x[1] == 'P' else 1))
        return

    ## Changes the selfing rate of a population
    # @param pop_nickname The nickname of the population for which
    # you want to adjust the selfing rate
    # @param generation_start The generation at which the selfing
    # rate should take effect
    # @param new_rate The new selfing rate
    def change_pop_selfing(self, pop_nickname, generation_start, \
                           new_rate):
        """ Changes the selfing rate of a population

        Parameters
        ----------
        pop_nickname : str
            The nickname of the population for which you
            want to adjust the selfing rate
        generation_start : int
            The generation at which the selfing rate
            should take effect
        new_rate : float
            The new selfing rate
        """
        # Add the new param to the poulation queue
        self.populations[pop_nickname].append((generation_start, 'S', \
                  self.populations[pop_nickname][0][2],new_rate))
        # Sort by generation, but ensure the population declaration is
        # always first
        self.populations[pop_nickname] = sorted(self.populations[pop_nickname], \
                                                key=lambda x: (x[0], \
                                                0 if x[1] == 'P' else 1))
    
    ## Changes the size of a population starting at a given generation
    # @param pop_nickname The nickname of the population that you want
    # to change the size of
    # @param generation_start The generation at which the size change
    # should start
    # @param new_size The new size of the population
    def change_pop_size(self, pop_nickname, generation_start, new_size):
        """
        Changes the size of a population starting at a given generation

        Parameters
        ----------
        pop_nickname : str
            The nickname of the population that you want
            to change the size of
        generation_start : int
            The generation at which the size change
            should start
        new_size : int
            The new size of the population
        """
        # Add the size change
        self.populations[pop_nickname].append((generation_start, 'N', \
                                               self.populations[pop_nickname][0][2], \
                                               new_size))
        # Sort by generation, but ensure the population declaration
        # is always first
        self.populations[pop_nickname] = sorted(self.populations[pop_nickname], \
                                                key=lambda x: (x[0], \
                                                0 if x[1] == 'P' else 1))
        return
    
    ## Adds gene conversion to the simulation
    # @param fraction The fraction of recombination events that result
    # in gene conversion rather than crossing over
    # @param mean_stretch The mean length of the conversion stretch, which
    # is drawn from a geometric distribution
    def set_gene_conversion(self, fraction, mean_stretch):
        """ Adds gene conversion to the simulation

        Parameters
        ----------
        fraction : float
            The fraction of recombination events that result in
            gene conversion rather than crossing over
        mean_stretch : int
            The mean length of the conversion stretch, which
            # is drawn from a geometric distribution
        """
        self.gene_conversion_settings = [fraction,mean_stretch]
        return

    ## Sets the seed for the random number generator
    # @param seed A 32-bit integer
    def set_random_seed(self, seed):
        """ Sets the seed for the random number generator

        Parameters
        ----------
        seed : int
            A 32-bit integer
        """
        self.seed = seed
        return

    ## Writes the input file for the SLiM simulation
    # @param file The name for the input file
    # @param max_size The maximum number of bp allowed in any given file.
    # If this is specified, files will be output as file.1, file.2, etc.
    # Also, writing will stop as soon as the total size of the segments
    # are greater than or equal to the specified size.
    def write_input_file(self, file, max_size=None):
        """ Writes the input file for the SLiM simulation

        Parameters
        ----------
        file : str
            The name for the input file
        max_size : str
            The maximum number of bp alloewd in any given file.
            If this is specified, files will be output as file.1, file.2, etc.
            Writing will stop as soon as the total size of the segments are greater
            than or equal to the specified size
        """
        # Keep track of the last genomic element used
        last_element_ind = 0
        # Keep track of the last file written
        last_file_num = 0
        while 1:
            # Keep track of the total size of this written segment
            segment_size = 0
            if max_size is None:
                write_file = open(file,'w')
            else:
                last_file_num += 1
                write_file = open("%s.%d" % (file,last_file_num),'w')
            # Write mutation types
            write_file.write("#MUTATION TYPES\n")
            for k,v in self.mutation_types.iteritems():
                line = []
                for x in v:
                    if type(x) == str: line.append(x)
                    elif type(x) == int: line.append("%d" % x)
                    elif type(x) == float: line.append("%0.5g" % x)
                write_file.write(' '.join(line)+'/ %s \n' % k)
            # Write the mutation rate
            write_file.write("#MUTATION RATE\n")
            write_file.write("%0.5g\n" % self.mu)
            # Write genomic element types
            write_file.write("#GENOMIC ELEMENT TYPES\n")
            for k,v in self.element_types.iteritems():
                line = []
                for x in v:
                    if type(x) == str: line.append(x)
                    elif type(x) == int: line.append("%d" % x)
                    elif type(x) == float: line.append("%0.5g" % x)
                write_file.write(' '.join(line)+ '/ %s \n' % k)
            # Write chromsome organization
            write_file.write("#CHROMOSOME ORGANIZATION\n")
            for element in self.elements[last_element_ind:]:
                write_file.write("%s %d %d\n" % element)
                if max_size:
                    last_element_ind += 1
                    segment_size += (element[2]-element[1]+1)
                    if segment_size >= max_size: break
            # Write the recombination rates
            write_file.write("#RECOMBINATION RATE\n")
            for seg in self.recomb_rates:
                write_file.write("%d %0.5g\n" % (seg.upperEndpoint(),list(self.recomb_rates[seg])[0]))
            # Write the number of generations
            write_file.write("#GENERATIONS\n")
            write_file.write("%d\n" % self.generations)
            # Write the population structure
            write_file.write("#DEMOGRAPHY AND STRUCTURE\n")
            for k,v in self.populations.iteritems():
                write_file.write('/ %s\n' % k)
                for x in v:
                    line = []
                    for param in x:
                        if type(param) == str: line.append(param)
                        elif type(param) == int: line.append("%d" % param)
                        elif type(param) == float: line.append("%0.5g" % param)
                    write_file.write(' '.join(line)+'\n') 
            # Write the output setting if necessary
            if len(self.output_settings) > 0:
                write_file.write("#OUTPUT\n")
                for x in self.output_settings:
                    line = []
                    for e in x:
                        if type(e) == str: line.append(e)
                        elif type(e) == int: line.append("%d" % e)
                        elif type(e) == float: line.append("%0.5g" % e)
                    write_file.write(' '.join(line)+'\n')
            # Write gene conversion option if necessary
            if len(self.gene_conversion_settings) > 0:
                write_file.write("#GENE CONVERSION\n")
                write_file.write("%0.5g %d\n" % self.gene_conversion_settings)
            # Write initialization file if necessary
            if self.initialization_file:
                write_file.write("#INITIALIZATION\n")
                write_file.write("%s\n" % self.initialization_file)
            # Write seed if necessary
            if self.seed:
                write_file.write("#SEED\n")
                write_file.write("%d\n" % self.seed)
            write_file.close()
            # Break from loop if all the genomic elements have been written
            if max_size:
                if last_element_ind == (len(self.elements)): break
            else:
                break
        return
