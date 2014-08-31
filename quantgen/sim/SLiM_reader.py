import re
from Ranger import RangeMap
from Ranger import Range

## Compile regular expressions for determining floats and ints
int_re = re.compile(r'^-*[\d]+$')
float_re = re.compile(r'(^-*[\d]+\.[\d]+$|^-*[\d]{1}([\.]?[\d])*(E|e)-[\d]+$)')

## Class used to parser SLiM output files
class slim_parser(object):
    """ Class for parsing SLiM files
    """
    def __init__(self, filename):
        """
        Instantiates a parser for a SLiM output file

        Parameters
        ----------
        file : str
            The name of the SLiM output file
        """
        # Get handle of file
        self.handle = open(filename)
        # Create dictionary of (generation,type) -> byte where generation begins
        self.output_generation_dict = {}
        # Initialize all parameters
        self.mu = None
        self.generations= None
        self.mutation_types = {}
        self.recomb_rates = RangeMap()
        self.chrom_length = None
        self.populations = {}
        self.output_settings = []
        self.element_types = {}
        self.elements = RangeMap()
        self.gene_conversion_settings = []
        self.predetermined_mutations = []
        self.initialization_file = None
        self.seed = None
        # Parse parameters
        current_param = None
        for line in self.handle:
            line = line.strip()
            if line.startswith('#'):
                # Start out new parameter
                current_param = line[1:]
                if current_param.split(' ')[0] == 'OUT':
                    output = self._line_to_typed_tuple(line)
                    self.output_generation_dict[(output[0],output[1])] = self.handle.tell()
            elif current_param == 'MUTATION RATE':
                self.mu = float(line)
            elif current_param == 'MUTATION TYPES':
                mut_type = self._line_to_typed_tuple(line)
                self.mutation_types[mut_type[0]] = mut_type
            elif current_param == 'GENOMIC ELEMENT TYPES':
                element_type = self._line_to_typed_tuple(line)
                self.element_types[element_type[0]] = element_type
            elif current_param == 'CHROMOSOME ORGANIZATION':
                element = self._line_to_typed_tuple(line)
                self.elements[Range.closed(element[1],element[2])] = element[0]
            elif current_param == 'RECOMBINATION RATE':
                recomb_rate = self._line_to_typed_tuple(line)
                if len(self.recomb_rates) == 0:
                    self.recomb_rates[Range.closed(1,recomb_rate[0])] = recomb_rate[1]
                else:
                    self.recomb_rates[Range.openClosed(self.recomb_rates.ranges[-1].upperEndpoint(),
                                                       recomb_rate[0])] = recomb_rate[1]
            elif current_param == 'GENE CONVERSION':
                conversion = self._line_to_typed_tuple(line)
                self.gene_conversion_settings.append(conversion)
            elif current_param == 'GENERATIONS':
                self.generations = int(line)
            elif current_param == 'DEMOGRAPHY AND STRUCTURE':
                demo_info = self._line_to_typed_tuple(line)
                if demo_info[1] == 'P':
                    self.populations[demo_info[2]] = demo_info
    def _line_to_typed_tuple(self, line):
        """ Converts a line in the SLiM file to a tuple with appropriate
        types (e.g. ints where ints should be and floats where floats should be)

        Parameters
        ----------
        line : str
            The raw string (stripped)


        Returns
        -------
        A tuple with the correct types for entries
        """
        line = line.split(' ')
        for i,item in enumerate(line):
            if int_re.match(item):
                line[i] = int(item)
            elif float_re.match(item):
                line[i] = float(item)
        return tuple(line)
    def parse_mutations(self, generation, out_type='A'):
        """ Parses the mutations in the output for a certain generation of
        a certain output type

        Parameters
        ----------
        generation : int
            The generation for the output
        out_type : str
            The output type ('A','R','F')

        Returns
        -------
        generator of mutation_id, mutation_type, mutation_position,
        selection_coeff, h_coeff, count_in_sample
        """
        ## Go to the position in the file where the output starts
        self.handle.seek(self.output_generation_dict[(generation, out_type)])
        for line in self.handle:
            line = self._line_to_typed_tuple(line)
            if line[0] == 'Mutations:': continue
            elif isinstance(line[0], int):
                yield line
            elif line.startswith('Genotypes'):
                break
            elif line.startswith('#'): break
    def parse_genotypes(self, generation, out_type='A'):
        """ Parses the genotypes in the output for a certain generation of
        a certain output type

        Parameters
        ----------
        generation : int
            The generation for the output
        out_type : str
            The output type ('A','R','F')

        Returns
        -------
        Generator of population, haplotype ID, mutations in haplotype
        """
        self.handle.seek(self.output_generation_dict[(generation,out_type)])
        ready = False
        for line in self.handle:
            if line.startswith('Genotypes'):
                ready = True
            elif not ready:
                continue
            elif len(line) < 2: break
            elif line.startswith('#'): break
            else:
                line = line.strip().split(' ')
                pop = line[0].split(':')[0]
                id = line[0].split(':')[1]
                mutations = map(int,line[1:])
                yield pop, id, mutations
            
