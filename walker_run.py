import json
from genotype import *
from phenotype import *

def load_genotype(id):
    gen = Genome(id, defined = True)
    genome = {}
    with open('data1.txt', 'r') as f:
        data = f.readlines()
        for g in data:
            genome = json.loads(g)
            if genome['genome_id'] == id:
                break

    gen.fitness = genome['fitness'] + 200

    gen.neurons = []
    for i in range(len(genome['neurons'])):
        gen.neurons.append(NeuronGen(genome['neurons'][i], genome['neuron_type'][i], genome['neuron_pos'][i][0], genome['neuron_pos'][i][1]))

    gen.links = []
    for i in range(len(genome['from_neuron'])):
        gen.links.append(LinkGen(genome['from_neuron'][i], genome['to_neuron'][i], genome['innovation_id'][i], weight = genome['weight'][i], disabled = genome['disabled'][i]))

    return gen
