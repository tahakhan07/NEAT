import numpy as np
from random import random, randint
from math import sqrt
import json

from innovation import *

def rand_clamp():
    return random()*2 - 1

class NeuronType:
    INPUT, OUTPUT, HIDDEN, BIAS = range(4)

class NeuronGen:
    def __init__(self, neuron_id, neuron_type, pos_x, pos_y, activation_response = 1/4.924273):
        self.id = neuron_id
        self.type = neuron_type
        #self.activation_response = activation_response
        self.pos_x = pos_x
        self.pos_y = pos_y

class LinkGen:
    def __init__(self, neuron1_id, neuron2_id, innovation_id, disabled = False, weight = None, recurrent=False):
        if weight == None:
            weight = rand_clamp()
        self.from_neuron_id = neuron1_id
        self.to_neuron_id = neuron2_id
        self.weight = weight
        self.disabled = disabled
        self.recurrent = recurrent
        self.innovation_id = innovation_id

class Genome:
    def __init__(self, genome_id, innovation_db=None, neurons=None, links=None, n_inputs=2, n_outputs=1, phenotype=None, defined = False):
        self.id = genome_id
        self.innovation_db = innovation_db
        self.neurons = neurons
        self.links = links
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.fitness = 0.0
        self.champion = False
        self.super_champ_offspring = 0
        self.orig_fitness = 0.0
        self.solved = False
        self.adjusted_fitness = 0.0
        self.spawns_required = 0
        self.species_id = None
        self.species = None
        self.eliminate = False
        #self.phenotype = phenotype

        self.stdev_weight = 2.0
        self.stdev_mutate_weight = 1.5

        self.tries_to_find_unlinked_neurons = 5
        self.chance_to_add_recurrent_link = 0
        self.tries_to_find_old_link = 5
        self.chance_to_add_neuron = 0.15 #0.03
        self.max_neurons = float('inf')
        self.chance_to_add_link = 0.35 #0.3
        self.weight_mutation_rate = 0.3 #0.1
        self.chance_to_reset_weight = 0.1
        self.weight_range = (-10., 10.)

        if defined:
            return


        if neurons != None:
            #if np.all([l.disabled for l in links]):
            #    pass
            self.neurons.sort(key = lambda x: x.id)
            #for i in range(len(self.neurons)):
            #    self.neurons[i].idx = i
            return

        input_pos_x = 1./(n_inputs+1)
        output_pos_x = 1./(n_outputs)
        next_neuron_id = 0
        self.neurons = []
        self.neurons.append(NeuronGen(next_neuron_id, NeuronType.BIAS, 0.5*input_pos_x, 0.0))
        next_neuron_id += 1

        for i in range(n_inputs):
            self.neurons.append(NeuronGen(next_neuron_id, NeuronType.INPUT, (i+1+0.5)*input_pos_x, 0.0))
            next_neuron_id += 1

        for i in range(n_outputs):
            self.neurons.append(NeuronGen(next_neuron_id, NeuronType.OUTPUT, (i+0.5)*output_pos_x, 1.0))
            next_neuron_id += 1

        innovation_db.next_neuron_id = max(innovation_db.next_neuron_id, next_neuron_id)

        self.links = []
        for i in [x for x in self.neurons if x.type==NeuronType.INPUT or x.type==NeuronType.BIAS]:
            for o in [x for x in self.neurons if x.type==NeuronType.OUTPUT]:
                innovation = innovation_db.get_innovation(InnovationType.LINK, in_neuron_id = i.id, out_neuron_id = o.id)
                weight = np.random.normal(0, self.stdev_weight)
                self.links.append(LinkGen(i.id, o.id, innovation.link_id, weight = weight))



    def make_copy(self, new_id):
        g = Genome(new_id, n_inputs = self.n_inputs, n_outputs = self.n_outputs, defined = True)
        g.innovation_db = self.innovation_db
        g.neurons = self.neurons
        g.links = self.links
        g.fitness = 0.0
        g.champion = self.champion
        g.super_champ_offspring = self.super_champ_offspring
        g.orig_fitness = 0.0
        g.solved = self.solved
        g.adjusted_fitness = self.adjusted_fitness
        g.spawns_required = self.spawns_required
        g.species = self.species

        return g


    def exist_link(self, neuron1_id, neuron2_id):
        for link in self.links:
            if link.from_neuron_id == neuron1_id and link.to_neuron_id == neuron2_id:
                return link
        return None

    def exist_neuron(self, neuron_id):
        for neuron in self.neurons:
            if neuron.id == neuron_id:
                return neuron
        return None

    def get_neurons(self):
        return [x for x in self.neurons]

    def add_link(self):
        neuron1 = neuron2 = None
        for _ in range(self.tries_to_find_unlinked_neurons):
            tmp_neuron1 = self.neurons[randint(0, len(self.neurons)-1)]
            tmp_neuron2 = self.neurons[randint(1+self.n_inputs, len(self.neurons)-1)]

            if not self.exist_link(tmp_neuron1.id, tmp_neuron2.id):
                if tmp_neuron1.pos_y >= tmp_neuron2.pos_y:
                    if random() < self.chance_to_add_recurrent_link:
                        neuron1 = tmp_neuron1
                        neuron2 = tmp_neuron2
                        recurrent = True
                        break
                else:
                    neuron1 = tmp_neuron1
                    neuron2 = tmp_neuron2
                    recurrent = False
                    break

        if neuron1 == None or neuron2 == None:
            return None

        innovation = self.innovation_db.get_innovation(InnovationType.LINK, neuron1.id, neuron2.id)

        weight = np.random.normal(0, self.stdev_weight)
        link = LinkGen(neuron1.id, neuron2.id, innovation.link_id, weight = weight, recurrent = recurrent)
        self.links.append(link)
        return link



    def add_neuron(self):
        link = None
        size_threshold = self.n_inputs + self.n_outputs + 5
        if len(self.links) < size_threshold:
            for _ in range(self.tries_to_find_old_link):
                tmp_link = self.links[randint(0, len(self.links)-1-int(sqrt(len(self.links)-1)))]
                if not tmp_link.disabled and not tmp_link.recurrent and self.exist_neuron(tmp_link.from_neuron_id).type != NeuronType.BIAS:
                    link = tmp_link
                    break
            if link == None:
                return
        else:
            while link == None:
                tmp_link = self.links[randint(0, len(self.links)-1)]
                if not tmp_link.disabled and not tmp_link.recurrent and self.exist_neuron(tmp_link.from_neuron_id).type != NeuronType.BIAS:
                    link = tmp_link

        from_neuron = self.exist_neuron(link.from_neuron_id)
        to_neuron = self.exist_neuron(link.to_neuron_id)

        split_x = (from_neuron.pos_x + to_neuron.pos_x) / 2
        split_y = (from_neuron.pos_y + to_neuron.pos_y) / 2
        recurrent = from_neuron.pos_y > to_neuron.pos_y

        innovation = self.innovation_db.get_innovation(InnovationType.NEURON, from_neuron.id, to_neuron.id)

        neuron = self.exist_neuron(innovation.neuron_id)
        if neuron == None:
            neuron = NeuronGen(innovation.neuron_id, NeuronType.HIDDEN, split_x, split_y)
            self.neurons.append(neuron)

            innovation1 = self.innovation_db.get_innovation(InnovationType.LINK, from_neuron.id, neuron.id)
            link1 = LinkGen(from_neuron.id, neuron.id, innovation1.link_id, weight=1.0, recurrent=recurrent)
            self.links.append(link1)

            innovation2 = self.innovation_db.get_innovation(InnovationType.LINK, neuron.id, to_neuron.id)
            link2 = LinkGen(neuron.id, to_neuron.id, innovation2.link_id, weight=link.weight, recurrent=recurrent)
            self.links.append(link2)

            link.disabled = True

            return (neuron, link1, link2)
        else:
            return None

    def mutate_link_weights(self):
        for link in self.links:
            link.weight += np.random.normal(0, self.stdev_mutate_weight)
            link.weight = np.clip(link.weight, self.weight_range[0], self.weight_range[1])

    def mutate(self):
        if random() < self.chance_to_add_neuron and len(self.neurons) < self.max_neurons:
            self.add_neuron()

        if random() < self.chance_to_add_link:
            self.add_link()

        for link in self.links:
            if random() < self.weight_mutation_rate:
                if random() < self.chance_to_reset_weight:
                    link.weight = np.random.normal(0, self.stdev_weight)
                else:
                    link.weight += np.random.normal(0, self.stdev_mutate_weight)
                    link.weight = np.clip(link.weight, self.weight_range[0], self.weight_range[1])


    def save(self):
        str = {'genome_id': self.id, 'fitness': self.fitness - 200, 'neurons': [x.id for x in self.neurons], 'neuron_type': [x.type for x in self.neurons], 'neuron_pos': [(x.pos_x, x.pos_y) for x in self.neurons], 'from_neuron': [x.from_neuron_id for x in self.links], 'to_neuron': [x.to_neuron_id for x in self.links], 'weight': [x.weight for x in self.links], 'disabled': [x.disabled for x in self.links], 'innovation_id': [x.innovation_id for x in self.links]}
        with open('data1.txt', 'a') as f:
            f.write(json.dumps(str))
            f.write('\n')

    def save_info(self):
        str = {'genome_id': self.id, 'fitness': self.fitness - 200}
        with open('data_info1.txt', 'a') as f:
            f.write(json.dumps(str))
            f.write('\n')

    def str(self):
        print("Genome ID:  %s" % self.id)
        print("\nGenes:")
        for i in self.neurons:
            print(i.id, end=' ')
        print("\nConnections:")
        for i in self.links:
            print(str(i.from_neuron_id) + "-->" + str(i.weight) + "-->" +str(i.to_neuron_id) + " : " + str(i.innovation_id))

'''
p = InnovationDB()
a = Genome(0, p)
re = False
a.str()
for _ in range(15):
    a.mutate()
    a.str()
    for link in a.links:
        if link.recurrent == True:
            print("RECURRENT-->")
            re = True
            break
    if re == True:
        break
'''
