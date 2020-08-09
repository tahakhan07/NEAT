import numpy as np
from genotype import *

class NeuronType:
    INPUT, OUTPUT, HIDDEN, BIAS = range(4)

def sigmoid(x, response = 1.0):
    #response = 1/4.924273
    return (1.0 / (1.0 + np.exp(-x/response))) * 2.0 - 1.0

def relu(x):
    return x if x>= 0 else (-x)

class Neuron:
    def __init__(self,neuron_gen):
        self.id = neuron_gen.id
        self.type = neuron_gen.type
        self.input_links = []
        self.output_links = []
        #self.sum_activation = 0
        self.value = 0
        #self.activation_response = neuron_gen.activation_response
        self.pos_x = neuron_gen.pos_x
        self.pos_y = neuron_gen.pos_y

class Link:
    def __init__(self, neurons, link_gen):
        self.input_neuron = next(filter(lambda n: n.id == link_gen.from_neuron_id, neurons))
        self.output_neuron = next(filter(lambda n: n.id == link_gen.to_neuron_id, neurons))
        self.input_neuron.output_links.append(self)
        self.output_neuron.input_links.append(self)
        self.weight = link_gen.weight

class Network:
    def __init__ (self, genome):
        self.genotype = genome

        self.neurons = []
        splits = set()
        for neuron_gen in genome.neurons:
            self.neurons.append(Neuron(neuron_gen))
            splits.add(neuron_gen.pos_y)
        self.depth = len(splits)

        self.links = []
        for link_gen in genome.links:
            if not link_gen.disabled:
                self.links.append(Link(self.neurons, link_gen))


    def feed(self, inputs):
        outputs = []
        i_i = 0
        i_b = 0
        self.neurons.sort(key = lambda x: (x.pos_y, x.pos_x))
        for neuron in self.neurons:
            if neuron.type == NeuronType.INPUT:
                neuron.value = inputs[i_i]
                i_i += 1
            elif neuron.type == NeuronType.BIAS:
                neuron.value = 1
                i_b += 1
            else:
                sum = 0.0
                for link in neuron.input_links:
                    sum = sum + link.weight * link.input_neuron.value
                if neuron.type == NeuronType.HIDDEN:
                    value = relu(sum)
                    neuron.value = value
                elif neuron.type == NeuronType.OUTPUT:
                    value = sigmoid(sum)
                    neuron.value = value
                    outputs.append(value)

        return np.array(outputs)

    def str(self):
        for neuron in self.neurons:
            print(neuron.id)

'''
p = InnovationDB()
#o = Walker()
a = Genome(0, p)
b = Genome(1, p)

for _ in range(100):
    a.mutate()
b = Network(a)
o.evaluate(b)


for _ in range(5):
    a.mutate()
    b.mutate()
a.str()
b.str()

c = Network(a)
c.str()
y = c.feed([1, 1, 1])
c.str()
print(y)
'''
