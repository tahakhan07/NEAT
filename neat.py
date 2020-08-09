import numpy as np
from random import randint, choice
from copy import deepcopy

from innovation import *
from genotype import *
from species import *
from phenotype import *
from consts import *

class GeneticAlgorithm:
    def __init__(self, task):
        self.genomes = []
        self.species = []
        self.bests = []
        self.best_ever = None
        self.best_ever_old = None
        self.best_old_2 = None
        self.best_old_3 = None
        self.best_old_4 = None
        self.innovation_db = InnovationDB()
        self.task = task
        self.generation = 0
        self.next_genome_id = 0
        self.next_species_id = 0
        self.compatibility_threshold = 3.0

        self.population_size = Neat.pop_size
        self.target_species = 30
        self.number_generations_allowed_to_not_improve = 5 #20
        self.crossover_rate = 0.7
        self.survival_rate = 0.5
        self.trys_in_tournament_selection = 3
        self.elitism = True
        self.min_elitism_size = 1 #5
        self.young_age_threshold = 10
        self.young_age_fitness_bonus = 1.3
        self.old_age_threshold = 50
        self.old_age_fitness_penalty = 0.7

        self.substrat = 1

    def epoch(self):

        for i in range(len(self.species)):
            print("SPECIES: %s   LEADER: %s  AGE: %s  GNI: %s" % (self.species[i].id, self.species[i].leader.id, self.species[i].age, self.species[i].generations_not_improved))

        total_average = sum(s.average_fitness for s in self.species)
        #print("\n\nNEW EPOCH\n\ntotal_average = %s" % total_average)
        for s in self.species:
            s.spawns_required = int(round(self.population_size * s.average_fitness / total_average))
            #print("SPECIES ID: %s  SPAWNS: %s" % (s.id, s.spawns_required))

        print("\n")
        for i in range(len(self.species)):
            print("SPECIES: %s   SPAWNS: %s" % (self.species[i].id, self.species[i].spawns_required))

        species = []
        for s in self.species:
            if s.generations_not_improved < self.number_generations_allowed_to_not_improve and s.spawns_required > 0:
                species.append(s)

        self.species[:] = species

        for s in self.species:
            k = max(1, int(round(len(s.members) * self.survival_rate)))
            pool = s.members[:k]
            #print("POOL: %s" % [x.id for x in pool])
            s.members[:] = []

            if self.elitism and len(pool) >= self.min_elitism_size:
                s.add_member(s.leader)

            while len(s.members) < s.spawns_required:
                n = min(len(pool), self.trys_in_tournament_selection)
                g1 = self.tournament_selection(pool, n)
                g2 = self.tournament_selection(pool, n)
                child = self.crossover(g1, g2, self.next_genome_id)
                #print("NEW CHILD: %s   crossed between   %s  and  %s" % (child.id, g1.id, g2.id))
                #print("G1:")
                #g1.str()
                #print("G2:")
                #g2.str()
                #print("CHILD:")
                #child.str()
                self.next_genome_id += 1
                child.mutate()
                s.add_member(child)
                #print([x.id for x in s.members])

        self.genomes[:] = []
        for s in self.species:
            self.genomes.extend(s.members)
            s.members[:] = []
            s.age += 1

        #print("\nCreating new Genomes --")
        while len(self.genomes) < self.population_size:
            genome = Genome(self.next_genome_id, self.innovation_db, None, None, self.task.n_inputs, self.task.n_outputs)
            self.genomes.append(genome)
            #genome.str()
            self.next_genome_id += 1

        x = 0
        for g in self.genomes:
            if self.best_ever!=None and g.id == self.best_ever.id:
                print("\nPRESENT\n")
                x=1
                break
        if x==0:
            print("\nABSENT\n")

        #print("\nevaluating:")
        for g in self.genomes:
            network = Network(g)
            fitness = self.task.evaluate(network)
            g.fitness = fitness
            #print("ID: %s   FITNESS: %s" % (g.id, g.fitness))
            #g.solved = int(solved)

        self.genomes.sort(key = lambda x: x.fitness, reverse = True)

        self.best_old_4 = self.best_old_3
        self.best_old_3 = self.best_old_2
        self.best_old_2 = self.best_ever_old
        self.best_ever_old = self.best_ever

        if self.best_ever == None or self.best_ever.fitness < self.genomes[0].fitness:
            self.best_ever = self.genomes[0]
        print("new genomes:  genome: %s   fitness: %s" % (self.genomes[0].id, self.genomes[0].fitness))
        '''
        print("\nAFTER SORTING:")
        for g in self.genomes:
            print("ID: %s   FITNESS: %s" % (g.id, g.fitness))
        print("BEST: %s" % (self.best_ever.id))
        '''

        #print("\nFORMING SPECIES: ")
        for g in self.genomes:
            added = False
            for s in self.species:
                compatibility = self.compatibility_score(g, s.leader)
                if compatibility <= self.compatibility_threshold:
                    s.add_member(g)
                    added = True
                    break
            if not added:
                s = Species(g, self.next_species_id)
                self.next_species_id += 1
                self.species.append(s)

        #for s in self.species:
        #    for g in s.members:
        #        print("SPECIES: %s   MEMBER: %s" % (s.id, g.id))



        self.species[:] = filter(lambda s: len(s.members) > 0, self.species)

        if len(self.species) < self.target_species:
            self.compatibility_threshold *= 0.95
        elif len(self.species) > self.target_species:
            self.compatibility_threshold *= 1.05

        for s in self.species:
            s.members.sort(key = lambda x: x.fitness, reverse=True)
            s.leader_old = s.leader
            s.leader = s.members[0]
            #print("SPECIES: %s   LEADER: %s" % (s.id, s.leader.id))

            if s.leader.fitness > s.max_fitness:
                s.generations_not_improved = 0
            else:
                s.generations_not_improved += 1
            s.max_fitness = ((s.max_fitness*(s.age)) + s.leader.fitness)/(s.age+1)

            sum_fitness = 0.0
            for m in s.members:
                fitness = m.fitness
                sum_fitness += fitness 

                if s.age < self.young_age_threshold:
                    fitness *= self.young_age_fitness_bonus
                elif s.age > self.old_age_threshold:
                    fitness *= self.old_age_fitness_penalty

                m.adjusted_fitness = fitness/len(s.members)

            s.average_fitness = sum_fitness/len(s.members)

    @staticmethod
    def tournament_selection(genomes, number_to_compare):
        champion = None
        for _ in range(number_to_compare):
            g = genomes[randint(0, len(genomes) - 1)]
            if champion == None or g.fitness > champion.fitness:
                champion = g
        return champion

    @staticmethod
    def crossover(mum, dad, baby_id=None):
        n_mum = len(mum.links)
        n_dad = len(dad.links)

        if mum.fitness == dad.fitness:
            if n_mum == n_dad:
                better = (mum,dad)[randint(0,1)]
            elif n_mum < n_dad:
                better = mum
            else:
                better = dad
        elif mum.fitness > dad.fitness:
            better = mum
        else:
            better = dad

        baby_neurons = []
        baby_links = []

        i_mum = i_dad = 0
        neuron_ids = set()
        while i_mum < n_mum or i_dad < n_dad:
            mum_gene = mum.links[i_mum] if i_mum < n_mum else None
            dad_gene = dad.links[i_dad] if i_dad < n_dad else None
            selected_gene = None
            if mum_gene and dad_gene:
                if mum_gene.innovation_id == dad_gene.innovation_id:
                    idx = randint(0, 1)
                    selected_gene = (mum_gene, dad_gene)[idx]
                    selected_genome = (mum, dad)[idx]
                    i_mum += 1
                    i_dad += 1
                elif dad_gene.innovation_id < mum_gene.innovation_id:
                    if better == dad:
                        selected_gene = dad.links[i_dad]
                        selected_genome = dad
                    i_dad += 1
                elif mum_gene.innovation_id < dad_gene.innovation_id:
                    if better == mum:
                        selected_gene = mum.links[i_mum]
                        selected_genome = mum
                    i_mum += 1
            elif mum_gene and dad_gene==None:
                if better == mum:
                    selected_gene = mum.links[i_mum]
                    selected_genome = mum
                i_mum += 1
            elif mum_gene==None and dad_gene:
                if better == dad:
                    selected_gene = dad.links[i_dad]
                    selected_genome = dad
                i_dad += 1

            if selected_gene and len(baby_links) and baby_links[len(baby_links)-1].innovation_id == selected_gene.innovation_id:
                selected_gene == None

            if selected_gene != None:
                baby_links.append(deepcopy(selected_gene))

                if not selected_gene.from_neuron_id in neuron_ids:
                    neuron = selected_genome.exist_neuron(selected_gene.from_neuron_id)
                    if neuron != None:
                        baby_neurons.append(deepcopy(neuron))
                        neuron_ids.add(selected_gene.from_neuron_id)
                if not selected_gene.to_neuron_id in neuron_ids:
                    neuron = selected_genome.exist_neuron(selected_gene.to_neuron_id)
                    if neuron != None:
                        baby_neurons.append(deepcopy(neuron))
                        neuron_ids.add(selected_gene.to_neuron_id)

        for neuron in mum.get_neurons():
            if not neuron.id in neuron_ids:
                baby_neurons.append(deepcopy(neuron))
                neuron_ids.add(neuron.id)

        if all([l.disabled for l in baby_links]):
            choice(baby_links).disabled = False

        innovation_db = mum.innovation_db
        n_inputs = mum.n_inputs
        n_outputs = mum.n_outputs
        baby = Genome(baby_id, innovation_db, baby_neurons, baby_links, n_inputs, n_outputs)

        return baby

    @staticmethod
    def compatibility_score(genome1, genome2):
        n_match = n_disjoint = n_excess = 0
        weight_difference = 0

        n_g1 = len(genome1.links)
        n_g2 = len(genome2.links)
        i_g1 = i_g2 = 0

        while i_g1 < n_g1 or i_g2 < n_g2:
            if i_g1 == n_g1:
                n_excess += 1
                i_g2 += 1
                continue
            if i_g2 == n_g2:
                n_excess += 1
                i_g1 += 1
                continue

            link1 = genome1.links[i_g1]
            link2 = genome2.links[i_g2]

            if link1.innovation_id == link2.innovation_id:
                n_match += 1
                i_g1 += 1
                i_g2 += 1
                weight_difference = weight_difference + abs(link1.weight - link2.weight)
                continue

            if link1.innovation_id < link2.innovation_id:
                n_disjoint += 1
                i_g1 += 1
                continue

            if link1.innovation_id > link2.innovation_id:
                n_disjoint += 1
                i_g2 += 1
                continue

        #n_match += 1
        score = (1.0*n_excess + 1.0*n_disjoint)/max(n_g1, n_g2) + 0.4*weight_difference/n_match
        return score
'''
p = InnovationDB()
a = Genome(0, p)
b = Genome(1, p)
for _ in range(3):
    a.mutate()
    b.mutate()
a.str()
b.str()
task = []
ga = GeneticAlgorithm(task)
c = ga.crossover(a, b, 2)
c.str()
'''
