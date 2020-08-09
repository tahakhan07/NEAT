from math import floor, fmod
from consts import *
from copy import deepcopy
from random import random, randint

from genotype import *

class Species:
    def __init__(self, first_member, species_id):
        self.leader = first_member
        self.leader_old = None
        #self.representative = first_member
        self.members = []
        self.id = species_id
        self.age_of_last_improvement = 0
        self.age = 0
        self.spawns_required = 0
        self.max_fitness = 0.0
        self.max_fitness_ever = 0.0
        self.average_fitness = 0.0
        #self.eliminate = False
        self.obliterate = False
        self.survival_rate = 0.5

        self.young_age_threshold = 10
        self.young_age_fitness_bonus = 1.3
        self.old_age_threshold = 50
        self.old_age_fitness_penalty = 0.7

        self.dropoff_age = Neat.dropoff_age

        self.add_member(first_member)

    def add_member(self, member):
        #member.species_id = self.id
        self.members.append(member)
        member.species = self

    def remove_member(self, member):
        for m in self.members[:]:
            if m.id == member.id:
                #print('found')
                self.members.remove(m)


    def adjust_fitness(self):
        age_debt = (self.age - self.age_of_last_improvement + 1) - self.dropoff_age

        if age_debt == 0:
            age_debt == 1

        for member in self.members:
            member.orig_fitness = member.fitness

            if age_debt >= 1:
                member.fitness = member.fitness * 0.01

            if self.age <= self.young_age_threshold:
                member.fitness = member.fitness * self.young_age_fitness_bonus

            if member.fitness < 0:
                member.fitness = 0.0001

            member.fitness = member.fitness / len(self.members)

        self.members.sort(key = lambda x: x.fitness, reverse = True)
        if self.members[0].orig_fitness > self.max_fitness_ever:
            self.age_of_last_improvement = self.age
            self.max_fitness_ever = self.members[0].orig_fitness

        num_parents = int(floor(self.survival_rate * len(self.members) + 1))

        self.members[0].champion = True
        for i in range(len(self.members) - num_parents):
            self.members[i+num_parents].eliminate = True

    def compute_average_fitness(self):
        total = 0.0
        for m in self.members:
            total += m.fitness

        self.average_fitness = total / len(self.members)
        return self.average_fitness

    def compute_max_fitness(self):
        max = 0.0
        for m in self.members:
            if m.fitness > max:
                max = m.fitness

        self.max_fitness = max
        return max

    def count_offspring(self, skim):
        spawns = 0

        for m in self.members:
            e_o_intpart = int(floor(m.spawns_required))
            e_o_fracpart = fmod(m.spawns_required, 1.0)

            spawns += e_o_intpart
            skim += e_o_fracpart

            if skim>1.0:
                skim_intpart = floor(skim)
                spawns += int(skim_intpart)
                skim -= skim_intpart
        self.spawns_required = spawns
        return skim

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
        score = 6.0*(1.0*n_excess + 1.0*n_disjoint)/max(n_g1, n_g2) + 0.7*weight_difference/n_match
        return score


    def reproduce(self, generation, pop, sorted_species):
        champ_done = False

        if self.spawns_required>0 and len(self.members)==0:
            return False

        poolsize = len(self.members) - 1
        thechamp = self.members[0]

        for i in range(self.spawns_required):

            mut_struct_baby = False
            mate_baby = False

            outside = False

            if thechamp.super_champ_offspring > 0:
                mom = thechamp
                new_genome = mom.make_copy(pop.next_genome_id)
                pop.next_genome_id+=1

                if thechamp.super_champ_offspring == 1:
                    pass

                if thechamp.super_champ_offspring > 1:
                    if random() < 0.8:
                        new_genome.mutate_link_weights()

                    else:
                        new_genome.add_link()
                        mut_struct_baby = True

                baby = new_genome

                thechamp.super_champ_offspring -= 1

            elif (not champ_done) and self.spawns_required > 5:
                mom = thechamp
                new_genome = mom.make_copy(pop.next_genome_id)
                pop.next_genome_id+=1

                baby = new_genome
                champ_done = True

            elif random()<Neat.mutate_only_prob or poolsize == 0:
                mom = self.members[randint(0, poolsize)]
                new_genome = mom.make_copy(pop.next_genome_id)
                pop.next_genome_id+=1

                if random()<Neat.mutate_add_node_prob:
                    new_genome.add_neuron()
                    mut_struct_baby = True

                if random()<Neat.mutate_add_link_prob:
                    new_genome.add_link()
                    mut_struct_baby = True

                if random()<Neat.mutate_link_weights_prob:
                    new_genome.mutate_link_weights

                baby = new_genome

            else:
                mom = self.members[randint(0, poolsize)]

                if random()>Neat.interspecies_mate_rate:
                    dad = self.members[randint(0, poolsize)]
                else:
                    randspecies = self
                    giveup = 0
                    while randspecies==self and giveup<5:
                        randmult = Neat.gaussrand()/4
                        if randmult<0:
                            randmult *= -1.0
                        if randmult > 1.0:
                            randmult = 1.0

                        randspeciesnum = int(floor(randmult * (len(sorted_species)-1) + 0.5))
                        randspecies = sorted_species[randspeciesnum]
                        giveup += 1

                    dad = randspecies.members[0]
                    outside = True

                new_genome = self.crossover(mom, dad, pop.next_genome_id)
                pop.next_genome_id += 1

                mate_baby = True

                if random()>Neat.mate_only_prob or dad.id == mom.id or self.compatibility_score(mom, dad) == 0.0:
                    if random()<Neat.mutate_add_node_prob:
                        new_genome.add_neuron()
                        mut_struct_baby = True

                    if random()<Neat.mutate_add_link_prob:
                        new_genome.add_link()
                        mut_struct_baby = True

                    if random()<Neat.mutate_link_weights_prob:
                        new_genome.mutate_link_weights

                    baby = new_genome

                else:
                    baby = new_genome

            #if mut_struct_baby:
                #print("Mutated")
            #if mate_baby:
                #print("Mated")

            if len(pop.species) == 0:
                newspecies = Species(baby, pop.next_species_id)
                pop.next_species_id += 1
                pop.species.append(newspecies)
            else:
                found = False
                for s in pop.species:
                    if self.compatibility_score(baby, s.members[0]) < Neat.compat_threshold:
                        s.add_member(baby)
                        found = True
                        break

                if found == False:
                    newspecies = Species(baby, pop.next_species_id)
                    pop.next_species_id += 1
                    pop.species.append(newspecies)
