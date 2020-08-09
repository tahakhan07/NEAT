from innovation import *
from genotype import *
from species import *
from phenotype import *
from consts import *

class GeneticAlgorithm:
    def __init__(self, task):
        self.genomes = []
        self.species = []
        self.best_ever = None
        self.innovation_db = InnovationDB()
        self.task = task
        self.generation = 0
        self.next_genome_id = 0
        self.next_species_id = 0

        self.highest_fitness = 0.0
        self.highest_last_changed = 0

        while len(self.genomes) < Neat.pop_size:
            genome = Genome(self.next_genome_id, self.innovation_db, None, None, self.task.n_inputs, self.task.n_outputs)
            self.genomes.append(genome)
            #genome.str()
            self.next_genome_id += 1

        self.speciate()

        for g in self.genomes:
            network = Network(g)
            g.fitness = self.task.evaluate(network)

        self.genomes.sort(key = lambda x: x.fitness, reverse = True)
        self.generation += 1

        for s in self.species:
            for g in s.members:
                print(s.id, " : ", g.id)


    def speciate(self):
        for g in self.genomes:
            if len(self.species)==0:
                newspecies = Species(g, self.next_species_id)
                self.next_species_id += 1
                self.species.append(newspecies)

            else:
                found = False
                for s in self.species:
                    if Species.compatibility_score(g, s.members[0]) < Neat.compat_threshold:
                        s.add_member(g)
                        found = True
                        break

                if found == False:
                    newspecies = Species(g, self.next_species_id)
                    self.next_species_id += 1
                    self.species.append(newspecies)

    def epoch(self):
        total_organisms = Neat.pop_size
        total = 0.0
        #sorted_species = sorted(self.species, key = lambda x: x.members[0].orig_fitness, reverse = True)

        #if generation%30 == 0:
        #    for i in range(len(sorted_species)-1, -1, -1):
        #        if sorted_species[i].age >= 20
        #            sorted_species[i].obliterate=True
        #            break

        for s in self.species:
            s.adjust_fitness()

        for g in self.genomes:
            total += g.fitness

        overall_average = total / total_organisms


        for g in self.genomes:
            g.spawns_required = g.fitness / overall_average

        skim = 0.0
        total_expected = 0
        for s in self.species:
            skim = s.count_offspring(skim)
            total_expected += s.spawns_required

        if total_expected < total_organisms:
            print('ERROR')
            max_expected = 0
            final_expected = 0
            for s in self.species:
                if s.spawns_required >= max_expected:
                    max_expected = s.spawns_required
                    best_species = s
                final_expected += s.spawns_required

            best_species.spawns_required += 1
            final_expected += 1

            if final_expected < total_organisms:
                print("MASSIVE ERROR")
                for s in self.species:
                    s.spawns_required = 0
                best_species.spawns_required = total_organisms

        sorted_species = sorted(self.species, key = lambda x: x.members[0].orig_fitness, reverse = True)
        best_species_num = sorted_species[0].id

        if sorted_species[0].members[0].orig_fitness > self.highest_fitness:
            self.highest_fitness = sorted_species[0].members[0].orig_fitness
            self.highest_last_changed = 0
        else:
            self.highest_last_changed += 1
        '''
        if self.highest_last_changed >= Neat.dropoff_age+5:
            print("RESET")
            self.highest_last_changed = 0
            half_pop = int(Neat.pop_size/2)

            s = sorted_species[0]
            s.members[0].super_champ_offspring = half_pop
            s.spawns_required = half_pop
            s.age_of_last_improvement = s.age

            if len(sorted_species)>1:
                s = sorted_species[1]
                s.members[0].super_champ_offspring = Neat.pop_size - half_pop
                s.spawns_required = Neat.pop_size - half_pop
                s.age_of_last_improvement = s.age

                for s in sorted_species[2:]:
                    s.spawns_required = 0

            else:
                s = sorted_species[0]
                s.members[0].super_champ_offspring += Neat.pop_size - half_pop
                s.spawns_required += Neat.pop_size - half_pop
        '''
        '''
        print('Before eliminate: ')
        for s in self.species:
            for g in s.members:
                print(s.id, " : ", g.id)
        '''
        for g in self.genomes[:]:
            if g.eliminate:
                g.species.remove_member(g)
                self.genomes.remove(g)


        #self.genomes = list(filter(lambda x: not x.eliminate, self.genomes))
        #for s in self.species:
        #    s.members = list(filter(lambda x: not x.eliminate, s.members))
        '''
        print('\nAfter eliminate:')
        for s in self.species:
            for g in s.members:
                print(s.id, " : ", g.id)
        print('\nGenomes:')
        for g in self.genomes:
            print(g.id)
        '''
        last_id = self.species[0].id
        for s in self.species:
            s.reproduce(self.generation, self, sorted_species)
            last_id = s.id
        '''
        print("After reproduce:")
        for s in self.species:
            for g in s.members:
                print(s.id, " : ", g.id)
        print('\nGenomes:')
        for g in self.genomes:
            print(g.id)
        '''
        for g in self.genomes[:]:
            g.species.remove_member(g)
            self.genomes.remove(g)

        for s in self.species[:]:
            if len(s.members) == 0:
                self.species.remove(s)

            else:
                s.age += 1

                for m in s.members:
                    self.genomes.append(m)
        '''
        print('\nAfter more eliminate:')
        for s in self.species:
            for g in s.members:
                print(s.id, " : ", g.id)
        print('\nGenomes:')
        for g in self.genomes:
            print(g.id)
        '''
        for g in self.genomes:
            network = Network(g)
            g.fitness = self.task.evaluate(network)

        self.genomes.sort(key = lambda x: x.fitness, reverse = True)
        '''
        print('\nAfter sorting:')
        print('\nGenomes:')
        for g in self.genomes:
            print(g.id)
        '''

        for s in self.species:
            for g in s.members:
                print(s.id, " : ", g.id, ' : ', len(g.neurons), ' : ', g.fitness-200)

        self.generation += 1
