import gym
from neat2 import *
from walker_run import *

class Normalizer():
    # Normalizes the inputs
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

class Walker:
    def __init__(self):
        self.env = gym.make("BipedalWalker-v3")
        self.n_inputs = 24 #24
        self.n_outputs = 4
        self.normalizer = Normalizer(self.n_inputs)

        self.env.reset()

    def evaluate(self, network):
        n = 1
        sum = 0.0
        for _ in range(n):
            state = self.env.reset()
            outputs = []
            fitness = 0.0
            done = False
            while not done:
                #self.normalizer.observe(state)
                #state = self.normalizer.normalize(state)
                outputs = network.feed(state)
                state, reward, done, _  = self.env.step(outputs)
                if reward == -100:
                    reward = -25
                    #print("TOPPLED")
                fitness += reward
            self.env.close()
            sum += fitness
        avg = sum / n
        return (avg+200)

    def show(self, network):
        state = self.env.reset()
        outputs = []
        fitness = 0.0
        done = False
        while not done:
            #self.normalizer.observe(state)
            #state = self.normalizer.normalize(state)
            outputs = network.feed(state)
            state, reward, done, _  = self.env.step(outputs)
            fitness += reward
            self.env.render()
        print(fitness)
        self.env.close()


walker = Walker()

ga = GeneticAlgorithm(walker)
#s = ga.species[0]
#for g in s.members:
#    print(g.id, ' : ', g.fitness)
#print('\n')
#s.adjust_fitness()
#for g in s.members:
#    print(g.id, ' : ', g.fitness, ' : ', g.eliminate)

for i in range(100000):
    print("\n---------------------\n\n%s" % (i+1))
    ga.epoch()
    print("Species: ", end='')
    for s in ga.species:
        print(s.id, end=', ')
    print("Current BEST: %s --> %s" % (ga.genomes[0].id, ga.genomes[0].fitness-200))
    print("Current SECOND BEST: %s --> %s" % (ga.genomes[1].id, ga.genomes[1].fitness-200))
    print("Current THIRD BEST: %s --> %s" % (ga.genomes[2].id, ga.genomes[2].fitness-200))
    print("Current FOURTH BEST: %s --> %s" % (ga.genomes[3].id, ga.genomes[3].fitness-200))
    #print("\nAll time BEST: %s --> %s" % (ga.best_ever.id, ga.best_ever.fitness))
    #if i>0:
    #    print("\nPREVIOUS 1: %s --> %s" % (ga.best_ever_old.id, ga.best_ever_old.fitness))
    #if i>1:
    #    print("PREVIOUS 2: %s --> %s" % (ga.best_old_2.id, ga.best_old_2.fitness))
    #if i>2:
    #    print("PREVIOUS 3: %s --> %s" % (ga.best_old_3.id, ga.best_old_3.fitness))
    #if i>3:
    #    print("PREVIOUS 4: %s --> %s" % (ga.best_old_4.id, ga.best_old_4.fitness))
    #ga.genomes[0].str()
    #for _ in range(0):
    #    print(walker.evaluate(Network(ga.genomes[0])))
    #print("\n")
    #walker.show(Network(ga.genomes[0]))
    #print("\n")
    #print(walker.evaluate(Network(ga.genomes[0])))
    if (i+1)%10 == 0:
        ga.genomes[0].save()
        ga.genomes[0].save_info()
        #walker.show(Network(ga.genomes[0]))
    #    print([x.id for x in ga.species])
    #    walker.show(Network(ga.genomes[0]))
    #if ga.best_ever_old != None and ga.genomes[0].fitness - ga.best_ever_old.fitness > 50:
    #    walker.show(Network(ga.genomes[0]))
    #if ga.genomes[0].fitness>100:
    #    for _ in range(3):
    #        print(walker.evaluate(Network(ga.genomes[0])))

'''
def load(x):
    g = load_genotype(x)
    for _ in range(1):
        gen = walker.evaluate(Network(g))
        print(gen-200)
        if gen>25:
            return True
        return False

for i in range(10000):
    print("\n")
    #x = load(520177)
    #y = load(20741)
    #z = load(286531)
    #walker.show(Network(load_genotype(486195)))
    x = load_genotype(244128)
    walker.show(Network(x))
    #gen = walker.evaluate(Network(x))
    #print(gen-200)

    y = x.make_copy(0)
    walker.show(Network(y))
'''


#env = gym.make("BipedalWalker-v3")
#print(env.reset())
