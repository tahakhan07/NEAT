import gym
from neat import *

class MountainCar:
    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        self.n_inputs = 2
        self.n_outputs = 3

        self.env.reset()

    def evaluate(self, network):
        state = self.env.reset()
        outputs = []
        fitness = 0.0
        done = False
        while not done:
            outputs = network.feed(state)
            output = np.argmax(outputs)
            state, reward, done, _  = self.env.step(output)
            fitness += reward
        self.env.close()
        return fitness

    def show(self, network):
        state = self.env.reset()
        outputs = []
        fitness = 0.0
        done = False
        while not done:
            outputs = network.feed(state)
            output = np.argmax(outputs)
            state, reward, done, _  = self.env.step(output)
            fitness += reward
            self.env.render()
        print(fitness)
        self.env.close()

mountaincar = MountainCar()
ga = GeneticAlgorithm(mountaincar)

for i in range(1000):
    print(i+1)
    ga.epoch()
    print("Current BEST: %s --> %s" % (ga.genomes[0].id, ga.genomes[0].fitness))
    print("All time BEST: %s --> %s" % (ga.best_ever.id, ga.best_ever.fitness))
    if (i+1)%25 == 0:
        print([x.id for x in ga.species])
        mountaincar.show(Network(ga.genomes[0]))
