import gym
from neat import *

class CartPole:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.n_inputs = 4
        self.n_outputs = 1

        self.env.reset()

    def evaluate(self, network):
        state = self.env.reset()
        outputs = []
        fitness = 0.0
        done = False
        while not done:
            outputs = network.feed(state)
            if outputs[0]<=0:
                outputs = 0
            else:
                outputs = 1
            state, reward, done, _  = self.env.step(outputs)
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
            if outputs[0]<=0:
                outputs = 0
            else:
                outputs = 1
            state, reward, done, _  = self.env.step(outputs)
            fitness += reward
            self.env.render()
        print(fitness)
        self.env.close()

cartpole = CartPole()
ga = GeneticAlgorithm(cartpole)

for i in range(1000):
    print(i)
    ga.epoch()
    print("Current BEST: %s --> %s" % (ga.genomes[0].id, ga.genomes[0].fitness))
    print("All time BEST: %s --> %s" % (ga.best_ever.id, ga.best_ever.fitness))
    if i%25 == 0:
        print([x.id for x in ga.species])
        cartpole.show(Network(ga.genomes[0]))
