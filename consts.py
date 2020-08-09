from random import random
from math import sqrt, log

class Neat:
    pop_size = 200
    dropoff_age = 20
    mutate_only_prob = 0.3
    mutate_add_node_prob = 0.01
    mutate_add_link_prob = 0.1
    mutate_link_weights_prob = 0.3
    mate_only_prob = 0.3
    compat_threshold = 1.5
    interspecies_mate_rate = 0.05


    iset = 0
    gset = 0.0

    @staticmethod
    def gaussrand():
        if Neat.iset==0:
            while True:
                v1 = 2.0*random() - 1.0
                v2 = 2.0*random() - 1.0
                rsq = v1*v1 + v2*v2
                if not (rsq>=1.0 or rsq==0.0):
                    break
            fac = sqrt(-2.0 * log(rsq)/rsq)
            Neat.gset = v1*fac
            Neat.iset = 1
            return v2*fac
        else:
            Neat.iset = 0
            return Neat.gset
