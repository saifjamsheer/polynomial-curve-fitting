from random import randint, random
from operator import add
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import math

def individual(length, min, max):
    """
    Create a member of the population (i.e. an individual).

    length: number of values per individual
    min: min possible value for an individual
    max: max possible value for an individual    

    """
    return [ randint(min,max) for x in range(length) ]

def population(count, length, min, max):
    """
    Create a number of individuals (i.e. a population).

    count: number of individuals in the population
    length: number of values per individual
    min: min possible value in an individual's list of values
    max: max possible value in an individual's list of values

    """
    return [ individual(length, min, max) for x in range(count) ]

def fitness(individual, target):
    """
    Determine the fitness of an individual (i.e. the sum of the 
    difference in values between the individual and target lists).
    A lower value is better.

    individual: the individual to evaluate
    target: the list of values that individuals are aiming for
   
    """
    ind = np.array(individual)
    tar = np.array(target)
    sub = abs(np.subtract(tar,ind))

    return np.sum(sub)

def grade(pop, target):
    """
    Calculate the average fitness for a population.

    pop: the population of individuals
    target: the list of values that individuals are aiming for

    """
    summed = reduce(add, (fitness(x,target) for x in pop), 0)
    return summed / (len(pop) * 1.0)

def select(pop, target, method, retain, random_select):
    """
    Selection process to determine the most suitable individuals
    to undergo mutation and crossover

    pop: the population of individuals
    target: the list of values that individuals are aiming for
    method: selection method ('rank' or 'roulette wheel')
    retain: the percentage of best-performing individuals to retain
    random_select: determines if an individual should be randomly
    added to the retained list

    """
    # calculate the fitness of all individuals in the population
    graded = [ (fitness(x, target), x) for x in pop]

    if method == 'rank':

        # rank the individuals from best-performing to worst-performing
        graded = [ x[1] for x in sorted(graded) ]

        # retain the best performing individuals and label them as parents
        retain_length = int(len(graded)*retain)
        parents = graded[:retain_length]

        # randomly add other individuals to promote genetic diversity
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)

    elif method == 'roulette wheel':

        # convert the fitness values to ensure that higher values imply
        # that the individual is fitter
        fitnesses = [ 1/(0.1+x[0]) for x in graded ]

        # calculate sum of all fitnessees
        total = sum(fitnesses)

        # divide fitness over sum of fitnesses for each individual
        r_fitness = [ f/total for f in fitnesses ]

        # calculate the cumulative sum to generate a probability 
        # distribution
        probs = [sum(r_fitness[:i+1]) for i in range(len(r_fitness))]

        # calculate the number of individuals that should be retained
        retain_length = int(len(fitnesses)*retain)
        
        # initialize a list of parents
        parents = []

        # add individuals to the list of parents through the roulette
        # wheel method
        for i in range(retain_length):
            r = random()
            for (i, individual) in enumerate(pop):
                if r <= probs[i]:
                    parents.append(individual)
                    break

    return parents

def crossover(pop, parents, type, n):
    """
    Combining the information from two parents through one of two
    techniques to generate offspring.

    pop: the population of individuals
    parents: the parents that will be used for crossover
    type: crossover technique ('split' or 'uniform)
    n: number of elite values that were passed on

    """
    parents_length = len(parents)
    desired_length = len(pop) - parents_length - n # desired length of children
    children = [] # initializing list of children

    while len(children) < desired_length:
        # selecting the positions of the male and female
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            # selecting a male and female from the list of parents
            male = parents[male]
            female = parents[female]
            
            if type == 'split':
                # randomly select cutoff point for one-point crossover
                cutoff = randint(1, len(male)-1)
                # combine information from parents to create child
                child = male[:cutoff] + female[cutoff:]
                children.append(child)

            elif type == 'uniform':
                # create the crossover mask
                mask = np.random.choice([0,1], size=(20,))
                # combine information from parents to create child
                child = male*mask + female*(1-mask)
                children.append(child.tolist())

    parents.extend(children)
    return parents

def evolve(pop, target, selection_type, crossover_type, elitism=False, retain=0.2, random_select=0.05, mutate=0.01):
    """
    Evolve the individuals of a population towards a better solution
    through mutation, crossover, and selection. 

    pop: the population of individuals
    target: the target value that individuals are aiming for
    retain: the percentage of best-performing individuals to retain
    random_select: determines if an individual should be randomly
    added to the retained list
    mutate: determines if an individual should be mutated

    """
    # select parents
    parents = select(pop, target, selection_type, retain, random_select)

    # initialize number of elite members to pass 
    n = 0

    # only use elitism if selection is done through ranking
    if elitism and selection_type == 'rank':
        # calculate number of elite members to pass onto the next generation
        n = int(0.05*len(parents))
        # select elite members
        elite = parents[:n]
        parents = parents[n:]
    
    # mutate some individuals through random resetting
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)
            individual[pos_to_mutate] = randint(min(target)-1, max(target)+1)

    if elitism and selection_type == 'rank':
        parents = elite + parents

    # crossover to create children
    parents = crossover(pop, parents, crossover_type, n)
    
    return parents

def main():
    
    poly = [25, 18, 31, -14, 7, -19] # polynomial coefficients
    values = np.linspace(-1,1,20) # values to generate points on the curve
    target = [int(x) for x in np.polyval(poly,values)] # points on the polynomial
    p_count = 100 # number of individuals in a population
    i_length = len(target) # length of an individual
    i_min = -100 # minimum possible value in an individual's list of values
    i_max = 100 # maximum possible value in an individual's list of values
    p_selection = 'rank' # selection method ('rank', 'roulette wheel')
    p_crossover = 'uniform' # crossover technique ('split', 'uniform')
    elitism = False # boolean to determine if elitism should be implemented
    r_percentage = 0.75 # retain percentage
    r_select = 0.05 # random select probability for genetic diversity
    m_probability = 0.1 # mutation probability

    # create an initial population of individuals
    p = population(p_count, i_length, i_min, i_max)

    # create an initial list to store population fitness values
    fitness_history = [grade(p,target)]

    # initialize number of generations to reach solution
    generations = 0

    # evolve the initial population until an optimal solution is reached
    while fitness_history[-1] >= 1.0 and fitness(p[0], target) != 0:
        p = evolve(p, target, p_selection, p_crossover, elitism, retain=r_percentage, random_select=r_select, mutate=m_probability)
        fitness_history.append(grade(p,target))
        generations += 1

    print('Generations: {iter}'.format(iter=generations))

    plot_type = 1 # integer that determines which figure should be plotted

    if plot_type == 1:
        # plot fitness history to display when solution is reached
        plt.plot(fitness_history, c='crimson')
        plt.ylabel('Average fitness of population')
        plt.xlabel('Number of generations')
    else:
        points = p[0]
        # plot approximation of best performing individual to the polynomial
        plt.plot(values, target, c='seagreen',marker='.', linewidth=2)
        plt.plot(values, points, c='blue', marker='.', linewidth=2)
        plt.xticks([])
        plt.yticks([])
        plt.legend(['Target', 'Approximation'], loc='lower right')

    plt.show()

if __name__ == '__main__':
	main()