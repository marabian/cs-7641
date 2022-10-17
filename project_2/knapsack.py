# source: https://github.com/pushkar/ABAGAIL/blob/master/jython/knapsack.py 
import sys
import os
import time
import csv

sys.path.append("./ABAGAIL.jar")

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import shared.ConvergenceTrainer as ConvergenceTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction

from array import array


# Random number generator */
random = Random()

class Knapsack:
    # The number of items
    NUM_ITEMS = 40
    # The number of copies each
    COPIES_EACH = 4
    # The maximum weight for a single element
    MAX_WEIGHT = 50
    # The maximum volume for a single element
    MAX_VOLUME = 50
    # The volume of the knapsack 
    KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

    def __init__(self, algo_name):
        random.setSeed(56)

        self.algo_name = algo_name

        # create copies
        fill = [self.COPIES_EACH] * self.NUM_ITEMS
        copies = array('i', fill)

        # create weights and volumes
        fill = [0] * self.NUM_ITEMS
        weights = array('d', fill)
        volumes = array('d', fill)
        for i in range(0, self.NUM_ITEMS):
            weights[i] = random.nextDouble() * self.MAX_WEIGHT
            volumes[i] = random.nextDouble() * self.MAX_VOLUME


        # create range
        fill = [self.COPIES_EACH + 1] * self.NUM_ITEMS
        ranges = array('i', fill)

        self.ef = KnapsackEvaluationFunction(weights, volumes, self.KNAPSACK_VOLUME, copies)
        self.problem_name = "knapsack"

        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = UniformCrossOver()
        df = DiscreteDependencyTree(.1, ranges)
        hcp = GenericHillClimbingProblem(self.ef, odd, nf)
        gap = GenericGeneticAlgorithmProblem(self.ef, odd, mf, cf)
        pop = GenericProbabilisticOptimizationProblem(self.ef, odd, df)

        if algo_name == "RHC":
            rhc = RandomizedHillClimbing(hcp)
            self.func = rhc
        if algo_name == "SA":
            sa = SimulatedAnnealing(1e13, .90, hcp)
            self.func = sa
        if algo_name == "GA":
            gap = GenericGeneticAlgorithmProblem(self.ef, odd, mf, cf)
            ga = StandardGeneticAlgorithm(200, 100, 10, gap)
            self.func = ga
        if algo_name == "MIMIC":
            pop = GenericProbabilisticOptimizationProblem(self.ef, odd, df)
            mimic = MIMIC(200, 20, pop)
            self.func = mimic


    def run_problem(self, iters=5000, runs=10):
        for p in range(1, runs+1):
            fit = FixedIterationTrainer(self.func, 10)
            FILE_NAME="{}_{}_{}.csv".format(self.algo_name, self.problem_name, str(p))
            OUTPUT_FILE = os.path.join("data/csv", FILE_NAME)
            with open(OUTPUT_FILE, "wb") as data:
                writer= csv.writer(data, delimiter=',')
                writer.writerow(["iters","fevals","fitness"])
                for i in range(0, iters, 10):
                    fit.train()
                    writer.writerow([i, self.ef.getFunctionEvaluations()-i, self.ef.value(self.func.getOptimal())])
            
            print self.algo_name + " runs #" + str(p)
            print self.algo_name + ": " + str(self.ef.value(self.func.getOptimal()))
            print "Function Evaluations: " + str(self.ef.getFunctionEvaluations()-iters)
            print "Iters: " + str(iters)
            print "####"


# run algorithms
rhc = Knapsack("RHC")
rhc.run_problem()

sa = Knapsack("SA")
sa.run_problem()

ga = Knapsack("GA")
ga.run_problem()

mimic = Knapsack("MIMIC")
mimic.run_problem()