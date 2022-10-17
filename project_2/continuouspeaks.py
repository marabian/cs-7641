# source: https://github.com/pushkar/ABAGAIL/blob/master/jython/continuouspeaks.py
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
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction

from array import array

random = Random()

class ContinuousPeaks:
    N=60
    T=N/10

    def __init__(self, algo_name):
        random.setSeed(56)
        self.algo_name = algo_name

        fill = [2] * self.N
        ranges = array('i', fill)

        self.ef = ContinuousPeaksEvaluationFunction(self.T)
        self.problem_name = "continuouspeaks"

        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        df = DiscreteDependencyTree(.1, ranges)
        hcp = GenericHillClimbingProblem(self.ef, odd, nf)
        gap = GenericGeneticAlgorithmProblem(self.ef, odd, mf, cf)
        pop = GenericProbabilisticOptimizationProblem(self.ef, odd, df)

        if self.algo_name == "RHC":
            rhc = RandomizedHillClimbing(hcp)
            self.func = rhc
        elif self.algo_name == "SA":
            sa = SimulatedAnnealing(1e13, .90, hcp)
            self.func = sa
        elif self.algo_name == "GA":
            gap = GenericGeneticAlgorithmProblem(self.ef, odd, mf, cf)
            ga = StandardGeneticAlgorithm(200, 100, 10, gap)
            self.func = ga
        elif self.algo_name == "MIMIC":
            pop = GenericProbabilisticOptimizationProblem(self.ef, odd, df)
            mimic = MIMIC(200, 20, pop)
            self.func = mimic

    def run_problem(self, iters=14000, runs=10):



        for p in range(1, runs+1):
            fit = FixedIterationTrainer(self.func, 10)
            times = [0]

            FILE_NAME="{}_{}_{}.csv".format(self.algo_name, self.problem_name, str(p))
            OUTPUT_FILE = os.path.join("data/csv", FILE_NAME)
            with open(OUTPUT_FILE, "wb") as data:
                writer= csv.writer(data, delimiter=',')
                writer.writerow(["iters","fevals","fitness","times"])
                for i in range(0, iters, 10):
                    start = time.clock()
                    fit.train()
                    dur = time.clock() - start
                    times.append(times[-1] + dur)
                    #print str(i) + ", " + str(ef.getFunctionEvaluations()) + ", " + str(ef.value(func.getOptimal()))
                    writer.writerow([i, self.ef.getFunctionEvaluations()-i, self.ef.value(self.func.getOptimal())])
            
            print self.algo_name + " runs #" + str(p)
            print self.algo_name + ": " + str(self.ef.value(self.func.getOptimal()))
            print "Function Evaluations: " + str(self.ef.getFunctionEvaluations()-iters)
            print "Iters: " + str(iters)
            print "####"


# run algorithms
rhc = ContinuousPeaks("RHC")
rhc.run_problem()

sa = ContinuousPeaks("SA")
sa.run_problem()

ga = ContinuousPeaks("GA")
ga.run_problem()

mimic = ContinuousPeaks("MIMIC")
mimic.run_problem()