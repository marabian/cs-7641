# source: https://github.com/pushkar/ABAGAIL/blob/master/jython/travelingsalesman.py
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
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
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
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array



random = Random()


class TravelingSalesman:
    N = 50
    def __init__(self, algo_name):
        random.setSeed(56)
        self.algo_name = algo_name

        self.problem_name = "travelingsalesman"

        odd = DiscretePermutationDistribution(self.N)
        nf = SwapNeighbor()
        mf = SwapMutation()
        cf = TravelingSalesmanCrossOver(ef)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

        if algo_name == "RHC":
            rhc = RandomizedHillClimbing(hcp)
            self.func = rhc
        elif algo_name == "SA":
            sa = SimulatedAnnealing(1e13, .90, hcp)
            self.func = sa
        elif algo_name == "GA":
            gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
            ga = StandardGeneticAlgorithm(200, 150, 10, gap)
            self.func = ga
        elif algo_name == "MIMIC":
            fill = [self.N] * self.N
            ranges = array('i', fill)
            odd = DiscreteUniformDistribution(ranges)
            df = DiscreteDependencyTree(.1, ranges)
            pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
            pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
            self.mimic = MIMIC(200, 20, pop)
            self.func = self.mimic


    def run_problem(self, ef, iters=5000, runs=10):
        for p in range(1, runs+1):
            fit = FixedIterationTrainer(self.func, 10)
            FILE_NAME="{}_{}_{}.csv".format(self.algo_name, self.problem_name, str(p))
            OUTPUT_FILE = os.path.join("data/csv", FILE_NAME)
            with open(OUTPUT_FILE, "wb") as data:
                writer= csv.writer(data, delimiter=',')
                writer.writerow(["iters","fevals","fitness"])
                for i in range(0, iters, 10):
                    fit.train()
                    #print str(i) + ", " + str(ef.getFunctionEvaluations()) + ", " + str(ef.value(func.getOptimal()))
                    writer.writerow([i, ef.getFunctionEvaluations()-i, ef.value(self.func.getOptimal())])
            
            print self.algo_name + " run #" + str(p)
            print self.algo_name + " Inverse of Distance: " + str(ef.value(self.func.getOptimal()))
            print "Route:"

            # if self.algo_name == "MIMIC":
            #     path = []
            #     optimal = self.mimic.getOptimal()
            #     fill = [0] * optimal.size()
            #     ddata = array('d', fill)
            #     for i in range(0,len(ddata)):
            #         ddata[i] = optimal.getContinuous(i)
            #     order = ABAGAILArrays.indices(optimal.size())
            #     ABAGAILArrays.quicksort(ddata, order)
            #     print order
            # else:
            #     path = []
            #     for x in range(0,self.N):
            #         path.append(self.func.getOptimal().getDiscrete(x))
            #     print path



N = TravelingSalesman.N
points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

ef = TravelingSalesmanRouteEvaluationFunction(points)

# run algorithms
rhc = TravelingSalesman("RHC")
rhc.run_problem(ef)

sa = TravelingSalesman("SA")
sa.run_problem(ef)

ga = TravelingSalesman("GA")
ga.run_problem(ef)



# for mimic we use a sort encoding
ef = TravelingSalesmanSortEvaluationFunction(points)
mimic = TravelingSalesman("MIMIC")
mimic.run_problem(ef)
