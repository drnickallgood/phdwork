import dimod
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
import neal

# we only want to prefer states that are same value in each qubit
# so 0,0 or 1,1
#Q = {(0,0): 3, (0,1): 50, (1,0): 51, (1,1): 99}

h = {}

exact_sampler = dimod.ExactSolver()
#sampler = EmbeddingComposite(DWaveSampler())

#bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
bqm = dimod.BinaryQuadraticModel.from_ising(h,J)

#response = exact_sampler.sample(bqm, sweep_size=1000)
response = exact_sampler.sample(bqm)
#response = sampler.sample(bqm, num_reads=10000)

#print(response.record)
print(response.first)
#print(response.first.sample)
#print(response.first.energy)

#sampleset = sampler.sample_ising(h, J, num_reads=1000)

#print(sampleset)



