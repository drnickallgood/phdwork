import dimod
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

# we only want to prefer states that are same value in each qubit
# so 0,0 or 1,1
#Q = {(0,0): 3, (0,1): 50, (1,0): 51, (1,1): 99}

# Using 64-bit hash values
#Q = {(0,1): 3891962469035829225, (1,2): 261024130118881976, (2,3): 1278075525769194288, (3,4): 1461127261386963019}

h = {}
J = {(3891962469035829225, 261024130118881976): 0.0, (1278075525769194288, 1461127261386963019): 0.0 }


#Q = {('0x1234', '0x4321', '0x1111', '0x1221'): -0.5}

sampler = EmbeddingComposite(DWaveSampler())

#bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

#response = sampler.sample(bqm, num_reads=10000)

#print(response)

sampleset = sampler.sample_ising(h, J, num_reads=1000)

print(sampleset)


