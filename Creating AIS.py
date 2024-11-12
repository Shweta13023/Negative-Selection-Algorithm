#!/usr/bin/env python
# coding: utf-8

# In[1]:


#All the installations done to make the code work
get_ipython().system('pip install Qiskit')
get_ipython().system('pip install --user qiskit')
get_ipython().system('pip install qiskit-aer')
get_ipython().system('pip install qiskit --upgrade')
get_ipython().system('pip install qiskit-algorithms')
get_ipython().system('pip install qiskit-ibm-runtime')
get_ipython().system('pip install scipy')
get_ipython().system('pip install qiskit-algorithms')


# In[2]:


import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Pauli , SparsePauliOp 
import numpy as np
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit import assemble
from qiskit_ibm_runtime import ( EstimatorV2 as Estimator , QiskitRuntimeService ,                               )
from qiskit.quantum_info import Pauli , SparsePauliOp
from qiskit.visualization import plot_histogram
from qiskit import ClassicalRegister , QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import Parameter , QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA


# In[ ]:


import numpy as np
from numpy.random import uniform


# In[3]:


import random

# Negative Selection Algorithm
class NegativeSelection:
    def __init__(self, population_size, detector_size):
        self.population_size = population_size
        self.detector_size = detector_size
        self.population = []
        self.detectors = []

    def generate_population(self):
        for _ in range(self.population_size):
            self.population.append(random.random())

    def generate_detectors(self):
        for _ in range(self.detector_size):
            self.detectors.append(random.random())

    def match(self, antigen):
        for detector in self.detectors:
            if abs(detector - antigen) < 0.1:
                return True
        return False

# Artificial Immune Network (AIN)
class ArtificialImmuneNetwork:
    def __init__(self, network_size):
        self.network_size = network_size
        self.network = []

    def generate_network(self):
        for _ in range(self.network_size):
            self.network.append(random.random())

    def stimulate(self, antigen):
        for node in self.network:
            if abs(node - antigen) < 0.1:
                return True
        return False

# Clonal Selection Algorithm
class ClonalSelection:
    def __init__(self, population_size, antibody_size):
        self.population_size = population_size
        self.antibody_size = antibody_size
        self.population = []
        self.antibodies = []

    def generate_population(self):
        for _ in range(self.population_size):
            self.population.append(random.random())

    def generate_antibodies(self):
        for _ in range(self.antibody_size):
            self.antibodies.append(random.random())

    def select(self, antigen):
        for antibody in self.antibodies:
            if abs(antibody - antigen) < 0.1:
                return True
        return False

# Danger Theory
class DangerTheory:
    def __init__(self, danger_signals):
        self.danger_signals = danger_signals

    def detect_danger(self, antigen):
        for signal in self.danger_signals:
            if abs(signal - antigen) < 0.1:
                return True
        return False

# Dendritic Cell Algorithm
class DendriticCell:
    def __init__(self, signals):
        self.signals = signals

    def process_signals(self, antigen):
        for signal in self.signals:
            if abs(signal - antigen) < 0.1:
                return True
        return False

# Example usage
ns = NegativeSelection(100, 10)
ns.generate_population()
ns.generate_detectors()
print(ns.match(0.5))

ain = ArtificialImmuneNetwork(50)
ain.generate_network()
print(ain.stimulate(0.5))

cs = ClonalSelection(100, 10)
cs.generate_population()
cs.generate_antibodies()
print(cs.select(0.5))

dt = DangerTheory([0.2, 0.4, 0.6])
print(dt.detect_danger(0.5))

dc = DendriticCell([0.2, 0.4, 0.6])
print(dc.process_signals(0.5))


# In[4]:





# In[5]:


#Negative selection algorithm using the quantum computing frameworks
#initializing the parameters
from qiskit import  transpile
def parameters(self, population_size, detector_size):
        self.population_size = population_size
        self.detector_size = detector_size
        self.population = []
        self.detectors = []
#encoding  the quantam circuit and antigen
def encode_antigen (qc , antigen):
    for i, value in enumerate(antigen):
        qc.rx(value, i)
    qc.cx(0,1)
#Encoding the Quantum Similarity
def quantum_similarity(qc, encoded_detector, shots=1024):
    """Compute similarity between qc (antigen) and encoded_detector using quantum measurement."""
    # Append encoded detector to quantum circuit
    qc.append(encoded_detector.to_instruction(), range(qc.num_qubits))
    
    # Measure all qubits in the circuit
    qc.measure_all()
    
    # Choose the backend
    backend = Aer.get_backend('qasm_simulator')
    
    # Transpile the quantum circuit for the backend
    transpiled_circuit = transpile(qc, backend)
    
    # Run the job using the backend and retrieve results
    job = backend.run(transpiled_circuit, shots=shots).result().get_counts()
    
    # Compute similarity based on measurement results
    similarity = sum([v for v in job.values()]) / shots
    
    return similarity
def quantum_negative_selection(dataset, encoded_detectors, r):
    """Perform Negative Selection Algorithm using quantum computing."""
    non_self_indices = []

    for antigen_idx, antigen in enumerate(dataset):
        is_self = False
        for encoded_detector in encoded_detectors:
            qc = QuantumCircuit(len(antigen))
            encode_antigen(qc, antigen)
            similarity = quantum_similarity(qc, encoded_detector)
            if similarity > r:
                is_self = True
                break
        if not is_self:
            non_self_indices.append(antigen_idx)

    return non_self_indices


# In[6]:


def main():
    # Parameters
    num_detectors = 50  # Number of detectors
    num_antigens = 200  # Number of antigens (data points)
    r = 0.1  # Similarity threshold

    # Generate random dataset (antigens)
    np.random.seed(0)
    dataset = np.random.rand(num_antigens, 10)  # Example dataset with 10 features

    # Initialize quantum detectors
    detectors = np.random.rand(num_detectors, 10)  # Example detectors with 10 features
    encoded_detectors = []
    for detector in detectors:
        qc_det = QuantumCircuit(len(detector))
        encode_antigen(qc_det, detector)
        encoded_detectors.append(qc_det)

    # Perform NSA using quantum computing
    non_self_indices = quantum_negative_selection(dataset, encoded_detectors, r)

    # Print results
    print("Number of non-self patterns detected:", len(non_self_indices))
    print("Indices of non-self patterns:", non_self_indices)

if __name__ == "__main__":
    main()


# In[ ]:




