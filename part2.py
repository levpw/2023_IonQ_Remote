#your submission

from part1 import *

# Functions 'encode' and 'decode' are dummy.

def my_histogram_to_category(histogram):
    pass

def run_part2(image):
    #loade the quantum classifier circuit
    classifier=qiskit.QuantumCircuit.from_qasm_file('quantum_classifier.qasm')
    
    #encode image into circuit
    circuit=encode(image)
    
    #append with classifier circuit
    nq1 = circuit.width()
    nq2 = classifier.width()
    nq = max(nq1, nq2)
    qc = qiskit.QuantumCircuit(nq)
    qc.append(circuit.to_instruction(), list(range(nq1)))
    qc.append(classifier.to_instruction(), list(range(nq2)))
    
    #simulate circuit
    histogram=my_simulate(qc)
        
    #convert histogram to category
    label=my_histogram_to_category(histogram)
        
    return circuit,label

#score