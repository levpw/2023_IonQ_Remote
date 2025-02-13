import qiskit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
import pickle
import json
import os
import sys
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = '.'

#define utility functions

def simulate(circuit: qiskit.QuantumCircuit) -> dict:
    """Simulate the circuit, give the state vector as the result."""
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    state_vector = result.get_statevector()
    
    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population
    
    return histogram


def histogram_to_category(histogram):
    """This function takes a histogram representation of circuit execution results, and processes into labels as described in
    the problem description."""
    assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
        
    return positive

def count_gates(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """Returns the number of gate operations with each number of qubits."""
    counter = Counter([len(gate[1]) for gate in circuit.data])
    #feel free to comment out the following two lines. But make sure you don't have k-qubit gates in your circuit
    #for k>2
    #for i in range(2,20):
    #    assert counter[i]==0
        
    return counter


def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(255*image1,255*image2)

def test():
    #load the actual hackthon data (fashion-mnist)
    images=np.load(data_path+'/images.npy')
    labels=np.load(data_path+'/labels.npy')
    
    #test part 1

    n=len(images)
    mse=0
    gatecount=0

    for image in images:
        #encode image into circuit
        circuit,image_re=run_part1(image)
        image_re = np.asarray(image_re)

        #count the number of 2qubit gates used
        gatecount+=count_gates(circuit)[2]

        #calculate mse
        mse+=image_mse(image,image_re)

    #fidelity of reconstruction
    f=1-mse/n
    gatecount=gatecount/n

    #score for part1
    score_part1=f*(0.999**gatecount)
    
    #test part 2
    
    score=0
    gatecount=0
    n=len(images)

    for i in range(n):
        #run part 2
        circuit,label=run_part2(images[i])

        #count the gate used in the circuit for score calculation
        gatecount+=count_gates(circuit)[2]

        #check label
        if label==labels[i]:
            score+=1
    #score
    score=score/n
    gatecount=gatecount/n

    score_part2=score*(0.999**gatecount)
    
    print(score_part1, ",", score_part2, ",", data_path, sep="")


############################
#      YOUR CODE HERE      #
############################

import qiskit
from qiskit import transpile
from qiskit.providers.aer import QasmSimulator
import numpy as np
from typing import Dict, List
from qiskit.circuit.library import RYGate
from qiskit.compiler import transpile
import cv2

#define utility functions

def my_simulate(circuit) -> dict:
    """Official simulate is not compatible with our FRQI encoder implementation"""
    circuit.measure_all()
    simulator = QasmSimulator()

    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=65535)
    
    result = job.result()
    
    histogram = result.get_counts(compiled_circuit)
    
    return histogram

def x_gate_generation(n_qubits):
    x_gate_seq = []
    i = 1
    while i <= n_qubits:
        x_gate_seq = x_gate_seq + [i] + x_gate_seq
        i += 1

    return x_gate_seq+[0]

# Functions 'encode' and 'decode' are dummy.
def encode(image):
    # last is color_encoding qubit

    # downsample 28 -> 8
    image = cv2.resize(image, (8,8), interpolation=cv2.INTER_AREA)

    img_data = image.flatten()

    # calculated qubits needed for image size
    # n_qubits = np.ceil(2*np.log2(len(image))).astype(int)
    n_qubits = 6

    q = qiskit.QuantumRegister(n_qubits+1)
    qc = qiskit.QuantumCircuit(q)

    # linear scaling
    # theta_pixels = img_data*np.pi/255
    theta_pixels = img_data*np.pi/0.00392156862745098
    #theta_pixels = img_data*255

    for idx in range(n_qubits):
        qc.h(idx)

    x_gate_seq = x_gate_generation(n_qubits)
    
    for i in range(len(img_data)):
        pixel = theta_pixels[i]
        frqi_ry = RYGate(pixel).control(n_qubits)
        qc.append(frqi_ry, range(n_qubits+1))
        x_gates = x_gate_seq[i]
        for j in range(x_gates):
            qc.x(j)

    qc = qc.decompose()
    qc = transpile(qc, routing_method = 'sabre')

    return qc

def decode(histogram):
    # compressed to 8x8
    data_len = 64
    recovered_image_cos = np.zeros(data_len)
    recovered_image_sin = np.zeros(data_len)

    for key in histogram:
        idx_bin = key[1:]
        idx = int(idx_bin,2)
        if key[0] == '0':
            recovered_image_cos[idx] = histogram[key]
        else:
            recovered_image_sin[idx] = histogram[key]

    prob = recovered_image_sin/(recovered_image_sin+recovered_image_cos)
    re_image = prob.reshape(8,8)
    re_image = cv2.resize(re_image, (28,28), interpolation=cv2.INTER_LINEAR)*0.00392156862745098/np.max(prob)

    return re_image

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=my_simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit,image_re

def run_part2(image):
    # load the quantum classifier circuit
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
    histogram=simulate(qc)
        
    #convert histogram to category
    label=histogram_to_category(histogram)
    
    #thresholding the label, any way you want
    if label>0.5:
        label=1
    else:
        label=0
        
    return circuit,label

############################
#      END YOUR CODE       #
############################

test()