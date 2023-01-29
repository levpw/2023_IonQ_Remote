#submission to part 1, you should make this into a .py file

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