import cirq
import qiskit
import numpy as np
from qiskit.circuit.library import RYGate

def x_gate_generation(n_qubits):
    x_gate_seq = []
    i = 1
    while i <= n_qubits:
        x_gate_seq = x_gate_seq + [i] + x_gate_seq
        i += 1

    return x_gate_seq+[0]

def color_conversion(prob):
    pass

def encode_cirq(image):
    pass

def encode_qiskit(image, n_qubits):

    # last is color_encoding qubit

    q = qiskit.QuantumRegister(n_qubits+1)
    qc = qiskit.QuantumCircuit(q)

    # linear scaling
    theta_pixels = image*np.pi/255

    for idx in range(n_qubits):
        qc.h(idx)

    x_gate_seq = x_gate_generation(n_qubits=n_qubits)
    
    for i in range(len(image)):
        pixel = theta_pixels[i]
        frqi_ry = RYGate(pixel).control(n_qubits)
        qc.append(frqi_ry, range(n_qubits+1))
        x_gates = x_gate_seq[i]
        for j in range(x_gates):
            qc.x(j)

    return qc

def decode(histogram):
    pass

    '''
    if 1 in histogram.keys():
        image=[[0,0],[0,0]]
    else:
        image=[[1,1],[1,1]]
    return image
    '''

def encode_qiskit_abandoned(image, n_qubits):

    # second last is color_encoding qubit
    # last is color_scale_qubit

    q = qiskit.QuantumRegister(n_qubits+2)
    qc = qiskit.QuantumCircuit(q)

    # scale
    scale_factor = np.sum(image)
    scale_theta = np.arctan(scale_factor)

    # normalize
    normalized_pixels = image/scale_factor

    # linear scaling
    theta_pixels = image*np.pi/255

    # encode scale_theta in last qubit
    qc.u(scale_theta,0,0,n_qubits+1)

    for idx in range(n_qubits):
        qc.h(idx)

    x_gate_seq = x_gate_generation(n_qubits=n_qubits)
    
    for i in range(len(image)):
        pixel = theta_pixels[i]
        frqi_ry = RYGate(pixel).control(n_qubits)
        qc.append(frqi_ry, range(n_qubits+1))
        x_gates = x_gate_seq[i]
        for j in range(x_gates):
            qc.x(j)

    return qc