import pennylane as qml
from pennylane import numpy as np
dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev)
def circuit(control_qubit_index,target_qubit_index):
    qml.CNOT(wires=[control_qubit_index,target_qubit_index])
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]
file_index = 101
for control_qubit_index in range(4):
  for target_qubit_index in range(4):
    if control_qubit_index != target_qubit_index:
      print( [ control_qubit_index, target_qubit_index ] )
      q_matrix = np.asarray( qml.matrix( circuit )( control_qubit_index, target_qubit_index ) )
      q_matrix_real = np.real( q_matrix )
      q_matrix_imag = np.imag( q_matrix )
      np.save( "q_matrix_real_one_cnot" + str( file_index ) + ".npy", q_matrix_real )
      np.save( "q_matrix_imag_one_cnot" + str( file_index ) + ".npy", q_matrix_imag )
      file_index = file_index + 1

