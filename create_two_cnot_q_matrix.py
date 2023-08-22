import pennylane as qml
from pennylane import numpy as np
dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev)
def circuit(control_qubit_index1,target_qubit_index1,control_qubit_index2,target_qubit_index2):
    qml.CNOT(wires=[control_qubit_index1,target_qubit_index1])
    qml.CNOT(wires=[control_qubit_index2,target_qubit_index2])
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]
x = np.asarray( np.load("permutation_two_cnot.npy") )
y = x.numpy()
file_index = 101
for index_file in range(24):
  print( index_file )
  q_matrix = np.asarray( qml.matrix( circuit )( y[ index_file, 0 ]-1, y[ index_file, 1 ]-1, y[ index_file, 2 ]-1, y[ index_file, 3 ]-1 ) )
  q_matrix_real = np.real( q_matrix )
  q_matrix_imag = np.imag( q_matrix )
  np.save( "q_matrix_real_two_cnot" + str( file_index ) + ".npy", q_matrix_real )
  np.save( "q_matrix_imag_two_cnot" + str( file_index ) + ".npy", q_matrix_imag )
  file_index = file_index + 1
