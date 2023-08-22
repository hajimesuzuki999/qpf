# qpf
Quantum pre-processing filter for image classification

This repository contains MATLAB scripts for performing multi-class image classification using a quantum pre-processing filter (QPF) and neural networks.  Three datasets are used as follows:

- MNIST (http://yann.lecun.com/exdb/mnist/)
- EMNIST (https://www.westernsydney.edu.au/icns/resources/reproducible_research3/publication_support_materials2/emnist)
- GTSRB (https://benchmark.ini.rub.de/)

The following MATLAB scripts are provided:

- classify_mnist_nn.m: Image classification by NN against MNIST dataset.
- classify_mnist_encoding_only.m: Image classification by QPF-NN against MNIST dataset using a quantum circuit with encoding only.
- classify_mnist_one_cnot.m: Image classification by QPF-NN against MNIST dataset using a quantum circuit with one CNOT.
- classify_mnist_two_cnot.m: Image classification by QPF-NN against MNIST dataset using a quantum circuit with two CNOTs.
- classify_emnist_nn.m: Image classification by NN against EMNIST dataset.
- classify_emnist_encoding_only.m: Image classification by QPF-NN against EMNIST dataset using a quantum circuit with encoding only.
- classify_emnist_one_cnot.m: Image classification by QPF-NN against EMNIST dataset using a quantum circuit with one CNOT.
- classify_emnist_two_cnot.m: Image classification by QPF-NN against EMNIST dataset using a quantum circuit with two CNOTs.
- classify_gtsrb_nn.m: Image classification by NN against GTSRB dataset.
- classify_gtsrb_encoding_only.m: Image classification by QPF-NN against GTSRB dataset using a quantum circuit with encoding only.
- classify_gtsrb_one_cnot.m: Image classification by QPF-NN against GTSRB dataset using a quantum circuit with one CNOT.
- classify_gtsrb_two_cnot.m: Image classification by QPF-NN against GTSRB dataset using a quantum circuit with two CNOTs.

Pre-processed dataset ready to be used by the MATLAB scripts above is provided only for GTSRB.  Please download other datasets from respective web sites.  MNIST dataset (e.g. train-labels.idx1-ubyte, etc.) is read directly by the MATLAB scripts.  EMNIST dataset is provided as MATLAB format by the above web site.

In addition, the following MATLAB scripts are provided:

- classify_mnist_all_one_cnot.m: Image classification by QPF-NN against MNIST dataset using quantum circuits with 12 different arrangements of CNOT.
- classify_mnist_all_two_cnot.m: Image classification by QPF-NN against MNIST dataset using quantum circuits with 24 different arrangements of two CNOTs.
- plot_result_mnist.m: Plot test set accuracy by NN and QPF-NN against MNIST dataset.
- plot_result_emnist.m: Plot test set accuracy by NN and QPF-NN against EMNIST dataset.
- plot_result_gtsrb.m: Plot test set accuracy by NN and QPF-NN against GTSRB dataset.
- plot_validation_accuracy_one_cnot.m: Plot test set accuracy by QPF-NN against MNIST dataset using quantum circuits with 12 different arrangements of CNOT.
- plot_validation_accuracy_two_cnot.m: Plot test set accuracy by QPF-NN against MNIST dataset using quantum circuits with 24 different arrangements of two CNOTs.

The following Python scripts are used to build quantum circuits with different one CNOT and two CNOTs arrangements on PennyLane.  

-  create_one_cnot_q_matrix.py
-  create_two_cnot_q_matrix.py

The quantum circuits are described as unitary matrices and are saved as real and imaginary separate files in NPY format.  The exported NPY files are named as follows:

- One CNOT, real part: q_matrix_real_one_cnot101.npy to q_matrix_real_one_cnot112.npy.
- One CNOT, imag part: q_matrix_imag_one_cnot101.npy to q_matrix_imag_one_cnot112.npy.
- Two CNOTs, real part: q_matrix_real_two_cnot101.npy to q_matrix_real_two_cnot124.npy.
- Two CNOTs, imag part: q_matrix_imag_two_cnot101.npy to q_matrix_imag_two_cnot124.npy.

12 different permutations for one CNOT is generated internally within the create_one_cnot_q_matrix.py Python script.  24 different permutations are generated by create_permutation_two_cnot.m MATLAB script and are saved in a file permutation_two_cnot.npy NPY file.  This file is needed in the execution of create_two_cnot_q_matrix.py Python script.

To read and write NPY files on MATLAB, please download npy-matlab (https://github.com/kwikteam/npy-matlab).  In particular, the following files should be in the working directory.

- constructNPYheader.m
- datToNPY.m
- readNPY.m
- readNPYheader.m
- writeNPY.m

MATLAB scripts have been tested using MATLAB 2021b running on Windows PCs.  Deep Learning Toolbox is required.
