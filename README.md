# Quantum Chemistry with PySCF

## Overview

This project demonstrates the use of the PySCF (Python-based Simulations of Chemistry Framework) library for quantum chemistry calculations. It focuses on solving the electronic Hamiltonian of molecular systems using ab initio methods. The example provided uses a water molecule to showcase the capabilities of PySCF in performing Hartree-Fock (HF) and Time-Dependent Hartree-Fock (TD-HF) calculations.

## Usage
## Tutorial

For a detailed example of how the PySCF library can be used for quantum chemistry calculations, please see the `activespace.ipynb` Jupyter Notebook included in this repository. This tutorial provides step-by-step instructions and practical insights into computational techniques for analyzing quantum systems.

To access the tutorial, navigate to the `ActiveSpace` directory after cloning the repository and open the Jupyter Notebook:

```bash
cd ActiveSpace
jupyter lab activespace.ipynb

### Prerequisites

Make sure you have Python and the necessary libraries installed. You can install the required packages using pip:

```sh
pip install pyscf matplotlib numpy scipy
```

### Running the Example

1. **Clone the repository**:

    ```sh
    git clone https://github.com/chohnna/ActiveSpace.git
    cd ActiveSpace
    ```

2. **Run the Jupyter Notebook**:

    Open the `activespace.ipynb` notebook in Jupyter and execute the cells to perform the quantum chemistry calculations.

    ```sh
    jupyter notebook activespace.ipynb
    ```

### Example Code

Here is a brief snippet to get you started with PySCF:

```python
import pyscf
from pyscf import gto, scf

# Define the molecule
mol = gto.M(
    atom = 'O 0 0 0; H 0 0 1; H 1 0 0',
    basis = 'sto-3g'
)

# Perform Hartree-Fock calculation
mf = scf.RHF(mol)
energy = mf.kernel()

print('Hartree-Fock energy:', energy)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

1. Szabo, A.; Ostlund, N. S. Modern Quantum Chemistry; McGraw-Hill: New York, 1989.
2. Sun, Q., Berkelbach, T. C., Blunt, N. S., Booth, G. H., Guo, S., Li, Z., Liu, J., McClain, J. D., Sayfutyarova, E. R., Sharma, S., Wouters, S., Chan, G. K.-L. (2017). PySCF: the Python-based simulations of chemistry framework. WIREs Computational Molecular Science, 8(1), e1340. doi:10.1002/wcms.1340
