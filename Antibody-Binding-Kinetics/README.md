# Antibody Binding Kinetics: Numerical Methods and Machine Learning

## Project Overview

This project investigates the **Antibody Binding Kinetics** model by combining
classical numerical methods with modern machine learning techniques. The
objective is to simulate antibody-antigen interactions, compare different
numerical approaches, and evaluate the capability of machine learning models
to predict the system dynamics.

The project is implemented in Python and includes visualization, performance
evaluation, and comparative analysis.

## Features

- Numerical solution of the Antibody Binding Kinetics model
  - Finite Difference Method (FDM)
  - Finite Element Method (FEM)
- Machine learning approximations
  - Physics-Informed Neural Network (PINN)
  - Deep Operator Network (DeepONet)
  - Graph Neural Network (GNN)
- Comparison between numerical and machine learning approaches
- Visualization of concentration profiles and prediction results

## Project Structure

```
Antibody-Binding-Kinetics/
│
├── numerical_methods/
│   ├── antibody_fem.py        # Finite Element Method solver
│   └── coupled_system.py      # Finite Difference Method solver
│
├── machine_learning/
│   ├── pinn_model.ipynb       # Physics-Informed Neural Network
│   ├── deeponet.ipynb         # Deep Operator Network
│   └── gnn_model.ipynb        # Graph Neural Network
│
├── docs/
│   ├── Final_Project_Report.pdf
│   ├── Presentation.pptx
│   └── references/
│       └── Schiesser_PDE_Biomedical_Engineering.pdf
│
├── images/
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Numerical Methods

### Finite Difference Method (FDM)
`numerical_methods/coupled_system.py` discretizes the coupled bulk-diffusion /
surface-binding governing equations on a structured 1D grid and advances the
solution in time using an explicit (FTCS) scheme, with the time step chosen
to satisfy the diffusive stability limit.

### Finite Element Method (FEM)
`numerical_methods/antibody_fem.py` converts the governing equations into
their weak form and approximates the solution using linear finite elements
with Gauss quadrature for element-matrix assembly, including full plotting
utilities for concentration profiles.

## Machine Learning Models

### Physics-Informed Neural Network (PINN)
`machine_learning/pinn_model.ipynb` — a neural network constrained by the
governing differential equations (via automatic differentiation) to learn
physically consistent solutions without requiring a dense set of labeled
training data.

### DeepONet
`machine_learning/deeponet.ipynb` — a deep operator learning architecture
that learns the nonlinear solution operator of the system, enabling fast
inference across varying parameters/initial conditions.

### Graph Neural Network (GNN)
`machine_learning/gnn_model.ipynb` — a graph-based neural network (built on
PyTorch Geometric) used to capture spatial relationships across the
discretized computational domain.

## Technologies Used

- Python
- NumPy / SciPy
- Matplotlib / Pandas
- Scikit-learn
- TensorFlow / Keras
- PyTorch / PyTorch Geometric
- Jupyter Notebook

## Results

The project compares classical numerical methods against machine learning
models across:

- Accuracy
- Computational efficiency
- Prediction capability
- Numerical stability

See `docs/Final_Project_Report.pdf` and `docs/Presentation.pptx` for full
results, figures, and discussion.

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd Antibody-Binding-Kinetics

# Install dependencies
pip install -r requirements.txt

# Run the classical numerical solvers
python numerical_methods/coupled_system.py
python numerical_methods/antibody_fem.py

# Explore the machine learning notebooks
jupyter notebook machine_learning/
```

> Note: the notebooks were developed for Google Colab and include
> Colab-specific cells (e.g. `google.colab.drive` mounting). Remove or adapt
> these cells to run locally.

## Future Work

- Improve prediction accuracy using larger datasets.
- Extend the model to three-dimensional domains.
- Explore additional deep learning architectures.
- Optimize computational performance.

## Authors

- Ahmed Hatem
- Ahmed El Manzalawi
- Salma Khaled
- Fatma Saied
- Mohab Hisham

Biomedical Engineering

## License

This project is released under the [MIT License](LICENSE).
