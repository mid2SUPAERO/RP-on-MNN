# Mechanical Neural Network with Piezoelectric Actuation

This repository contains a Python implementation of a **Mechanical Neural Network (MNN)** based on a 2D lattice structure of beam elements actuated by piezoelectric patches.

The project explores how mechanical systems can emulate neural network behavior, where:
- **nodes** act as neurons,
- **beams** act as connections,
- **stiffness and actuation** act as weights.

The goal is to control the structural response of the network by tuning the voltages applied to piezoelectric patches, allowing the structure to reproduce prescribed displacement patterns.

---

##  Main Features

- **Finite Element Model (FEM)**
  - 2D Euler–Bernoulli beam elements
  - Arbitrary mesh refinement (user-defined number of elements per beam)
  - Assembly of global stiffness matrix and reduced system

- **Piezoelectric Actuation**
  - Two patches per beam (top & bottom)
  - Conversion of voltage into equivalent axial forces and bending moments
  - Fully coupled with FEM formulation

- **Optimization Framework**
  - Gradient-based optimization (SLSQP)
  - Design variables: voltages on piezoelectric patches
  - Voltage bounds: ±200 V

- **Mechanical Neural Network Behaviour**
  - Ability to "learn" displacement targets
  - Supports:
    - single load case optimization
    - multiple load cases (MNN behaviour)

---

##  Concept

The structure behaves as a **mechanical analogue of a neural network**:

- Inputs → applied forces  
- Outputs → nodal displacements  
- Weights → beam stiffness + piezoelectric actuation  

After optimization, the desired behaviour becomes an **intrinsic property of the structure**.

---

##  Capabilities

- High accuracy for **single load case** (millimetric precision)
- Ability to approximate **multiple behaviours simultaneously**
- Visualization of:
  - deformed structure
  - piezoelectric voltage distribution
  - target vs achieved displacements

---

##  How It Works

1. Generate FEM mesh from lattice geometry  
2. Apply boundary conditions and loads  
3. Assemble stiffness matrix  
4. Convert piezo voltages → equivalent loads  
5. Solve FEM system  
6. Optimize voltages to match target displacements  

---

##  Limitations

- Static model (no dynamics or vibrations)
- Linear piezoelectric behaviour (no hysteresis)
- Simplified electromechanical coupling
- Limited network size

---

##  Possible Improvements

- Include dynamic effects (vibration, time response)
- Nonlinear piezoelectric modelling
- Optimization of structural properties (e.g. Young’s modulus)
- Larger and more complex network architectures

---

##  Applications

- Morphing aerospace structures  
- Adaptive and smart materials  
- Structural control systems  
- Mechanical computation  

---

## Authors

- Francesco Ardrizzini
- Tutor: Prof. Joseph Morlier
