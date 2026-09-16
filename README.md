# RAMMs

A Python package for constructing, visualizing, and analyzing **Reconfigurable Architected Mechanical Metamaterials (RAMMs)**.

RAMMs are modular mechanical structures whose global geometry and mechanical behavior emerge from the configuration and interactions of their constituent units. This package provides tools for computationally representing these structures and studying their kinematics, contact constraints, and reconfiguration.

Motivation

This project is inspired by Polycatenated Architected Materials (PAMs) developed by Chiara Daraio's group at Caltech: [3D Polycatenated Architected Materials (PAMs) Zhou et al. in Science](https://doi.org/10.1126/science.adr9713).

## Goals

The package is intended to provide a common framework for:

* Constructing RAMM units and chains from geometric parameters
* Visualizing RAMM configurations
* Computing the positions and orientations of individual units
* Detecting and tracking contacts between structural elements
* Formulating contact constraints and their Jacobians
* Analyzing the remaining mobility of a structure
* Exploring contact-induced transitions, jamming, and reconfiguration
* Eventually simulating and controlling larger RAMM structures

A major motivation is to connect the **geometry of the structure** to its **available motion**:

```text
Geometry
   ↓
Active Contacts
   ↓
Constraints
   ↓
Constraint Jacobian
   ↓
Remaining Mobility
```

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/e-chesoni/ramms.git
cd ramms
pip install -e .
```

Editable installation is recommended during development so changes to the source code are immediately available without reinstalling the package.

## Example

```python
from ramms import RAMM_Chain
```

As the API develops, this section will include examples for constructing RAMM chains, changing their configuration, visualizing their geometry, and performing contact and mobility analysis.

## Planned Capabilities

Development will progressively add support for:

**Geometry and kinematics**

* Individual RAMM unit geometry
* Chains of interconnected units
* Forward kinematics
* Configuration-space representations

**Contact mechanics**

* Node–segment gap functions
* Segment–segment gap functions
* Active-contact detection
* Contact constraint Jacobians

**Mobility analysis**

* Singular value decomposition (SVD) of constraint Jacobians
* Identification of allowable first-order motions
* Mechanism and self-stress analysis
* Detection of contact-induced mobility changes and jamming

**Simulation and control**

* Physics-based simulation
* Reconfiguration trajectories
* Underactuated motion and control
* Larger RAMM assemblies

## Research Context

A central question in this project is how contact changes the available degrees of freedom of a reconfigurable mechanical structure.

For a configuration vector

```math
q = [q_1, q_2, \ldots, q_n]^T,
```

active geometric contacts can be represented by gap functions

```math
g_i(q) = 0.
```

Collecting the derivatives of these constraints produces a contact Jacobian

```math
J(q) = \frac{\partial g}{\partial q}.
```

First-order motions that preserve the active contacts satisfy

```math
J(q)\dot{q} = 0.
```

The null space of `J(q)` therefore provides a local description of the motions available to the structure under its current set of contacts. Changes in the active contact set can alter this null space, producing transitions between different mobility regimes and, potentially, mechanically jammed configurations.

## Status

🚧 **Active research / early development**

The package API and repository structure are expected to change as the RAMM modeling and analysis framework develops.

## License

License information will be added as the project develops.
