# ReactGPU

## Overview

This package simulates reaction-diffusion systems of the form
$$\partial_t \vec\rho = \vec{R} + \mathbf{D} \nabla^2 \vec\rho\,$$
where $\vec{\rho} = (\rho^1, \dots, \rho^m)^\top \in \mathcal{C} \cong \mathbb{R}^m$ is an $m$-component chemical field, $\vec{R} \in \mathcal{C}$ is the chemical flux and $\mathbf{D}$ is a diagonal matrix of diffusion coefficients.

Integration of the system is performed using in CUDA via an Euler forward method with a second-order finite difference stencil. The front-end has a python interface.


## Installation

```bash
mkdir build
cd build
cmake ..
make -j8
```

## Running simulations.

To use the python front-end to run simulations see the example [simulatejacobs.py](examples/simulatejacobs.py).

## How to implement new models

To add a new reaction-diffusion model you must add your definitions to the following file
* Models are defined in [models.h](src/models.h): the meat of your definitions go here. Here you specify the chemical flux $\vec{R}$ that defines a particular model, as well as the model parameters; the number of fields is inferred from this information. Parameters passed by the host can differ from those passed to the GPU device, to allow optional pre-processing (cf. examples).
* Python interface is defined in [reactor.cc](src/reactor.cc): here you define python bindings for the member functions. Again, follow the examples.
* A line declaring the model needs to be added to the end of [reactor.cu](src/reactor.cu) so that CUDA will compile the device code for the new model.
