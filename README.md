# Bayesian inference in [JAX](https://github.com/google/jax)
Brett Morris

## Install

<div class="alert alert-block alert-warning">
    <strong>WARNING</strong>: It's possible that installing these dependencies in your usual working environment will break your other packages. Don't do that!
</div>


Please use `conda` or [`venv`](https://docs.python.org/3/library/venv.html) to create an isolated python environment for this tutorial.

For conda:

```bash
conda create -n jax-demo python=3.12
conda activate jax-demo
```

For venv:

```bash
python -m venv /path/to/new/virtual/environment
source <environment_name>/bin/activate
```


#### Installing jax

Try the following on your laptop (CPU): 
```bash
python -m pip install --upgrade "jax[cpu]"
```
jax can run on GPUs and TPUs but requires specific builds for each architecture. Check out the [jax installation docs](https://github.com/google/jax#installation) for details.

#### Other dependencies

Other installations needed for this tutorial can be installed with: 

```bash
python -m pip install numpy scipy matplotlib numpyro arviz corner ipywidgets
```

## Why jax?

jax leverages [just-in-time code compilation](https://docs.jax.dev/en/latest/jit-compilation.html), with [automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html), and [_accelerated linear algebra_](https://github.com/openxla/xla) with a _numpy-like API_ to calculate blazing fast, differentiable models. Let's break that down: 

* Automatic differentiation allows you to compute gradients of your mathematical models without explicitly deriving gradients for each function. These gradients can be used in gradient-based inference techniques like gradient descent optimization, or Hamiltonian Monte Carlo.
* Accelerated linear algebra package is an optimizing compiler designed for machine learning. You write Python code and it gets just-in-time compiled for your computer architecture (CPU or GPU) at runtime.


