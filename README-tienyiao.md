# Tien-Yiao's README file

This README is to help with slab ocean model development.
This branch translates the slab ocean model in SpeedyPy into JAX-gcm.


## Needed package

- pip: `pip install jax dinosaur`.


## TODO

- [v] Figure out how to test the code.
- [v] Create slab ocean model parameter class.
- Translates component
    1. [v] init
    2. [v] coupled atm ocn
    3. [v] run
- Put a demo for other people to work on.
- Write a recorder that manages data output.
- Discuss and figure out the role of different initialization functions. Currently, I think the initialization is too coupled.



