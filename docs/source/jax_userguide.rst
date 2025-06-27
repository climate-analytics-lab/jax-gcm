Autodiff Userguide  
=============
Here, we describe how to utilize JAX to calculate gradients for already 
estalished functions (in our case, calculating gradients throughout a 
forward pass of JAX-GCM).  First, we go through JAX basics that are 
important to know before trying to calculate gradients.  Next, we show several 
simple examples of the different automatic differentiation methods that can be 
used to calculate gradient information.  Lastly, we show examples of taking the 
gradients throughout JAX-GCM for calculating sensitivities and for simple
optimization tests.  

JAX Gotchas 
-----------
Here, we describe the basics of `JAX <https://docs.jax.dev/en/latest/quickstart.html>`_ 
and how taking derivatives of functions works within the JAX framework. 
JAX provides a quickstart guide, but for a condensed version of everything you 
need to know before taking derivatives in JAX-GCM, please refer to the following
examples. 

There are several different JAX functions that can be used to calculate 
derivative information.  The simplest case is the ``jax.grad()`` function.  
Here, you can take the gradient of a scalar function with respect to 
your input parameters.  

``jax.grad()``
^^^^^^^^^^^^^^

Important notes
^^^^^^^^^^^^^^^
You cannot take the derivative with respect to an integer type. 
