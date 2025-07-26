Autodiff Userguide  
==================
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
.. code-block:: python

    import jax 
    import jax.numpy as jnp

    def f(x, y): # Function we want to take gradients with respect to
        return jnp.sum(x**2 + y**2) # Output is scalar

    grad_f_x = jax.grad(f, argnums = 0) # Gradient function w.r.t. x
    grad_f_y = jax.grad(f, argnums = 1) # Gradient function w.r.t. y

    # For scalar inputs
    grad_f_x(2.0, 4.0) # Returns df/dx for x = 2.0
    grad_f_y(2.0, 4.0) # Returns df/dy for y = 4.0
    
    xx = jnp.array([1.0, 2.0])
    yy = 4*jnp.ones(2)

    # For vector inputs
    grad_f_x(xx, yy) # Returns [df/dx_1, df/dx_2]

If the function you are interested in does not have a scalar output, there are
additional JAX functions that can be used.  The first function of note is called 
``jax.vjp()``, which computes the vector-Jacobian product (reverse mode 
automatic differentiation).  Scenarios best for using
``jax.vjp`` include when the dimension of the function output is small and the 
input dimension is large.  Below are some examples of how to extra gradients 
using ``jax.vjp()`` and some common notation.

``jax.vjp()``
^^^^^^^^^^^^^^
.. code-block:: python

    def f(x, y):  # Function we want to take gradients with respect to
        return x**2 + y**3  # Output is the dimension of x/y

    def grad_f(f, x, y): # Gradient function
        primals, f_vjp = jax.vjp(f, x, y)  # primals: output of f(), f_vjp: Jacobian vector product function
        cotangent = (jnp.ones_like(primals)) # Cotangent: must be the shape of the output of f() 
                                         # To get the gradients w.r.t the function parameters, 
                                         # the input to f_vjp needs to have all one values
        df_dx, df_dy = f_vjp(cotangent) # Takes derivate with respect to each parameter
        return df_dx, df_dy

    # Test 
    xx = 2*jnp.ones(3) # x input 
    yy = jnp.arange(0.0, 3.0, 1) # y input
    df_dx, df_dy = grad_f(f, xx, yy) # Call gradient function
    # Get derivatives w.r.t. x and y

The primals are the output of the function, given the input parameters you provide. 
The cotangent needs to be the same shape as the primals.  When the cotangent is 
initialized with all ones, you can extra the gradients. ``jax.vjp()`` always returns a vector 
of the same shape/size as the function parameters.

Another useful function is to calculate the Jacobian vector product (forward mode
automatic differentiation) using ``jax.jvp()``. Using ``jax.jvp()`` is best for 
scenarios where the output dimension is significantly greater than the input
dimension.  Only the gradients w.r.t. one parameter can be computed at a time 
using ``jax.jvp()``.  Below are some examples for using ``jax.jvp()``. 

``jax.jvp()``
^^^^^^^^^^^^^^
.. code-block:: python

    def f(x, y):  # Function we want to take gradients with respect to
        return x**2 + y**3  # Output is the dimension of x/y

    def grad_f_x(f, x, y): # Gradient function w.r.t. x
        tangent = [jnp.ones_like(xx), # set ones for the parameter you want to take the gradient with respect to
                    jnp.zeros_like(yy)] # set all other parameters to zeros
        primals, df_dx = jax.jvp(f, [x, y], tangent) 
        return df_dx

    def grad_f_y(f, x, y): # Gradient function w.r.t. y
        tangent = [jnp.zeros_like(xx), jnp.ones_like(yy)]
        primals, df_dy = jax.jvp(f, [x, y], tangent)
        return df_dy

    # Test
    xx = 2*jnp.ones(3) # x input 
    yy = jnp.arange(0.0, 3.0, 1) # y input
    df_dx = grad_f_x(my_func, xx, yy) # Get derivatives w.r.t. x
    df_dy = grad_f_y(my_func, xx, yy) # Get derivatives w.r.t. y

If the tangent is set to all ones, then the output of ``jax.jvp()`` will sum 
over all the partial derivatives for each element of the function output. 
``jax.jvp()`` always returns a vector the same shape/size as the function output.



Important notes
^^^^^^^^^^^^^^^
1. You cannot take the derivative with respect to an integer type.
2. When using ``jax.vjp()``, if your function has multiple output objects, the gradients will be summed over the objects. 
3. To extract gradients using ``jax.jvp()``, only the gradient w.r.t one parameter at a time can be computed.
