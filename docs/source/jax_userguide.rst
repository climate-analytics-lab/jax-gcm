Autodiff Userguide  
=============
Here, we describe how to utilize JAX to calculate gradients for already estalished functions (in our case,
calculating gradients throughout a forward pass of JAX-GCM).  First, we go through JAX basics that are 
important to know before trying to calculate gradients.  Next, we show several simple examples of the 
different automatic differentiation methods that can be used to calculate gradient information.  Lastly, 
we show examples of taking the gradients throughout JAX-GCM for calculating sensitivities and for simple
optimization tests.  

