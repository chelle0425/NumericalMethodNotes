""" 
- Be able to compute the integral of a function numerically in 1D using several different methods
- More practice at developing algorithms, mathematical reasoning, and implementing those algorithms in code
- Understand the concept of the order (of convergence) of an algorithm, and approaches we can use to improve the accuracy of our results

> Midpoint (rectangle) rule
> Trapezoid rule
> Simpson's rule
    combines expected error from midpt and trap (error midpt = 2 error trap)
> Composite Simpson's rule
    Simpsons rule but groups summations for productivity
> Weddle's rule
    takes 2 diff Simpvalues for interval size and extrapolates Simpson to improve accuracy
"""

#%%
### Numerical Integration (Quadrature)
# splitting functions into smaller intervals and summing segments
# approx value within each interval (numerical discretisation)


### Midpoint rule
# approximates integral over interval with base length and value at midpoint of interval
# sums up subintervals
# error log plot is quadratic-- error dec by a factor of 4

true_area = 1/3
# Midpoint rule example
import numpy as np
import matplotlib.pyplot as plt

# define our x for plotting purposes
x = np.linspace(0, 1, 1000)

def func(x):
    return x ** 2

def midpoint_rule(a, b, function, number_intervals=10):
    " a and b = end points for interval"
    "'function' is the function of x in [a,b]"
    "number_intervals = number of subintervals we split [a,b] into"
    "Returns the integral of function(x) over [a,b]"

    interval_size = (b - a)/number_intervals

    " assert checks whether statement is 'true', and if 'false' raises an 'AssertionError' w optional message"
    " eg assert x == 10, 'x should be equal to 10'"
    
    assert interval_size > 0
    assert type(number_intervals) == int
    
    # Initialise to zero the variable that will contain the cumulative sum of all the areas
    I_M = 0.0
    
    # Find the first midpoint -- i.e. the centre point of the base of the first rectangle
    mid = a + (interval_size/2.0)
    # and loop until we get past b, creating and summing the area of each rectangle
    while (mid < b):
        # Find the area of the current rectangle and add it to the running total
        # this involves an evaluation of the function at the subinterval midpoint
        I_M += interval_size * function(mid)
        # Move the midpoint up to the next centre of the interval
        mid += interval_size

    # Return our running total result
    return I_M

# check the function runs and agrees with our first version used to generate the schematic plot of the method above:
print('midpoint_rule(0, np.pi, np.sin, number_intervals=5) = ', midpoint_rule(0, np.pi, np.sin, number_intervals=5))

# Now let's test the midpoint function by varying no. of rectangles
print("The exact area found by direct integration = 2")
rect_no = []
errors_rect = []
for i in (1, 2, 4, 8, 16, 32, 100, 1000):
    area = midpoint_rule(0, 1, func, i)
    error = abs(area-true_area)
    rect_no.append(i)
    errors_rect.append(error)
    print("Area %d rectangle(s) = %g (error=%g)"%(i, area, abs(area-true_area)))


## Ex 3.1 Log-log plot

fig = plt.figure(figsize=(7, 5))
ax1 = plt.subplot(111)

ax1.set_title('Convergence plot when integrating sin with midpoint and trapezoidal', fontsize=16)
ax1.set_xlabel('log(no. of intervals)', fontsize=16)
ax1.set_ylabel('log(error)', fontsize=16)

# plotting
" log-log plot"
ax1.loglog(rect_no, errors_rect, 'r.-', label = 'midpoint')

### Trapezoid rule
# top --> linear line fit defined by value of function at two end points
# trapezium = 1/2 * h * (a + b)

# Ex 3.2 log-log plot for Trapezoid rule

def trapezoidal_rule(a, b, function, number_intervals=10):
    # see the composite implementation in the homework for a more efficient version
    " a and b = end points for interval"
    "'function' is the function of x in [a,b]"
    "number_intervals = number of subintervals we split [a,b] into"
    "Returns the integral of function(x) over [a,b]"

    interval_size = (b - a)/number_intervals

    assert interval_size > 0
    assert type(number_intervals) == int

    I_T = 0.0

    # Loop to create each trapezoid
    for i in range(number_intervals):
        x_value = a + (i * interval_size)
        a_plus_b = function(x_value + interval_size) + function(x_value)
        I_T += 0.5 * interval_size * a_plus_b

    # Return our running total result
    return I_T

### Solution:
# for (i, number_intervals) in enumerate(interval_sizes_T):
   # area_scipy_trap = si.trapz(f(np.linspace(0, np.pi, number_intervals+1)), 
   #                            np.linspace(0, np.pi, number_intervals+1))
    "linspace --> array of evenly spaced numbers between 0 and np.pi"
    "takes array of y-values (integration func) and array of x-values"

print("The exact area found by direct integration = 2")
trap_no = []
errors_trap = []

for i in (1, 2, 4, 8, 16, 32, 100, 1000):
    area = trapezoidal_rule(0, 1, func, i)
    error = abs(area-true_area)
    trap_no.append(i)
    errors_trap.append(error)
    print("Area %d trapezoid(s) = %g (error=%g)"%(i, area, abs(area-true_area)))

# adding to log-log plot
ax1.loglog(trap_no, errors_trap, 'b.-', label = 'trapezoidal')
ax1.legend(loc='best', fontsize=14)


## Error: is equal to M if it is exact for all polynomials of degree up to and 
# including M, but not exact for some polynomial of degree M+1
# ie if it is exact for polynomial up to x^3 (constant, x^2 and x^3) --> degree of precision = 3, error order of 4
# midpt and trapz both have deg of precision = 1 and order of error = 2

# %%
### Simpson's rule
# takes expected error from midpt and trap (error midpt = 2 error trap)
# combines both eqns and rearranges to improve accuracy
# alternative derivation fits quadratic Lagrange interpolating a and b with vertex = midpoint
# evaluating at midpt --> doubles interval of si.simps

# integrates up to cubic --> passes 4 points hence error is order of 4

# Scipy function takes in discrete data points and fits polynomial acorss interval"
# to get same values --> double no of intervals
# passing it through 'number_intervals' no of data pts --> '2*number_intervals + 1'
import numpy as np
import matplotlib.pyplot as plt
import scipy as si

def f(x):
    return np.sin(x)

print("The exact area found by direct integration = 2")
interval_sizes_S = [1, 2, 4, 8, 16, 32, 100, 1000]
errors_S = np.zeros_like(interval_sizes_S, dtype='float64')
areas_S = np.zeros_like(interval_sizes_S, dtype='float64')


for (i, number_intervals) in enumerate(interval_sizes_S):
    area_scipy_simpson = si.simps(f(np.linspace(0, np.pi, 2*number_intervals + 1)),
                   np.linspace(0, np.pi, 2*number_intervals + 1))
    print('{0:.16f}, {1:.16e}'.format(area_scipy_simpson, abs(area_scipy_simpson - areas_S[i])))

#%%
## Ex 3.3 Implementing Simpsons' rule on sin
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return x ** 3

def simpsons_rule(a, b, func, number_intervals=10):
    # This is different to the function/implementation available with SciPy  
    # where discrete data only is passed to the function. 
    
    # Bear this in mind when comparing results - there will be a factor of two
    # n the definition of "n" we need to be careful about!
    interval_size = (b - a)/number_intervals

    assert interval_size > 0
    assert type(number_intervals) == int

    I_S = 0.0

    for i in range(number_intervals):
        x = a + (i * interval_size)
        x_h = x + interval_size
        c = x + (interval_size / 2)
        I_S += ((x_h-x) / 6) * (func(x) + 4*func(c) + func(x_h))


    return I_S

true_area = 1/4
print("The area found by direct integration = %g" % true_area)

for i in (1, 2, 10, 100, 1000):
    area = simpsons_rule(0, 1, func, i)
    print("Area %d rectangle(s) = %g (error=%g)"%(i, area, abs(area - true_area)))

# %%
### Composite Simpson's Rule
# Simpsons rule but groups summations for productivity
# sci.interpolate.simps --> requires n to be EVEN 
# Assumes only use data given instead of additional midpoint evaluations
# Takes in discrete data points and fits polynomial acorss interval

# n = 2 --> same as original formula with midpoint loc c

#%%
### Weddle's rule (extrapolated simpsons rule)
# takes expected error from orig simp and 2*simp (error simp = 16 error 2*simp)
# takes 2 diff values for interval size and extrapolates to improve accuracy
# example of Romberg integration-- using different interval sizes to improve accuracy

# see jupyter notebook for full solution