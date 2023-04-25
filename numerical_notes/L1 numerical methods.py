"""if you cant import numpy do command P >Python: Select Interpreter"""


""" 
- Learn about standard methods to approximate discrete data points.
- Understand the differences between interpolation and curve-fitting (e.g. of noisy data).
- Implement methods to compute simple polynomial interpolation in 1D.

> basic plots
> Lagrange polynomial
> Newton polunomial (alternative to Lagrange)
> Curve fitting with least squares
> Extrapolaion
> Linear fit
"""

#%%
## Basic plot + y = mx + c line of best fit
# some imports we will make at the start of every notebook
# later notebooks may add to this with specific SciPy modules

import numpy as np
import matplotlib.pyplot as plt
from sympy import E

# Invent some raw data - we will use the notation (xi,yi) for the
# given data, where xi and yi are of length N+1 (N=len(xi)-1)
xi = np.array([0.5, 2.0, 4.0, 5.0, 7.0, 9.0])
yi = np.array([0.5, 0.4, 0.3, 0.1, 0.9, 0.8])

# We will want to overlay a plot of the raw data a few times below so 
# let's do this via a function that we can call repeatedly
# [Note that I've been a bit lazy in later lectures and really should
# do this sort of thing more often to make code easier to read - apologies]
def plot_raw_data(xi, yi, ax):
    """plot x vs y on axes ax, 
    add axes labels and turn on grid
    """
    ax.plot(xi, yi, 'ko', label='raw data')
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.grid(True)


# set up figure
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)

# For clarity we are going to add a small margin to all the plots.
ax1.margins(0.1)

# plot the raw data
plot_raw_data(xi, yi, ax1)

# add a figure title
ax1.set_title('Our simple raw data', fontsize=16)

# Add a legend
ax1.legend(loc='best', fontsize=14)
# loc='best' means we let matplotlib decide the best place for the
# legend to go.  For other options see 
#  https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html


# Fit a polynomial of degree 1, i.e. a straight line, to our (xi, yi) data from above
# we'll explain what's going on here later in this lecture
degree = 1
poly_coeffs = np.polyfit(xi, yi, degree)
print('poly_coeffs: ',poly_coeffs)

# use poly1d to turn the coeffs into a function, p1, we can evaluate
p1 = np.poly1d(poly_coeffs)

# set up figure
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)
ax1.margins(0.1)

# Plot the linear fit - define 100 evenly spaced points (x) covering our
# x extent and plot our linear polynomial evaluated at these points (p1(x))
# of course 100 is overkill for this linear example
x = np.linspace(0., 9.5, 100)
# NB. the 'linspace' function from numpy returns evenly spaced numbers 
# over a specified interval. It takes three arguments; the first two 
# are the bounds on the range of values, and the third is the total 
# number of values we want.
# See the docs (i.e. np.linspace?) for additional options arguments

ax1.plot(x, p1(x), 'b', label=r'$y = {0:.4f}x+{1:.4f}$'.format(poly_coeffs[0], poly_coeffs[1]))

# Overlay raw data
plot_raw_data(xi, yi, ax1)

# Add a legend
ax1.legend(loc='best', fontsize=14)

# add a figure title
ax1.set_title('Raw data and the corresponding linear best fit line', fontsize=16);

#  plt.show() 
#### This is VERY important as this replaces %matplotlib incline ####

# estimates y of desired x based on polynomial
coefficients = np.polyfit(x, y, degree) 
y_value = np.polyval(coefficients, x_value)
#%%
## Lagrange Polynomial
# imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si


# data, where xi and yi are of length N+1 (N=len(xi)-1)
xi = np.array([0.5, 2.0, 4.0, 5.0, 7.0, 9.0])
yi = np.array([0.5, 0.4, 0.3, 0.1, 0.9, 0.8])

def plot_raw_data(xi, yi, ax):
    """plot x vs y on axes ax, 
    add axes labels and turn on grid
    """
    ax.plot(xi, yi, 'ko', label='raw data')
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.grid(True)

# set up figure (do this everytime)
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)
ax1.margins(0.1) # For clarity we are going to add a small margin to all the plots


###
# Create the Lagrange polynomial for the given points.
lp = si.lagrange(xi, yi)
# Evaluate this function at a high resolution (100 points here)
x = np.linspace(0.4, 9.1, 100)
# the first two arguments are the bounds on the range of values, and the third is the total 
# number of values we want (100 evenly spaced points (x) covering range)

# actually plot (x,y)=(x,lp(x)) on the axes with the label ax1
ax1.plot(x, lp(x), 'b', label='Lagrange interpolating polynomial')


# Overlay raw data on the same axes
plot_raw_data(xi, yi, ax1)


# add a figure title
ax1.set_title('Lagrange interpolating polynomial (SciPy)', fontsize=16)
# Add a legend
ax1.legend(loc='best', fontsize=14)
# %%
import numpy as np
import matplotlib.pyplot as plt

# set up our figs for plotting - we want three subplots arranged in a 1x3 grid
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
# add some padding otherwise axes the labels can overlap with the next subplot
fig.tight_layout(w_pad=4) 

# data
def y_func(x):
    return x**3

# as we will plot our approximation several times let's write a small function to do this
def plot_approximation(f, xi, ax):
    # Relatively fine x points for plotting our functions
    x = np.linspace(0.5, 3.5, 100)
    # Plot the original function
    ax.plot(x, f(x), 'k', label = 'Original function')

    # construct and plot the Lagrange polynomial
    lp = si.lagrange(xi, f(xi))
    # evaluate and plot the Lagrange polynomial at the x points
    ax.plot(x, lp(x), 'b', label = 'Lagrange poly. interpolant')

    # shade the region between the two to emphasise the difference
    ax.fill_between(x, f(x), lp(x))
    
    # add some axis labels
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$f(x), \; P_N(x)$', fontsize=14)

    # and add on top the interpolation points
    ax.plot(xi, f(xi), 'ko')
    
    # and a legend
    ax.legend(loc='best', fontsize=13)

# L0
plot_approximation(y_func, np.array([2., ]), ax1)
ax1.set_title('Approximating a cubic with a constant', fontsize=16)

# L1
plot_approximation(y_func, np.array([1., 3.]), ax2)
ax2.set_title('Approximating a cubic with a linear', fontsize=16)

# L0
plot_approximation(y_func, np.array([1., 2., 3.]), ax3)
ax3.set_title('Approximating a cubic with a quadratic', fontsize=16)


# %%
## Newton polynomial
import numpy as np
import matplotlib.pyplot as plt

# consider the above example data again
xi = np.array([0.5, 2.0, 4.0, 5.0, 7.0, 9.0])
yi = np.array([0.5, 0.4, 0.3, 0.1, 0.9, 0.8])

def plot_raw_data(xi, yi, ax):
    """plot x vs y on axes ax, 
    add axes labels and turn on grid
    """
    ax.plot(xi, yi, 'ko', label='raw data')
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.grid(True)


def calculate_newton_coeffs(xi, yi):
    '''Evaluate the coefficients a_i recursively using Newton's method'''
    # initialise the array 'a' with yi, but take a copy to ensure we don't
    # overwrite our yi data!
    a = yi.copy()

    # we have N+1 data points, and so
    N = len(a) - 1

    # For each k, we compute Δ^k y_i from the a_i = Δ^(k-1) y_i 
    # of the previous iteration
    # We start our loop witk k=1 since we initially don't want to 
    # touch a_0 which is already equal to y_0
    for k in range(1, N+1):
        # but only for i>=k
        for i in range(k, N+1):
            a[i] = (a[i] - a[k-1])/(xi[i]-xi[k-1])
    return a

# Given the coefficients a, and the data locations xi,
# define a function to evaluate the Newton polynomial
# at locations given in the array x.
# NB. this is just an evaluation of the P_n(x) = ... formula
# given at the start of this section.

" i dunno why but"
def eval_newton_poly(a, xi, x):
    """ Function to evaluate the Newton polynomial
    at x, given the data point xi and the polynomial coeffs a
    """
    N = len(xi) - 1  # polynomial degree
    # recursively build up polynomial evaluated at x
    P = a[N]
    for k in range(1, N+1):
        P = a[N-k] + (x - xi[N-k])*P
    return P

# set up figure
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)

# For clarity we are going to add a small margin to all the plots.
ax1.margins(0.1)

# Evaluate the coefficients of the Newton polynomial
a = calculate_newton_coeffs(xi, yi)

# Evaluate the polynomial at high resolution and plot
x = np.linspace(0.4, 9.1, 100)
ax1.plot(x, eval_newton_poly(a, xi, x), 'b', label='Newton poly')

# plot the raw data
plot_raw_data(xi, yi, ax1)

# add a figure title
ax1.set_title('Our simple raw data', fontsize=16)

# Add a legend
ax1.legend(loc='best', fontsize=14)

# %%
## Curve fitting with least squares (np.polyfit)
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

# consider the above example data again
xi = np.array([0.5, 2.0, 4.0, 5.0, 7.0, 9.0])
yi = np.array([0.5, 0.4, 0.3, 0.1, 0.9, 0.8])

def plot_raw_data(xi, yi, ax):
    """plot x vs y on axes ax, 
    add axes labels and turn on grid
    """
    ax.plot(xi, yi, 'o', label='raw data')
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.grid(True)

# Calculate coefficients of polynomial degree 0 - ie a constant value.
poly_coeffs=np.polyfit(xi, yi, 0)

# Construct a polynomial function which we can use to evaluate for arbitrary x values.
# finds the poly coeff with least square differences
p0 = np.poly1d(poly_coeffs)

# Fit a polynomial degree 1 - ie a straight line.
poly_coeffs=np.polyfit(xi, yi, 1)
p1 = np.poly1d(poly_coeffs)

# Quadratic
poly_coeffs=np.polyfit(xi, yi, 2)
p2 = np.poly1d(poly_coeffs)

# Cubic
poly_coeffs=np.polyfit(xi, yi, 3)
p3 = np.poly1d(poly_coeffs)

# set up figure
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)
ax1.margins(0.1)

x = np.linspace(0.4, 9.1, 100)

ax1.plot(x, p0(x), 'k', label='Constant')
ax1.plot(x, p1(x), 'b', label='Linear')
ax1.plot(x, p2(x), 'r', label='Quadratic')
ax1.plot(x, p3(x), 'g', label='Cubic')

# Overlay raw data
plot_raw_data(xi, yi, ax1)


## Ex 1.3 function that evaluates squared error E

def least_squares_calc(xi, yi, f):
    # define P (value of polynomial function to data evaluated at point xi and yi)
    P = np.poly1d(np.polyfit(xi, yi, f))

    E = (P(xi) - yi)**2
    return E.sum()

'''def sqr_error(p, xi, yi):
    """"function to evaluate the sum of square of errors"""
    # first compute the square of the differences
    diff2 = (p(xi)-yi)**2
    # and return their sum
    return diff2.sum()
    
    print(sqr_error(p0, xi, yi))'''

print(least_squares_calc(xi, yi, 0))
print(least_squares_calc(xi, yi, 1))
print(least_squares_calc(xi, yi, 2))
print(least_squares_calc(xi, yi, 3))

## Ex 1.4
N = len(xi) - 1

# N-bic
poly_coeffs=np.polyfit(xi, yi, N)
pN = np.poly1d(poly_coeffs)

# polynomial plot
ax1.plot(x, pN(x), 'g', label='N-bic')

# Lagrange
# Create the Lagrange polynomial for the given points.
lp = si.lagrange(xi, yi)
ax1.plot(x, lp(x), 'r', label='Lagrange')

ax1.legend(loc='best', fontsize = 12)
ax1.set_title('Polynomial approximations of differing degree', fontsize=16)


# %%
## Ex 1.5 Exptrapolation
import numpy as np
import matplotlib.pyplot as plt

# consider the above example data again
xi = np.array([0.5, 2.0, 4.0, 5.0, 7.0, 9.0])
yi = np.array([0.5, 0.4, 0.3, 0.1, 0.9, 0.8])

def plot_raw_data(xi, yi, ax):
    """plot x vs y on axes ax, 
    add axes labels and turn on grid
    """
    ax.plot(xi, yi, 'o', label='raw data')
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.grid(True)


"change linespace to correct x range"
x1 = np.linspace(-2, 11, 100)
x2 = np.linspace(0., 9.5, 100)

## increasing degree of polynomial
# Let's set up some space to store all the polynomial coefficients
# there are some redundancies here, and we have assumed we will only 
# consider polynomials up to degree N
N = 6
# to store polynomial
poly_coeffs = np.zeros((N, N))

''': slices element from the list (numpy array)
start index is 0 and end index is i+1

this reiterates for all N degree polynomials'''



# set up figure (2 subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.tight_layout(w_pad=4) 

# plotting first dataset
plot_raw_data(xi, yi, ax1)

for i in range(N):
    p = np.poly1d(np.polyfit(xi, yi, i))
    ax1.plot(x1, p(x1), label='Degree %i' % i)

'''this BREAKS the graph: 
ax1 = fig.add_subplot(111)
ax1.margins(0.1)'''

ax1.legend(loc='best', fontsize = 12)
ax1.set_title('Extrapolation x = (-2, 11)', fontsize=16)



# plotting second dataset
plot_raw_data(xi, yi, ax2)

for i in range(N):
    p = np.poly1d(np.polyfit(xi, yi, i))
    ax2.plot(x2, p(x2), label='Degree %i' % i)


ax2.legend(loc='best', fontsize = 12)
ax2.set_title('Extrapolation x = (0, 9.5)', fontsize=16)



# %%
## Ex 1.6 fit linear best fit into dataset
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt

# dataset loading
dataset_raw  = loadtxt("Length-Width.txt", dtype="float")

# convert into x and y array
length_xi = []
width_yi = []

for i in dataset_raw:
    length_xi.append(i[0])
    width_yi.append(i[1])

xi = np.log10(np.array(length_xi))
yi = np.log10(np.array(width_yi))

def plot_raw_data(xi, yi, ax):
    """plot x vs y on axes ax, 
    add axes labels and turn on grid
    """
    ax.plot(xi, yi, 'o')
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.grid(True)


# using polyfit to fit a linear bestfit line
line = np.poly1d(np.polyfit(xi, yi, 1))

# plotting set up
# set up figure
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)
ax1.margins(0.1)

# plotting line
x = np.linspace(min(xi), max(xi), 100)
ax1.plot(x, line(x), 'b', label="log(y) = %s" %line)


# plot the raw data
plot_raw_data(xi, yi, ax1)

# add a figure title and legend
ax1.set_title('Submarine landslides in the North Atlantic basin', fontsize=16)
ax1.set_xlabel('log(length)', fontsize=16)
ax1.set_ylabel('log(width)', fontsize=16)
ax1.legend(loc='best', fontsize=14)

# %%
