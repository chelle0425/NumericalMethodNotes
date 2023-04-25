""" 
- Learn about finite difference approximations to derivatives
    - Forward difference and Taylor series
- Be able to implement forward and central difference methods
- Calculate higher-order derivatives
- Solve simple ODEs using the finite difference method

> Finite forward and backward difference
> Taylors series about x0
> Central differencing
> Forward euler and Heun (refines estimation from euler)
"""
#%%
### Finite difference (single spatial dimension)
# estimates derivative of functions

# forward difference (forward = delta x > 0)
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6))
ax1 = plt.subplot(111)
ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(-0.1, 1.1)
ax1.set_title('Forward difference example', fontsize=16)
ax1.set_xlabel('$x$', fontsize=16)
ax1.set_ylabel('$f(x)$', fontsize=16)
# define our x for plotting purposes
x = np.linspace(0, 1, 1000)

# define our example function and its exact derivative

def f(x):
    return x**2

def df(x):
    return 2 * x

# plot the 'exact' solution
ax1.plot(x, f(x), 'k')

# choose and plot two x locations to take the difference between
x0 = 0.35
dx = 0.5
x1 = x0 + dx
# plot a line representing the discrete derivative
ax1.plot([x0, x1], [f(x0), f(x1)], 'r', label = 'Forward diff. approx. deriv.')
"[x0, x1] are start and end point x values"
"[f(x0), f(x1)] are tart and end point y values"

# plot a line representing the exact derivative (given by function f(.)) at x=x0
h = dx/2
"length of blue line = dx; h * df(x0) = change in y due to h"
ax1.plot([x0 - h, x0 + h], [f(x0) - (h * df(x0)), f(x0) + (h * df(x0))], 'b', label = 'Exact derivative')

# add some axes labels and lines etc
ax1.set_xticks((x0, x1)) 
"x1 = x0 + dx"
# 'g:' lines are for green dotted lines
ax1.set_xticklabels(('$x_0$', '$x_0+\Delta x$'), fontsize=16)
ax1.plot([x0, x0], [-0.1, f(x0)], 'g:')
ax1.plot([x1, x1], [-0.1, f(x1)], 'g:')
ax1.set_yticks((f(x0), f(x1)))
ax1.set_yticklabels(('$f(x_0)$', '$f(x_0+\Delta x)$'), fontsize=16)
ax1.plot([-0.1, x0], [f(x0), f(x0)], 'g:')
ax1.plot([-0.1, x1], [f(x1), f(x1)], 'g:')

ax1.legend(loc='best', fontsize=14)

# %%
### Taylor series 
# f (x0 + h) = f(x0) + hf'(x0) + ...
# about point x0 (function on RHS are evaluated at this point)
# more terms = better approximation valid a larger distance from x0

## Ex 2.1 forward difference to compute an appriximation to f'(2.36)
# f(2.36)=0.85866
# f(2.37)=0.86289
# You should get an answer of 0.423.

dx = 0.01
# f'(2.36) = (f(2.36 + 0.01) - f(2.36)) / 0.01
df = (0.86289 - 0.85866) / dx
print(df)

# %%
### Central difference 
# estimates derivative of functions but is second-order accurate
# takes 2 f(x) points and estimates in between
    # requires f (x + dx), f (x - dx) and dx
# halve h --> error drop by a factor of 4 instead of 2 (first-order forward/ backword differencing)

# 2 Taylor series from opposite directions (positive and negative x from x0)
# subtract second equation from first and rearrange for f'(x0)
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6))
ax1 = plt.subplot(111)
ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(-0.1, 1.1)

# ax.grid(True)
ax1.set_title('Central difference example', fontsize=16)
ax1.set_xlabel('$x$', fontsize=16)
ax1.set_ylabel('$f(x)$', fontsize=16)
# define our x for plotting purposes
x = np.linspace(0, 1, 1000)

# define our example function and its exact derivative

def f(x):
    return x**2

def df(x):
    return 2 * x

# plot the exact solution
ax1.plot(x, f(x), 'k')
# choose and plot two x locations to take the difference between
dx = 0.4
x0 = 0.5
xl = x0 - dx
xr = x0 + dx
# plot a line representing the discrete derivative
ax1.plot([xl, xr], [f(xl), f(xr)], 'r', label = 'Central diff. approx. deriv.')
# plot a line representing the exact derivative at x=x0
h = dx / 2
ax1.plot([x0 - h, x0 + h], [f(x0) - h * df(x0), f(x0) + h * df(x0)], 'b', label = 'Exact derivative')
# add some axes labels and lines etc
ax1.set_xticks((xl, x0, xr))
ax1.set_xticklabels(('$x_0-\Delta x$', '$x_0$', '$x_0+\Delta x$'), fontsize=16)
ax1.plot([xl, xl], [-0.1, f(xl)], 'k:')
ax1.plot([xr, xr], [-0.1, f(xr)], 'k:')
ax1.plot([x0, x0], [-0.1, f(x0)], 'k:')
ax1.set_yticks((f(xl), f(xr)))
ax1.set_yticklabels(('$f(x_0-\Delta x)$', '$f(x_0+\Delta x)$'), fontsize=16)
ax1.plot([-0.1, xl], [f(xl), f(xl)], 'k:')
ax1.plot([-0.1, xr], [f(xr), f(xr)], 'k:')
ax1.legend(loc='best', fontsize=14)

# %%
## Ex 2.2 central differencing

dx = 0.1
# central differencin about 0.2 with f(0.1) and f(0.3)
# df = [f(0.3) - f(0.1)] / 2 * dx
df = (0.192916 - 0.078348) / 2 * dx

print(df)

# %%
## Function to perform numerical differentiation
# returns the approximation of the derivative of Python f(x)

# apply to differentiate f(x) = e ** x at x = 0; f(x) = e ** (-2 * x) at x = 0;
# f (x) = cos(x) at x = 2pi; f(x) = ln(x) at x = 1

import numpy as np

dx = 0.01

def diff(f , x, dx = 1.0e-6):
    numerator = f(x + dx) - f(x - dx)
    derivative = numerator / (2.0 * dx)

    return derivative

# Differntiate f(x) = e ** x at x = 0
# estimating derivative of f(x) with f (x + dx) and f (x - dx)
x = 0
f = np.exp
derivative = diff(f , x, dx)
print("The approximate derivative of exp(x) at x = 0 is: %f. The error is %f."
      % (derivative, abs(derivative - 1)))

# Differentiate f(x) = e ** (-2 * x) at x = 0
x = 0
def g(x):
    return np.exp (-2 * x)

derivative = diff(g, x, dx)
print('The approximate derivative of exp(-2*x) at x = 0 is: {0:.5f}.  The error is {1:.5f}.'
        .format(derivative, abs(derivative - (-2.0))))

# Differentiate f (x) = cos(x) at x = 2pi
x = 2 * np.pi
f = np.cos

derivative = diff(f, x, dx)

print('The approximate derivative of cos(x) at x = 2*pi is: {0:.5f}.  The error is {1:.5f}.'
        .format(derivative, abs(derivative - 0)))

# Differentiate f(x) = ln(x) at x = 1
x = 1
f = np.log

derivative = diff(f, x, dx)

print('The approximate derivative of ln(x) at x = 0 is: {0:.5f}.  The error is {1:.5f}.'
        .format(derivative, abs(derivative - 1)))

# %%
## Ex 2.3 compute the derivative of sin(x) 
# using a) forward differencing and b) central differencing
# decreasing values of h (h = 1 start) dx = 0.01

import numpy as np
import matplotlib.pyplot as plt

def forward(f, x0, h):  # f = np.sin and x = 0.8
    fx0 = f(x0)
    fxph = f(x + h)
    df = (fxph - fx0) / h
    return df

def central(f, x0, h):
    fxph = f(x0 + h)
    fxnh = f(x0 - h)
    df = (fxph - fxnh) / (2*h) # ignore O(dx^2) terms
    return df

exact = np.cos(0.8)
print("the exact derivative of sin(0.8) =", exact)

# headers for the following error outputs
print('%20s%40s' % ('Forward differencing', 'Central differencing'))

# storing values for plotting
forwarrderr = []
centralerr = []
hstore = []

# initial h 
h = 1
x = 0.8
for i in range(10):
    # calculate values
    fd = forward(np.sin, x, h)  # note how f = np.sin and x = 0.8
    cd = central(np.sin, x, h)

    # calculate and store errors
    fd_err = abs(fd - exact)
    forwarrderr.append(fd_err)
    cd_err = abs(cd - exact)
    centralerr.append(cd_err)

    hstore.append(h)

    print('%g (error=%.2g)         %g (error=%.2g)' %
        (fd, abs(fd - exact), cd, abs(cd - exact))) 
        # space translates to actual table space, .2g is 2sig inc e notation

    h = h / 2 # half h for next iteration


# plotting setup

fig = plt.figure(figsize=(7, 5))
ax1 = plt.subplot(111)
ax1.set_title('Derivative of sin(x)')
ax1.set_xlabel('h', fontsize=16)
ax1.set_ylabel('Error', fontsize=16)
# define our x for plotting purposes
x = np.linspace(0, 1, 1000)

# plotting
ax1.plot(hstore, forwarrderr, 'r.-', label = 'forward') # .- adds dots
ax1.plot(hstore, centralerr, 'b.-', label = 'central')

# add a figure title and legend
ax1.set_title('Forward diff and Central diff approx of d sin(0.8)/ dx', fontsize=16)
ax1.legend(loc='best', fontsize=14)

# %%
## Ex 2.4 compute the second derivative at x = 1 with data
import numpy as np
import matplotlib.pyplot as plt


def forward(f, x0, h):  # f = np.sin and x = 0.8
    fx0 = f(x0)
    fxph = f(x + h)
    df = (fxph - fx0) / h
    return df

def backward(f, x0, h):
    fx0 = f(x0)
    fxph = f(x - h)
    df = (fx0 - fxph) / h
    return df


# (forward(f, x0, h) - backward(f, x0, h)) / h

dx = 0.08
ddf = (0.339596 - 2 * 0.367879 + 0.398519)/ (dx*dx)
print(ddf)

#%%
### Non-central differencing and differentiation by polynomial fit
# good for derv at or near boundaries

# ODEs with Forward (Explicit) Euler
# explicit-- all info to compute sol (RHS)
# calculates value of one time step forward from last known value The error is therefore the local truncation error. If we actually wish to know the value at some fixed time 
## actual value-- (T-t0)/h steps

## Ex 2.5 
t_max = 10

def euler(f,u0,t0,t_max,dt):
    u=u0; t=t0
    # these lists will store all solution values 
    # and associated time levels for later plotting
    u_all=[u0]; t_all=[t0]
    
    '''u0(t0) has to be functions'''

    while t < t_max:
        # u(0.05) = u(0) + 0,05 u'(0)
        t = t + dt
        u = u + dt * f(u, t)

        u_all.append(u)
        t_all.append(t)
    
    return(u_all,t_all)


def f(u,t):
    val = u
    return val


(u_all, t_all) = euler(f,1.0,0.0,10.0,0.1)


# plotting setup

fig = plt.figure(figsize=(7, 5))
ax1 = plt.subplot(111)

ax1.set_xlabel('t', fontsize=16)
ax1.set_ylabel('x(t)', fontsize=16)
# define our x for plotting purposes
x = np.linspace(0, 1, 1000)

# plotting
ax1.plot(t_all, u_all, 'b')

# %%
### Heun's method
# uses derivative info at BOTH the start and end of interval
# requires u (t + dt) --> first guess with Euler's method
# second order mehtod

## Ex 2.6
t_max = 10

def euler(f,u0,t0,t_max,dt):
    u=u0; t=t0
    # these lists will store all solution values 
    # and associated time levels for later plotting
    u_all=[u0]; t_all=[t0]
    
    '''u0(t0) has to be functions'''

    while t < t_max:
        # u(0.05) = u(0) + 0,05 u'(0)
        u = u + dt * f(u, t)
        u_all.append(u)
        t = t + dt
        t_all.append(t)
    
    return(u_all,t_all)

def heun(f, u0, t0, t_max, dt):
    u=u0; t=t0
    # these lists will store all solution values 
    # and associated time levels for later plotting
    u_all=[u0]; t_all=[t0]

    while t < t_max:
        # u(0.05) = u(0) + 0,05 u'(0)
        u_euler = u + dt * f(u, t)
        t_new = t + dt
        u_heun = u + (0.5 * dt * (f(u, t) + f(u_euler, t_new)))

        u_all.append(u_heun)

        u = u_heun
        t = t_new
        t_all.append(t)
    
    return(u_all,t_all)


def f(u,t):
    val = u
    return val


(u_euall, t_euall) = euler(f,1.0,0.0,10.0,0.4)

(u_heuall, t_heuall) = heun(f,1.0,0.0,10.0,0.4)


# plotting setup

fig = plt.figure(figsize=(7, 5))
ax1 = plt.subplot(111)

ax1.set_xlabel('t', fontsize=16)
ax1.set_ylabel('x(t)', fontsize=16)
# define our x for plotting purposes
x = np.linspace(0, 1, 1000)

# plotting
ax1.plot(t_euall, u_euall, 'b', label = 'euler')
ax1.plot(t_heuall, u_heuall, 'r', label = 'heun')
ax1.plot(t_euall, np.exp(t_euall), "k", label = 'exact')
ax1.legend(loc='best', fontsize=14)

print(u_heuall, t_heuall)
