# Author: Joseph Pate
# Class: Ast 4994w
# Date: 11/11/2017
'''
This program implements the numerical scheme shown in French Peterson (1996)
The scheme is a system of equations which are first order in time and space 
using CG/CG spaces for the trial functions and CG/DG spaces for the test functions

The problem is posed as
u_t - v = 0		   in \Omega \times [0,T]
v_t - \Delta u = f in \Omega \times [0,T]
u,v = 0 		   on \partial \Omega \times [0,T]
u(0,x) = 0
u_t(0,x) = 2

f is taken to be 0 for our initial test.

The numerical scheme is:

Find u,v in the space S_{p,q}, (space and time degree polynomials, respectively)
such that
(u_t - v, w) = 0 	\forall w \in S_p \cross P_{q-1}
(v_t, y)+(grad(u), grad(y)) = f forall y \in S_p \cross P_{q-1}

although what is actually implemented for the time being is a time stepping algorithm solving 
the wave equation as a system of coupled equations.
'''

from dolfin import *
import sys

# create mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(10.0, 1.0), 500, 50)

# create time step
dt = 0.5
t_f = 10
steps = int(t_f/dt)

# create function spaces
P1 = FiniteElement('P', triangle, 1)
element = MixedElement([P1, P1])
V = FunctionSpace(mesh, element)

# create test and trial functions
Y = TrialFunction(V)
Z = TestFunction(V)

# extract test and trial functions
u,v = split(Y)
w,y = split(Z)

# create initial conditions
u0 = Expression(('0.0', '2.0'), degree=1)
u_n = project(u0, V)
u_1,v_1 = split(u_n)

# define bilinear forms
b1 = ((u - u_1)/dt)*w*dx - v*w*dx
b2 = ((v - v_1)/dt)*y*dx + inner(grad(u),grad(y))*dx
B = b1 + b2

# define linear functional
F = Constant(0.0)*w*dx + Constant(0.0)*y*dx

# create ouput file
u_file = File('fp/wave.pvd')

# loop over time from t = 0 to t= 10
t = 0
for i in range(steps):
	# increment time
	t += dt
	print 'time: ', t

	# solve linear system
	U = Function(V)
	solve(B == F, U)

	# output solution at current time
	u_sol, v_sol = U.split()
	u_file << (u_sol, t)

	# increment space steps
	u_n.assign(U)

