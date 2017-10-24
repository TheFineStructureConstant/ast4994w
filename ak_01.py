'''
This program implements the solution to the wave equation in 1D presented in Anderson-Kimn (2007)
The method is first order in time using CG elements in time and space

The problem is:
u_tt = au
u(0) = u_0 (=0)
u_t(0) = u_1 (=0)

We let a = 1.

The numerical scheme is: find u,w \in CG_1 such that

(u_t,v) = (w,v)		for all w \in CG_1
(w_t,y) = (au,y)	for all y \in CG_1

with zero boundary data at x = 0 and 
'''

# imports
from dolfin import *
# import matplotlib.pyplot as plt
import sys

# create boundary condition function
def boundary(x, on_boundary):
	return on_boundary and near(x[0], 0, 1e-14)

# get number of grid points
n = int(sys.argv[1])

# create mesh
mesh = UnitIntervalMesh(n)

# create function space
P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1])
CG = FunctionSpace(mesh, element)

# create test and trial functions
u,w= TrialFunction(CG)
v,y = TestFunctions(CG)

# create bilinear forms
b1 = (u.dx(0)-w)*v*dx
b2 = (w.dx(0)-u)*y*dx
B = b1+b2

# define boundaries
bc = DirichletBC(CG.sub(1), 0.0, boundary)

# solve variational problem
sol = Function(CG)
solve(B == 0, sol, bc)

u,w = split(sol)

fileu = File('Anderson-Kimn-u-0.pvd')
filew = File('Anderson-Kimn-w-0.pvd')

fileu << u
filew << w
