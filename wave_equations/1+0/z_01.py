'''
This program implements the solution to the wave equation in 1D presented in Zumbusch (2009)
The method is second order in time using DG elements in time and space

The problem is:
u_tt = au
u(0) = u_0 (=0)
u_t(0) = u_1 (=0)

We let a = 1
'''



# imports
from dolfin import *
import sys

# create boundary condition function
def boundary(x, on_boundary):
	return on_boundary and near(x[0], 0, 1e-14)


# get number of grid points
points = int(sys.argv[1])

# create mesh
mesh = UnitIntervalMesh(points)

# create function space
P1 = FiniteElement('DP', interval, 1)
DG = FunctionSpace(mesh, P1)

# create test and trial functions
u = TrialFunction(DG)
v = TestFunction(DG)

# set up DG method tools
# n = FacetNormal(mesh)
h = CellSize(mesh)
f = Function(DG)
f('+')
grad(f)('+')

# create bilinear form
b = dot(grad(u), grad(v))*dx - avg(u.dx(0))*jump(v)*dS - avg(v.dx(0))*jump(u)*dS + jump(u)*jump(v)*dS

# create linear form
L = u*v*dx

# create boundary conditions
bc = DirichletBC(DG, 0.0, boundary)

# solve variational problem
sol = Function(DG)
solve(b==L, sol, bc)

# output solution
file = File('z_dg-0.pvd')
file << sol
