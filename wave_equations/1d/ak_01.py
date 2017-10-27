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

# create boundary condition functions
# define initpoint(1) as the counting measure on the singleton set {0}
class point0(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 0.0)

class point1(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 1.0)

class point2(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 0.0)


# get number of grid points
# n = int(sys.argv[1])
u_t0 = int(sys.argv[1])

# create mesh
mesh = UnitIntervalMesh(10)

# create function space
P1 = FiniteElement('P', interval, 1)
R = FiniteElement('R', interval, 0)
element = MixedElement([P1, P1, R, R])
CG = FunctionSpace(mesh, element)

# create test and trial functions
u,w,a,c = TrialFunctions(CG)
v,y,b,d = TestFunctions(CG)

# setup boundaries on mesh
mask = MeshFunction('size_t', mesh, 0)
mask.set_all(0)
point0().mark(mask, 0)
point1().mark(mask, 1)
point2().mark(mask, u_t0)
initpoint = ds(subdomain_data=mask)

# create bilinear forms
b1 = (u.dx(0)-w)*v*dx + u*d*initpoint(0) + w*b*initpoint(2) + v*c*initpoint(1)
b2 = (w.dx(0)-u)*y*dx + u*d*initpoint(0) + w*b*initpoint(2) + y*a*initpoint(1)
L = b1+b2

# create linear functional
F = Constant(0.0)*v*dx

# solve variational problem
sol = Function(CG)
solve(L == F , sol)

u,w,a,c = split(sol)

fileu = File('Anderson-Kimn-u-0.pvd')
filew = File('Anderson-Kimn-w-0.pvd')

fileu << u
filew << w