# Author: Joseph Pate
# Class: Ast 4994w
# Date: 30/10/2017
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
'''

from dolfin import *
import sys

# create boundary function
def boundary(x, on_boundary):
	return on_boundary and (near(x[1], 0.0, 1e-14) or near(x[1], 1.0, 1e-14))

# create initial conditions
def initial(x, on_boundary)
	return on_boundary and near(x[0], 0.0, 1-e12)

# create mesh
mesh = RectangleMesh(Point(0,0),Point(10,1),100,10)

# create function spaces and fin
deg = 3
PCG = FiniteElement('P', triangle, deg)
PDG = FiniteElement('dP', triangle, deg-1)
CG_CG_elem = MixedElement([PCG, PCG])
CG_DG_elem = MixedElement([PCG, PDG])
V_CG = FunctionSpace(mesh, CG_CG_elem)
V_DG = FunctionSpace(mesh, CG_DG_elem)
V_out = FunctionSpace(mesh, PCG)

# create test and trial functions
u,v = TrialFunctions(V_CG)
w,y = TestFunctions(V_DG)

# create bilinear forms
b1 = (u.dx(0) - v)*w*dx
b2 = v.dx(0)*y*dx + u.dx(1)*y.dx(1)*dx
B = b1+b2

# create linear Functional
f = Constant(0.0)*y*dx + Constant(0.0)*w*dx

# impose boundary conditions
bc1 = DirichletBC(CG.sub(0), 0.0, boundary)
bc2 = DirichletBC(CG.sub(1), 20.0, boundary)

# solve linear system
U = Function(V_CG)
solve(B == f, U, [bc1, bc2])

# get u from U
u,v = U.split()

# output solution
u_out.rename('u', 'f_p_1+1')
if('plot' in sys.argv):
		plot(u)
		interactive()

else:
	file_u = File('f_p.pvd')
	file_u << u