# Author: Joseph Pate
# Class: Ast 4994w
# Date: 30/10/2017
'''
This program implements the solution to the wave equation presented in Zumbusch (2009)

The problem is posed as:

u_tt = u_xx \in \Omega \times [0,T]
u(0,x) = 0
u_t(0,x) = 2

The numerical method is as follows.

To facilitate defining the bilinear form, let ()dt be integration wrt. time 
()dx integration wrt. x and () be integration over the entire domain.

The bilinear form is

b(u,v) = (u_t, v)dx - (u_t(0), v(0))dx + (u_t, v_t) + (u_x, v_x) - (avg(u_x), jump(v))dt
		-(jump(u), avg(v_x))dt + 0.5/h *(jump(u), jump(v))dt

The linear form is
F(v) = (f,v)

Then the numerical scheme is find u \in DG such that

b(u,v) = F(v) \forall v \in DG
'''
from dolfin import *
import sys

# define boundary function
def boundary(x, on_boundary):
	return on_boundary and near(x[0], 0.0, 1e-14)

# # define jump function
# def myjump(v, n, tol=1E-12):
# 	j = v('+')[0]*n('+')[0] + v('-')[0]*n('-')[0]
# 	return conditional(abs(n('-')[0]) > tol, j/n('-')[0], j)

# create mesh
mesh = RectangleMesh(Point(0,0),Point(10,1),100,10)

# create function space
DG = FiniteElement('DP', triangle, 1)
V = FunctionSpace(mesh, DG)

# create test and trial functions
u = TestFunction(V)
v = TrialFunction(V)

# define DG tools
h = CellSize(mesh)
n = FacetNormal(mesh)
h_avg = avg(h)
alpha = 0.5

# create initial conditions
ut0 = Expression('2.0', degree=1)

# define bilinear form
b = u.dx(0)*v*dx(1) - (ut0*v)*dx(1) + (u.dx(0)*v.dx(0))*dx + (u.dx(1)*v.dx(1))*dx - (avg(u.dx(1))*jump(v))*dx(0)\
		-(jump(u)*avg(v.dx(1)))*dx(0) + 0.5/h_avg*(jump(u)*jump(v))*dx(0)

# define linear functional
F = Constant(0.0)*v*dx

# create boundary conditions
bc1 = DirichletBC(V, 0.0, boundary)

# solve linear system
U = Function(V)
solve(b == F, U, bc1)

# output solution
File('zumbusch.pvd') << U