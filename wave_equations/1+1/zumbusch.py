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

we define the bilinear form as

b(u,v) = (u_t,v_t) + (grad(u), grad(v)) - (jump(u\cdot n), avg(v))dS 
- (avg(u), jump(v \cdot n))dS + a(avg(u),avg(v))dS 

Then the numerical scheme is find u \in DG such that

b(u,v) = f ( =0) \forall v \in DG
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

# define bilinear form
b = u.dx(0)*v.dx(0)*dx - inner(avg(v.dx(1)), jump(u,n))*dS - inner(avg(u.dx(0)), jump(v,n))*dS + (alpha/h_avg)*inner(jump(u,n), jump(v,n))*dS

# define linear functional
f = Constant(0.0)*v*dx

# create boundary conditions