'''
This program solves the 1+1 spcetime wave equation using the method described in Anderson and Kimn (2007)

The problem is posed as:
v_t - \Delta u = 0
-u_t + v = 0
u(0,x) = 0
u_t(0,x) = 2 

The numerical scheme is as follows.

Find u,v \in CG_1 such that
(v_t,w) + (grad(u) \dot grad(w))  = 0
(-u_t,y) + (v,y) = 0
for all w,y \in CG_1


'''
from dolfin import *
import sys

# create boundary function
def boundary(x, on_boundary):
	return on_boundary and near(x[0], 0.0, 1e-14)


# create mesh
mesh = RectangleMesh(Point(0,0),Point(10,1),100,10)

# create function space
P1 = FiniteElement('P', triangle, 1)
element = MixedElement([P1, P1])
CG = FunctionSpace(mesh, element)

# create test and trial functions
u,v = TrialFunctions(CG)
w,y = TestFunctions(CG)

# define bilinear forms
b1 = v.dx(0)*w*dx + u.dx(1)*w.dx(1)*dx
b2 = u.dx(0)*y*dx - v*y*dx
B = b1+b2

# define linear functional
L = Constant(0.0)*y*dx + Constant(0.0)*w*dx

# impose boundary conditions
bc1 = DirichletBC(CG.sub(0), 0.0, boundary)
bc2 = DirichletBC(CG.sub(1), 20.0, boundary)

# solve linear system
U = Function(CG)
solve(B == L, U, [bc1,bc2])

# get u from U
u,v = U.split()

# output solution
u_out.rename('u', 'a&k')
if('plot' in sys.argv):
	plot(u)
	interactive()

else:
	file_u = File('a_k.pvd')
	file_u << u







