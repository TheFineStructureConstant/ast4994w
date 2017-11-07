'''
This program solves the 1+1 spcetime wave equation using the method described in Anderson and Kimn (2007)
Let \Omega = [0,80]x[-5,5]
The problem is posed as:
v_t - \Delta u = f
-u_t + v = 0
u(0,x) = exp(-(x-1)^2)
u_t(0,x) = 0

with f = -2.0*cos(t)*exp(-(x-cos(t)^2)*(2.0*cos^3(t))-4.0*x*cos^2(t)+2.0*x^2*cos(t)-2.0*cos(t)+x)

The exact solution is u(x,t) = exp(-(x-cos(t))^2).

The numerical scheme is as follows.

Find u,v \in CG_1 such that
(v_t,w) + (grad(u) \dot grad(w))  = f*w
(-u_t,y) + (v,y) = 0
for all w,y \in CG_1


'''
from dolfin import *
import sys

# create boundary function
def boundary(x, on_boundary):
	return on_boundary and near(x[0], 0.0, 1e-14)

def make_waves(n):
	# create mesh
	mesh = RectangleMesh(Point(0,-5),Point(80,5),n*160,n*20)

	# create function space
	P1 = FiniteElement('P', triangle, 1)
	element = MixedElement([P1, P1])
	CG = FunctionSpace(mesh, element)
	V = FunctionSpace(mesh, P1)

	# create test and trial functions
	u,v = TrialFunctions(CG)
	w,y = TestFunctions(CG)

	# define bilinear forms
	b1 = v.dx(0)*w*dx + u.dx(1)*w.dx(1)*dx
	b2 = u.dx(0)*y*dx - v*y*dx
	B = b1+b2

	# define linear functional
	f = Expression('-2.0*cos(x[0])*exp(-pow(x[1]-cos(x[0]),2.0))*(2.0*pow(cos(x[0]),3.0)-4.0*x[1]*pow(cos(x[0]),2.0)+2.0*x[1]*x[1]*cos(x[0])-2.0*cos(x[0])+x[1])', degree=5)
	L = Constant(0.0)*y*dx + f*w*dx

	# create boundary data
	u0 = Expression('exp(-pow(x[1]-1.0, 2.0))', degree=5)
	u1 = Expression('0.0', degree=5)

	# impose boundary conditions
	bc1 = DirichletBC(CG.sub(0), u0, boundary)
	bc2 = DirichletBC(CG.sub(1), u1, boundary)

	# solve linear system
	U = Function(CG)
	solve(B == L, U, [bc1,bc2])

	# define exact solution
	uexp = Expression('exp(-pow((x[1] - cos(x[0])),2))', degree=5)
	uproj = project(uexp, V)

	# get u from U
	uh,vh = U.split()
	uhproj = project(uh, V)

	# calculate error in L 2 norm (for now)
	e = uproj - uhproj
	error = assemble(e**2*dx)

	# output solution
	uh.rename('u', 'a&k')
	file_uh = File('ak_2/a_k_t2_'+str(n)+'.pvd')
	file_uh << uh
	return error

# create list to hold error values
errors = []
for i in range(1,9):
	errors.append(make_waves(i))

print errors