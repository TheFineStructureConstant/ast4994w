from dolfin import *
import sys
"""
Solve   u' = f on (0, 1),  u(0) = u0   using CG_r trial functions with
the initial condition imposed, and CG_r test functions with a terminal
condition imposed.  As weak form we take

  Find u in CG_r, c in Real s.t.

  (u', v) + u(0) d + c v(1) = (f, v)  for all v in CG_r, d in Real.
"""
# set left boundary
u0 = int(sys.argv[1])
f = Expression("cos(20*x[0])", degree=4)
#uex = Expression(".05*sin(20*x[0])", degree=4)
r = 3
n = 50
mesh = UnitIntervalMesh(n)
DG = FiniteElement('DG', mesh.ufl_cell(), r)
R = FiniteElement('R', mesh.ufl_cell(), 0)
X = FunctionSpace(mesh, DG*R)
u, c = TrialFunctions(X)
v, d = TestFunctions(X)

# define initpoint(1) as the counting measure on the singleton set {0}
class point0(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 0.0)

class point1(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.)

mask = MeshFunction('size_t', mesh, 0)
mask.set_all(0)
point0().mark(mask, u0)
point1().mark(mask, 1)
initpoint = ds(subdomain_data=mask)

# define bilinear form and linear functional
a = u.dx(0) * v * dx + u * d * initpoint(0) + c * v * initpoint(1)
L = f * v * dx
uc = Function(X)
solve(a == L, uc)
u, c = uc.split()
plotmesh = UnitIntervalMesh(n*100)
V0 = FunctionSpace(plotmesh, 'DG', 1)
plot(interpolate(u, V0))