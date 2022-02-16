# experiments for the master thesis
from fenics import *
import numpy as np

class annihilation_3d:
    def __init__(self, n=2**5, dt=0.0005, T= 0.1):
        self.name="annihilation 3D"
        self.dim = 3
        self.fine = n
        self.dt = dt
        self.T = T
        # - model parameters namely v_el, const_A, nu, mu_1, mu_4, mu_5, mu_6, lam
        self.param_dict= {"v_el":.25, "const_A":1.0, "nu":1.0,"mu_1":1.0, "mu_4": 1.0, "mu_5":1.0, "mu_6":1.0 , "lam":1.0}
        self.parameters = list(self.param_dict.values())
        self.mesh = BoxMesh(Point(-0.5, -0.5,-0.5), Point(0.5, 0.5,0.5),n,n,n)
        def boundary(x):
            return x[0] < (DOLFIN_EPS -0.5) or x[0] > (0.5 - DOLFIN_EPS) or x[1] < (DOLFIN_EPS -0.5) or x[1] > (0.5 - DOLFIN_EPS) or x[2] < (DOLFIN_EPS -0.5) or x[2] > (0.5 - DOLFIN_EPS)  
        self.boundary = boundary

        # - initial conditions
        # - define custom Expression
        # see https://fenicsproject.discourse.group/t/problem-defining-initial-conditions-expression-problem/626
        class d0_expr(UserExpression):
            def eval(self,values,x):
                if x[1]==0 and (x[0]==0.25 or x[0]==-0.25):
                    values[0]=0.0
                    values[1]=0.0
                    values[2]=1.0
                else:
                    values[0]=4.0*pow(x[0],2)+4*pow(x[1],2)-0.25
                    values[1]=2.0*x[1]
                    values[2]=0.0
                    tmp_abs = np.sqrt(values[0]**2+values[1]**2+values[2]**2)
                    values[0]= values[0] / tmp_abs
                    values[1]= values[1] / tmp_abs
                    values[2]= values[2] / tmp_abs
            def value_shape(self):
                return (3,)
        zero_expr = Expression(("0.0","0.0","0.0"), degree=2)
        self.ics = [zero_expr,Constant(0.0),d0_expr(),zero_expr]
        # boundary conditions
        self.bcs = [zero_expr, Constant(0.0), Expression(("4.0*pow(x[0],2)+4*pow(x[1],2)-0.25","2.0*x[1]","0"),degree=1),None]

        # - parameters namely, v_el, const_A

class annihilation_2d:
    # as experiment one but in 2D
    # as in Becker et. al
    def __init__(self, n=2**5, dt=0.0005, T=0.1):
        self.name="annihilation 2D"
        self.dim = 2
        self.fine = n
        self.dt = dt
        self.T = T
        # - model parameters namely v_el, const_A, mu_1, mu_4, mu_5, mu_6, lam
        self.param_dict= {"v_el":1.0, "const_A":1.0, "nu":1.0,"mu_1":0.0, "mu_4": 0.0, "mu_5":0.0, "mu_6":0.0 , "lam":0.0}
        self.parameters = list(self.param_dict.values())
        self.param_dict["dim"]=self.dim
        self.mesh = RectangleMesh(Point(-1,-1),Point(1,1),n,n)
        def boundary(x):
            return x[0] < (DOLFIN_EPS -1) or x[0] > (1 - DOLFIN_EPS) or x[1] < (DOLFIN_EPS -1) or x[1] > (1 - DOLFIN_EPS)
        self.boundary = boundary

        # - initial conditions
        # - define custom Expression
        # see https://fenicsproject.discourse.group/t/problem-defining-initial-conditions-expression-problem/626
        class d0_expr(UserExpression):
            def eval(self,values,x):
                eta = 0.05
                values[0]=pow(x[0],2)+pow(x[1],2)-0.25
                values[1]=x[1]
                tmp_abs = values[0]**2+values[1]**2
                if tmp_abs > DOLFIN_EPS:
                    values[0]= values[0] / np.sqrt(tmp_abs + eta**2)
                    values[1]= values[1] / np.sqrt(tmp_abs + eta**2)
            def value_shape(self):
                return (2,)
        zero_expr = Expression(("0.0","0.0"), degree=2)
        self.ics = [zero_expr,Constant(0.0),d0_expr(),zero_expr]
        # boundary conditions
        self.bcs = [zero_expr, Constant(0.0), None,None]

class smooth_2d:
    # as experiment one but in 2D
    # as in Becker et. al
    def __init__(self, n=2**5, dt=0.0005, T=2.0):
        self.name="smooth 2D"
        self.dim = 2
        self.fine = n
        self.dt = dt
        self.T = T
        # - model parameters namely v_el, const_A, mu_1, mu_4, mu_5, mu_6, lam
        self.param_dict= {"v_el":1.0, "const_A":1.0, "nu":1.0,"mu_1":0.0, "mu_4": 0.0, "mu_5":0.0, "mu_6":0.0 , "lam":0.0}
        self.parameters = list(self.param_dict.values())
        self.param_dict["dim"]=self.dim
        self.mesh = RectangleMesh(Point(-1,-1),Point(1,1),n,n)
        def boundary(x):
            return x[0] < (DOLFIN_EPS -1) or x[0] > (1 - DOLFIN_EPS) or x[1] < (DOLFIN_EPS -1) or x[1] > (1 - DOLFIN_EPS)
        self.boundary = boundary

        # - initial conditions
        # - define custom Expression
        # see https://fenicsproject.discourse.group/t/problem-defining-initial-conditions-expression-problem/626
        zero_expr = Expression(("0.0","0.0"), degree=2)
        d0_expr = Expression(("sin( 2.0*pi*(cos(x[0])-sin(x[1]) ) )","cos( 2.0*pi*(cos(x[0])-sin(x[1]) ) )"), degree=2, pi = np.pi)
        self.ics = [zero_expr,Constant(0.0),d0_expr,zero_expr]
        # boundary conditions
        # - recheck whether the conditions for d and q are correct. Generalize to not using bcs at all
        #self.bcs = [zero_expr, Constant(0.0), d0_expr,zero_expr]
        self.bcs = [zero_expr, Constant(0.0), None, None]
        # - parameters namely, v_el, const_A

class velocity_driven_flow_2d:
    # as experiment one but in 2D
    # as in Becker et. al
    def __init__(self, n=2**5, dt=0.0005, T=1.0):
        self.name="velocity driven flow 2D"
        self.dim = 2
        self.fine = n
        self.dt = dt
        self.T = T
        # - model parameters namely v_el, const_A, mu_1, mu_4, mu_5, mu_6, lam
        self.param_dict= {"v_el":1.0, "const_A":1.0, "nu":1.0,"mu_1":0.0, "mu_4": 0.0, "mu_5":0.0, "mu_6":0.0 , "lam":0.0}
        self.parameters = list(self.param_dict.values())
        self.param_dict["dim"]=self.dim
        self.mesh = RectangleMesh(Point(-1,-1),Point(1,1),n,n)
        def boundary(x):
            return x[0] < (DOLFIN_EPS -1) or x[0] > (1 - DOLFIN_EPS) or x[1] < (DOLFIN_EPS -1) or x[1] > (1 - DOLFIN_EPS)
        self.boundary = boundary

        # - initial conditions
        # - define custom Expression
        # see https://fenicsproject.discourse.group/t/problem-defining-initial-conditions-expression-problem/626
        class d0_expr(UserExpression):
            def eval(self,values,x):
                eta = 0.05
                values[0]=pow(x[0],2)+pow(x[1],2)-0.25
                values[1]=x[1]
                tmp_abs = values[0]**2+values[1]**2
                if tmp_abs > DOLFIN_EPS:
                    values[0]= values[0] / np.sqrt(tmp_abs + eta**2)
                    values[1]= values[1] / np.sqrt(tmp_abs + eta**2)
            def value_shape(self):
                return (2,)
        v_expr = Expression(("-10.0*x[1]","10.0*x[0]"), degree=1)
        zero_expr = Expression(("0.0","0.0"), degree=2)
        self.ics = [v_expr,Constant(0.0),d0_expr(),zero_expr]
        # boundary conditions
        self.bcs = [v_expr, Constant(0.0), None, None]

        # - parameters namely, v_el, const_A

class velocity_driven_flow_3d:
    # as experiment one but in 2D
    # as in Becker et. al
    def __init__(self, n=2**5, dt=0.0005, T=0.5):
        self.name="velocity driven flow 3D"
        self.dim = 3
        self.fine = n
        self.dt = dt
        self.T = T
        # - model parameters namely v_el, const_A, mu_1, mu_4, mu_5, mu_6, lam
        self.param_dict= {"v_el":1.0, "const_A":.1, "nu":1.0,"mu_1":0.0, "mu_4": 0.0, "mu_5":0.0, "mu_6":0.0 , "lam":0.0}
        self.parameters = list(self.param_dict.values())
        self.mesh = BoxMesh(Point(-0.5, -0.5,-0.5), Point(0.5, 0.5,0.5),n,n,n)
        def boundary(x):
            return x[0] < (DOLFIN_EPS -0.5) or x[0] > (0.5 - DOLFIN_EPS) or x[1] < (DOLFIN_EPS -0.5) or x[1] > (0.5 - DOLFIN_EPS) or x[2] < (DOLFIN_EPS -0.5) or x[2] > (0.5 - DOLFIN_EPS)  
        self.boundary = boundary

        # - initial conditions
        # - define custom Expression
        # see https://fenicsproject.discourse.group/t/problem-defining-initial-conditions-expression-problem/626
        class d0_expr(UserExpression):
            def eval(self,values,x):
                if x[1]==0 and (x[0]==0.25 or x[0]==-0.25):
                    values[0]=0.0
                    values[1]=0.0
                    values[2]=1.0
                else:
                    values[0]=4.0*pow(x[0],2)+4*pow(x[1],2)-0.25
                    values[1]=2.0*x[1]
                    values[2]=0.0
                    tmp_abs = np.sqrt(values[0]**2+values[1]**2+values[2]**2)
                    values[0]= values[0] / tmp_abs
                    values[1]= values[1] / tmp_abs
                    values[2]= values[2] / tmp_abs
            def value_shape(self):
                return (3,)
        v_expr = Expression(("-20.0*x[1]","20.0*x[0]","0"), degree=2)
        zero_expr = Expression(("0.0","0.0","0"), degree=2)
        self.ics = [v_expr,Constant(0.0),d0_expr(),zero_expr]
        # boundary conditions
        self.bcs = [v_expr, Constant(0.0), None,None]

        # - parameters namely, v_el, const_A



