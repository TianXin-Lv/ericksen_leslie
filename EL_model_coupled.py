# Ericksen-Leslie-Model
from fenics import *
import numpy as np

#temporary
from my_plots import sliced_quiver, quiver_2d
from datetime import datetime

# Custom definitions
def grad_sym(v):
    return   0.5*(grad(v) + grad(v).T)
def grad_skw(v):
    return   0.5*(grad(v) - grad(v).T)

def cross_mat(d0, d1, dim):
    if dim ==2:
        S = as_matrix( [ [d1[1] , d1[0]] , [-d1[0] , d1[1]]] )
        D = as_matrix( [ [1.0 , 0.0] , [0.0 , 0.0]] )
        Sinv = as_matrix( [ [d0[1] , -d0[0]] , [d0[0] , d0[1]]] )
        return (S*D*Sinv)
        #return as_matrix( [ [1.0 , 0.0] , [0.0 , 1.0]] )
    if dim ==3:
        I= as_matrix( [ [1.0 ,0.0 , 0.0],  [0.0 , 1.0, 0.0] , [0.0 , 0.0, 1.0]] )
        return (I - outer(d1,d0))


"""
Linearized version for fixpoint argument
"""
class linear_fp:
    def __init__(self,  experiment):
        self.modelname = "Coupled Fixpoint scheme without unit norm constraint for the general Ericksen-Leslie-model"
        # - model parameters namely v_el, const_A, mu_1, mu_4, mu_5, mu_6, lam
        #self.parameters = [.25, 1.0, 1.0, 1.0, 1.0 , 1.0, 1.0]
        # parameters for experiment 1
        self.parameters = experiment.parameters
        self.dim = experiment.dim
        self.dt=experiment.dt
        self.t = 0
        self.boundary = experiment.boundary
        self.mesh = experiment.mesh
        #print("- creating function spaces...")
        self.create_function_spaces(self.mesh)
        #print("- creating the variational formulation...")
        self.create_variational_formulation()
        #print("- setting the initial conditions...")
        self.set_IC(experiment.ics)
        #print("- setting the boundary conditions...")
        self.set_BC(experiment.bcs)

        self.tol= 1e-09
     

    def iterate(self):
        [bc_v, bc_p, bc_d, bc_q] = self.bcs

        #---
        Ksolver = KrylovSolver("bicgstab","hypre_amg")
        Ksolver.parameters["absolute_tolerance"] = 1E-10
        Ksolver.parameters["relative_tolerance"] = 1E-9
        Ksolver.parameters["maximum_iterations"] = 1000
        #Ksolver.parameters["linear_solver"] = "mumps"
        #prm.maximum_iterations = 1000

        Aa = assemble(self.La)
        ba = assemble(self.Ra)
        bc_v.apply(Aa, ba)
        #bc_p.apply(Aa, ba)
        Ksolver.solve(Aa, self.ul.vector(), ba)
        #solve(self.La == self.Ra ,self.ul,bcs=[bc_v])
        #solver.solve(Aa, self.ul.vector(), ba)
        
        #--
        #solve(Fc == 0, dl1, bc_d)
        Ac = assemble(self.Lc)
        #print(Ac.array())
        bc = assemble(self.Rc)
        #print("bc  ",bc[1:100000])
        #bc_d.apply(Ac,bc)
        #print("bc  ", assemble(self.tmp_form)[1:1000000])

        #solve(Ac,self.dl.vector(),bc)
        solve(Ac, self.dl.vector(), bc, "mumps")
        

        #quiver_2d(self.mesh, self.dl, "d2", str(datetime.now()), "test") 
 
        Ab = assemble(self.Lb)
        bb = assemble(self.Rb)
        solve(Ab,self.ql.vector(),bb, "mumps") # solver_parameters={'linear_solver': 'mumps'})

        #dxL = dx(scheme='vertex', degree=1, metadata={'representation': 'quadrature', 'degree': 1})

        #print("Error in second equation is ", np.max(np.abs( assemble(self.parameters[1]* inner( grad(Constant(0.5) *(self.dl+self.d0)), grad(self.b))*dx -  inner(self.ql,self.b)*dxL) )))

    def update(self): 
        #print("before assign", np.max(np.abs(self.dl.vector()[:]-self.dl0.vector()[:])))
        # assigning 
        assign(self.ul0, self.ul)
        self.dl0.assign(self.dl)
        self.ql0.assign(self.ql)
        #self.dl0.vector()[:] = self.dl.vector()[:]
        #self.ql0.vector()[:] = self.ql.vector()[:]
        #print("after assign", np.max(np.abs(self.dl.vector()[:]-self.dl0.vector()[:])))
    
    def set_IC(self, ics):
        for i in range(len(self.init_functions)):
            assign(self.init_functions[i], interpolate(ics[i], self.init_spaces[i]))
        assign(self.ul0.sub(0), self.u0.sub(0))
        assign(self.ul0.sub(1), self.u0.sub(1))   
        #print("-- projecting d0...")
        self.grad_d0_project.assign(project(grad(self.d0),self.TensorF, solver_type="petsc"))
        #print("-- projection of d0 done.")
        # for consistency also compute q0 - but necessary??
        #print("-- computing q0...")
        Ab = assemble(self.Lb)
        bb = assemble(self.Rb)
        solve(Ab,self.q0.vector(),bb, "mumps") # solver_parameters={'linear_solver': 'mumps'})
        #print("-- computaton of q0 done.")
        self.dl0.assign(self.d0)
        self.ql0.assign(self.q0)

    def get_IC(self):
        return self.init_functions

    def push_to_IC(self):
        # sets the current solution as IC
        # assigning 
        #self.update()
        #print("before push to IC", np.max(np.abs(self.dl.vector()[:]-self.d0.vector()[:])))
        assign(self.u0, self.ul)
        self.d0.assign(self.dl)
        self.q0.assign(self.ql)
        #self.d0.vector()[:] = self.dl.vector()[:]
        #self.q0.vector()[:] = self.ql.vector()[:]
        #print("-- projectiong current d0...")
        self.grad_d0_project.assign(project(grad(self.d0),self.TensorF, solver_type="petsc"))
        #print("-- projection of current d0 done.")
        #print("after push to IC ", np.max(np.abs(self.dl.vector()[:]-self.d0.vector()[:])))

    def set_BC(self, bcs):
        self.bcs = []
        for i in range(len(self.boundary_spaces)):
            if bcs[i] != "":
                self.bcs.append(DirichletBC(self.boundary_spaces[i], bcs[i], self.boundary))
            else:
                self.bcs.append("")
    
    def get_BC(self):
        return self.bcs

    def get_functions(self,dc=False):
        (vl0,pl0,dl0,ql0)=self.ul0.split(deepcopy=dc)
        return [vl0,pl0,dl0,ql0]

    def create_function_spaces(self, mesh):
        # -- create Function spaces
        # make Taylor-Hood-space for the velocity
        if self.dim==3:
            basic_fem = tetrahedron
        if self.dim==2: 
            basic_fem = triangle
        
        self.V = VectorElement('P', basic_fem, 2 , dim=self.dim)
        self.P = FiniteElement('P', basic_fem, 1 )
        self.D = VectorElement('P', basic_fem, 1 , dim=self.dim)
        self.Q = VectorElement( 'P',basic_fem, 1 , dim=self.dim)
        #--
        self.element = MixedElement(self.V,self.P, self.D, self.Q)
        self.F = FunctionSpace(mesh,self.element)

        # later needed for the projection of grad (d)
        self.TensorF = TensorFunctionSpace(mesh, "P",1, shape=(self.dim,self.dim))

        self.function_spaces = [self.F.sub(0),self.F.sub(2), self.F.sub(3),self.TensorF]
        self.boundary_spaces = [self.F.sub(0),self.F.sub(1),self.F.sub(2),self.F.sub(3)]
        self.init_spaces = [self.F.sub(0).collapse(),self.F.sub(1).collapse(),self.F.sub(2).collapse(),self.F.sub(3).collapse()]

        self.normal_vector = FacetNormal(mesh)

    def create_variational_formulation(self):
        dt = Constant(self.dt)
        # the following is fine, although its a copy since the constants dont change
        [v_el, const_A, nu, mu_1, mu_4, mu_5, mu_6, lam] = self.parameters
        
        #print("- create functions...")
        # Define variational functions
        (self.vl1,self.pl1) = TrialFunctions(self.TH)
        self.ul0 = Function(self.TH)
        (self.vl0,self.pl0)=split(self.ul0)
        #ul0.assign(u0)
        self.ul = Function(self.TH)
        (self.vl,self.pl)=split(self.ul)
        self.u_test = TestFunction(self.TH)
        (self.a,self.h)=split(self.u_test)
        self.u0 = Function(self.TH)
        (self.v0,self.p0)=split(self.u0)
        #
        self.c = TestFunction(self.D)
        self.d0 = Function(self.D)
        #
        self.b = TestFunction(self.Q)
        self.q0 = Function(self.Q)
        #
        self.dl1 = TrialFunction(self.D)
        #dl1 = Function(D)
        self.dl0 = Function(self.D)
        self.dl = Function(self.D)
        #
        self.ql1 = TrialFunction(self.Q)
        self.ql0 = Function(self.Q)
        
        self.ql = Function(self.Q)

        self.init_functions = [self.u0.sub(0),self.u0.sub(1),self.d0,self.q0]

        #projection on L2 i still missing
        self.grad_d0_project = Function(self.TensorF) #solver_parameters={'linear_solver': 'mumps'})
        

        # Mass Lumping
        dxL = dx(scheme='vertex', degree=1, metadata={'representation': 'quadrature', 'degree': 1})


        #vl0, pl0, dl0, ql0, v0, p0, d0, q0 =  self.vl0, self.pl0, self.dl0, self.ql0, self.v0, self.p0, self.d0, self.q0 
        #grad_d0_project = self.grad_d0_project
        #vl1, pl1, dl1, ql1,
        #self.vl1, self.pl1, self.dl1, self.ql1,
        #print("- create UFL forms...")
        # Energy term
        E = const_A * inner( grad(Constant(0.5) *(self.dl+self.d0)), grad(self.b))

        I_dd0 = cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim)

        T_L = dt*v_el*Constant(mu_1+lam**2)*inner(inner(0.5 *(self.dl0+self.d0),dot(grad_sym(self.vl1),0.5 *(self.dl0+self.d0))),inner(0.5 *(self.dl0+self.d0),dot(grad_sym(self.a),0.5 *(self.dl0+self.d0))))*dx\
                + dt*Constant(mu_4)*inner( grad_sym(self.vl1), grad_sym(self.a))*dx \
                + dt*v_el* Constant(mu_5+mu_6-lam**2)*inner( dot(grad_sym(self.vl1),self.dl0), dot(grad_sym(self.a),self.dl0))*dx \
                - dt*Constant(lam)*inner(dot( I_dd0 , self.ql0), dot(grad_sym(self.a),0.5 *(self.dl0+self.d0)))*dxL \
                - dt*inner(dot(grad_skw(self.a),self.ql0),0.5 *(self.dl0+self.d0))*dxL 


        Fa =  inner( (self.vl1-self.v0) , self.a )*dx \
            + dt*inner(dot(self.v0, nabla_grad(self.vl1)), self.a)*dx + dt*0.5*div(self.v0)*inner(self.vl1, self.a)*dx \
            - dt*v_el* inner( dot(cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim), dot(self.grad_d0_project,self.a)) ,  self.ql0 )*dxL\
            -dt*inner(self.pl1,div(self.a))*dx + div(self.vl1)*self.h*dx \
            + T_L
            #

        self.La = lhs(Fa)
        self.Ra = rhs(Fa)

        #Fb = E*dx - inner(self.ql1,self.b)*dxL
        Fb = E*dx - inner(self.ql1,self.b)*dxL #+ dt*inner(self.c,dot(grad(self.dl),self.normal_vector))*ds
        self.Lb = lhs(Fb)
        self.Rb = rhs(Fb)

        #semi-implicit in dl1
        Fc = inner((self.dl1-self.d0), self.c)*dxL \
            + dt*v_el* inner( dot(cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim), dot(self.grad_d0_project,self.vl0)) ,  self.c )*dxL \
            + dt*inner(dot(cross_mat(0.5 *(self.dl1+self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.ql0), self.c)*dxL \
            + dt*v_el*Constant(lam)*inner(self.c,dot(cross_mat(0.5 *(self.dl1+self.d0), 0.5 *(self.dl0+self.d0),self.dim), dot(grad_sym(self.vl0),0.5 *(self.dl0+self.d0))))*dxL \
            - dt*v_el*inner(dot(grad_skw(self.vl0),0.5 *(self.dl1+self.d0)),self.c)*dxL 

            #+ Constant(lam)*inner(self.c,dot(cross_mat( 0.5 *(self.dl0+self.d0),self.dim), dot(grad_sym(self.vl0),0.5 *(self.dl0+self.d0))))*dxL\
            #- inner(dot(grad_skw(self.vl0),0.5 *(self.dl1+self.d0)),self.c)*dxL \
        self.Lc = lhs(Fc)
        self.Rc = rhs(Fc)
        


        #self.tmp_form = v_el* inner( dot(cross_mat(0.5 *(self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.grad_d0_project*self.vl0) ,  self.c )*dxL \
        #    + inner(dot(cross_mat(0.5 *(self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.ql0), self.c)*dxL

    def nodal_unit_norm(self):
        vertices = self.mesh.coordinates()
        abs_values = []
        for x in vertices:
            tmp = self.dl(x)
            abs_values.append(tmp[0]**2+tmp[1]**2)

        return abs_values

    def accuracy(self):
        """
        measures how precise the solution to our approximate problem is
        """
        dt = Constant(self.dt)
        # the following is fine, although its a copy since the constants dont change
        [v_el, const_A, nu, mu_1, mu_4, mu_5, mu_6, lam] = self.parameters
        # Mass Lumping
        dxL = dx(scheme='vertex', degree=1, metadata={'representation': 'quadrature', 'degree': 1})

        I_dd0 = cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim)

        T_L = dt*Constant(v_el)*Constant(mu_1+lam**2)*inner(inner(0.5 *(self.dl0+self.d0),dot(grad_sym(self.vl),0.5 *(self.dl0+self.d0))),inner(0.5 *(self.dl0+self.d0),dot(grad_sym(self.a),0.5 *(self.dl0+self.d0))))*dx\
                + dt*Constant(mu_4)*inner( grad_sym(self.vl), grad_sym(self.a))*dx \
                + dt*Constant(v_el)*Constant(mu_5+mu_6-lam**2)*inner( dot(grad_sym(self.vl),self.dl0), dot(grad_sym(self.a),self.dl0))*dx \
                - dt*Constant(lam)*inner(dot( I_dd0 , self.ql0), dot(grad_sym(self.a),0.5 *(self.dl0+self.d0)))*dxL\
                - dt*inner(dot(grad_skw(self.a),self.ql0),0.5 *(self.dl0+self.d0))*dxL

        Fa =  inner( (self.vl-self.v0) , self.a )*dx \
            + dt*inner(dot(self.v0, nabla_grad(self.vl)), self.a)*dx + dt*0.5*div(self.v0)*inner(self.vl, self.a)*dx \
            - dt*Constant(v_el)* inner( dot(cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim), dot(self.grad_d0_project,self.a)) ,  self.ql0 )*dxL \
            -dt*inner(self.pl,div(self.a))*dx \
            +T_L


        E = const_A * inner( grad(Constant(0.5) *(self.dl+self.d0)), grad(self.b))


        Fb = E*dx - inner(self.ql,self.b)*dxL

        #semi-implicit in dl1
        Fc = inner((self.dl-self.d0), self.c)*dxL \
            + dt* Constant(v_el)* inner( dot(cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.grad_d0_project*self.vl0) ,  self.c )*dxL \
            + dt*inner(dot(cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.ql0), self.c)*dxL \
            + Constant(lam)*inner(self.c,dot(cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl0+self.d0),self.dim), dot(grad_sym(self.vl0),0.5 *(self.dl0+self.d0))))*dxL\
            - inner(dot(grad_skw(self.vl0),0.5 *(self.dl+self.d0)),self.c)*dxL 
            #+ dt*inner(self.c,dot(nabla_grad(self.dl),self.normal_vector))*ds
         
        tmp = Function(self.TH)
        
        tmp.vector()[0]=1
        (a,b)= split(tmp)
        test_form = inner( dot(cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim), dot(self.grad_d0_project,a)) ,  self.ql0 )*dxL
        #print("rhs ",assemble(test_form))


        tmp = assemble(Fa)
        #print(tmp)

        #print("N-BC ", np.max(np.abs(assemble(inner(self.c,dot(self.normal_vector,grad(self.dl)))*ds)[:] )))
        # for b in boundaries:
        #     b.homogenize() # (Sets Dirichlet data to 0)
        #     b.apply(Fa.vector())
        #for bc in self.bcs:
        self.bcs[0].apply(tmp, self.ul.vector())
        return [tmp,assemble(Fb),assemble(Fc), assemble(div(self.vl)*self.h*dx)]

    def consistency(self):
        """
        measures how precise the solution to our linearized approximate problem solves the original non-linear problem
        """
        dt = Constant(self.dt)
        # the following is fine, although its a copy since the constants dont change
        [v_el, const_A, nu, mu_1, mu_4, mu_5, mu_6, lam] = self.parameters
        # Mass Lumping
        dxL = dx(scheme='vertex', degree=1, metadata={'representation': 'quadrature', 'degree': 1})

        I_dd1 = cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl+self.d0),self.dim)

        T_L = dt*Constant(v_el)*Constant(mu_1+lam**2)*inner(inner(0.5 *(self.dl+self.d0),dot(grad_sym(self.vl),0.5 *(self.dl+self.d0))),inner(0.5 *(self.dl+self.d0),dot(grad_sym(self.a),0.5 *(self.dl+self.d0))))*dx\
                + dt*Constant(mu_4)*inner( grad_sym(self.vl), grad_sym(self.a))*dx \
                + dt*Constant(v_el)*Constant(mu_5+mu_6-lam**2)*inner( dot(grad_sym(self.vl),self.dl), dot(grad_sym(self.a),self.dl))*dx \
                - dt*Constant(lam)*inner(dot( I_dd1 , self.ql), dot(grad_sym(self.a),0.5 *(self.dl+self.d0)))*dxL\
                - dt*inner(dot(grad_skw(self.a),self.ql),0.5 *(self.dl+self.d0))*dxL

        #+ 0.5 * inner(div(v0)*vl1,a)*dx \
        Fa =  inner( (self.vl-self.v0) , self.a )*dx \
            + dt* inner(grad(self.vl)*self.v0, self.a)*dx + dt*0.5*div(self.v0)*inner(self.vl, self.a)*dx \
            - dt* v_el* inner( dot(cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl+self.d0),self.dim), self.grad_d0_project*self.a) ,  self.ql )*dxL\
            - dt* inner(self.pl,div(self.a))*dx + div(self.vl)*self.h*dx #\


        E = const_A * inner( grad(Constant(0.5) *(self.dl+self.d0)), grad(self.b))


        Fb = E*dx - inner(self.ql,self.b)*dxL

        #semi-implicit in dl1
        Fc = inner((self.dl-self.d0), self.c)*dxL \
            + dt* v_el* inner( dot( I_dd1 , self.grad_d0_project*self.vl) ,  self.c )*dxL \
            + dt*inner(dot(cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.ql), self.c)*dxL \
            + Constant(lam)*inner(self.c,dot( I_dd1 , dot(grad_sym(self.vl0),0.5 *(self.dl+self.d0))))*dxL\
            - inner(dot(grad_skw(self.vl0),0.5 *(self.dl+self.d0)),self.c)*dxL 

        tmp = assemble(Fa)
        #print(tmp)
        # for b in boundaries:
        #     b.homogenize() # (Sets Dirichlet data to 0)
        #     b.apply(Fa.vector())
        #for bc in self.bcs:
        self.bcs[0].apply(tmp, self.ul.vector())
        return [tmp,assemble(Fb),assemble(Fc)]

"""
Simplified version of Linearized version for fixpoint argument, in order to fit the observations of Prohl.
"""
class linear_fp_simple:
    def __init__(self,  experiment):
        self.modelname = "Coupled Fixpoint scheme without unit norm constraint for the simplified Ericksen-Leslie-model"
        # - model parameters namely v_el, const_A, mu_1, mu_4, mu_5, mu_6, lam
        #self.parameters = [.25, 1.0, 1.0, 1.0, 1.0 , 1.0, 1.0]
        # parameters for experiment 1
        self.parameters = experiment.parameters
        self.dim = experiment.dim
        self.dt=experiment.dt
        self.t = 0
        self.boundary = experiment.boundary
        self.mesh = experiment.mesh
        #print("- creating function spaces...")
        self.create_function_spaces(self.mesh)
        #print("- creating the variational formulation...")
        self.create_variational_formulation()
        #print("- setting the initial conditions...")
        self.set_IC(experiment.ics)
        #print("- setting the boundary conditions...")
        self.set_BC(experiment.bcs)

        self.tol= 1e-09
     
    def iterate(self):
        [bc_v, bc_p, bc_d, bc_q] = self.bcs

        #---
        """
        in order to list the possible methods use
        list_linear_solver_methods()
        list_krylov_solver_methods()
        list_krylov_solver_preconditioners()
        """
        Ksolver = KrylovSolver("bicgstab","hypre_amg")
        Ksolver.parameters["absolute_tolerance"] = 1E-10
        Ksolver.parameters["relative_tolerance"] = 1E-9
        Ksolver.parameters["maximum_iterations"] = 1000
        #Ksolver.parameters["linear_solver"] = "mumps"
        #prm.maximum_iterations = 1000

        A = assemble(self.L)
        b = assemble(self.R)
        bc_v.apply(A, b)
        if bc_d!= None:
            bc_d.apply(A,b)
        #bc_p.apply(Aa, ba)
        #Ksolver.solve(A, self.ul.vector(), b)
        solve(self.L == self.R ,self.ul,bcs=[bc_v],solver_parameters={'linear_solver': 'mumps'}) #,'preconditioner': 'hypre_amg'})
        #solver.solve(Aa, self.ul.vector(), ba)
        #(vl,pl,dl,ql)=model.ul.split()
        #sliced_quiver(self.mesh, dl, "dbefore", t, self.path, scale = self.quiver_scale)
        #assign(self.ul.sub(2), normalize(interpolate(self.ul.sub(2), self.init_spaces[2]),dim=self.dim))
        #(vl,pl,dl,ql)=model.ul.split()
        #sliced_quiver(mesh, f, "dafter", t, self.path, scale = self.quiver_scale)
        #--
        #solve(Fc == 0, dl1, bc_d)
        #Ac = assemble(self.Lc)
        #print(Ac.array())
        #bc = assemble(self.Rc)
        #print("bc  ",bc[1:100000])
        #bc_d.apply(Ac,bc)
        #print("bc  ", assemble(self.tmp_form)[1:1000000])

        #solve(Ac,self.dl.vector(),bc)
        #solve(Ac, self.dl.vector(), bc, "mumps")
        

        #quiver_2d(self.mesh, self.dl, "d2", str(datetime.now()), "test") 
 
        #Ab = assemble(self.Lb)
        #bb = assemble(self.Rb)
        #solve(Ab,self.ql.vector(),bb, "mumps") # solver_parameters={'linear_solver': 'mumps'})

        #dxL = dx(scheme='vertex', degree=1, metadata={'representation': 'quadrature', 'degree': 1})

        #print("Error in second equation is ", np.max(np.abs( assemble(self.parameters[1]* inner( grad(Constant(0.5) *(self.dl+self.d0)), grad(self.b))*dx -  inner(self.ql,self.b)*dxL) )))

    def update(self): 
        #print("before assign", np.max(np.abs(self.dl.vector()[:]-self.dl0.vector()[:])))
        # assigning 
        assign(self.ul0, self.ul)
        #self.dl0.assign(self.dl)
        #self.ql0.assign(self.ql)
        #self.dl0.vector()[:] = self.dl.vector()[:]
        #self.ql0.vector()[:] = self.ql.vector()[:]
        #print("after assign", np.max(np.abs(self.dl.vector()[:]-self.dl0.vector()[:])))
    
    def set_IC(self, ics):
        for i in range(len(self.init_functions)):
            assign(self.init_functions[i], interpolate(ics[i], self.init_spaces[i]))
            assign(self.ul0.sub(i), self.u0.sub(i))
        #assign(self.ul0.sub(1), self.u0.sub(1))   
        #print("-- projecting d0...")
        self.grad_d0_project.assign(project(grad(self.d0),self.TensorF, solver_type="petsc"))
        #print("-- projection of d0 done.")
        # for consistency also compute q0 - but necessary??
        # print("-- computing q0...")
        # Ab = assemble(self.Lb)
        # bb = assemble(self.Rb)
        # solve(Ab,self.q0.vector(),bb, "mumps") # solver_parameters={'linear_solver': 'mumps'})
        # print("-- computaton of q0 done.")
        # self.dl0.assign(self.d0)
        # self.ql0.assign(self.q0)

    def get_IC(self):
        return self.init_functions

    def push_to_IC(self):
        # sets the current solution as IC
        # assigning 
        #self.update()
        #print("before push to IC", np.max(np.abs(self.dl.vector()[:]-self.d0.vector()[:])))
        assign(self.u0, self.ul)
        #self.d0.assign(self.dl)
        #self.q0.assign(self.ql)
        #self.d0.vector()[:] = self.dl.vector()[:]
        #self.q0.vector()[:] = self.ql.vector()[:]
        #print("-- projectiong current d0...")
        self.grad_d0_project.assign(project(grad(self.d0),self.TensorF, solver_type="petsc"))
        #print("-- projection of current d0 done.")
        #print("after push to IC ", np.max(np.abs(self.dl.vector()[:]-self.d0.vector()[:])))

    def set_BC(self, bcs):
        self.bcs = []
        for i in range(len(self.boundary_spaces)):
            if bcs[i] != None:
                self.bcs.append(DirichletBC(self.boundary_spaces[i], bcs[i], self.boundary))
            else:
                self.bcs.append(None)
    
    def get_BC(self):
        return self.bcs

    def get_functions(self,dc=False):
        (vl0,pl0,dl0,ql0)=self.ul0.split(deepcopy=dc)
        return [vl0,pl0,dl0,ql0]

    def create_function_spaces(self, mesh):
        # -- create Function spaces
        # make Taylor-Hood-space for the velocity
        if self.dim==3:
            basic_fem = tetrahedron
        if self.dim==2: 
            basic_fem = triangle
        
        self.V = VectorElement('P', basic_fem, 2 , dim=self.dim)
        self.P = FiniteElement('P', basic_fem, 1 )
        self.D = VectorElement('P', basic_fem, 1 , dim=self.dim)
        self.Q = VectorElement( 'P',basic_fem, 1 , dim=self.dim)
        #--
        self.element = MixedElement(self.V,self.P, self.D, self.Q)
        self.F = FunctionSpace(mesh,self.element)

        # later needed for the projection of grad (d)
        self.TensorF = TensorFunctionSpace(mesh, "P",1, shape=(self.dim,self.dim))

        self.function_spaces = [self.F.sub(0),self.F.sub(2), self.F.sub(3),self.TensorF]
        self.boundary_spaces = [self.F.sub(0),self.F.sub(1),self.F.sub(2),self.F.sub(3)]
        self.init_spaces = [self.F.sub(0).collapse(),self.F.sub(1).collapse(),self.F.sub(2).collapse(),self.F.sub(3).collapse()]

        self.normal_vector = FacetNormal(mesh)

    def create_variational_formulation(self):
        dt = Constant(self.dt)
        # the following is fine, although its a copy since the constants dont change
        [v_el, const_A, nu, mu_1, mu_4, mu_5, mu_6, lam] = self.parameters
        
        #print("- create functions...")
        # Define variational functions
        (self.vl1,self.pl1,self.dl1,self.ql1) = TrialFunctions(self.F)
        self.ul0 = Function(self.F)
        (self.vl0,self.pl0,self.dl0,self.ql0)=split(self.ul0)
        #ul0.assign(u0)
        self.ul = Function(self.F)
        (self.vl,self.pl,self.dl,self.ql)=split(self.ul)
        self.u_test = TestFunction(self.F)
        (self.a,self.h,self.b,self.c)=split(self.u_test)
        self.u0 = Function(self.F)
        (self.v0,self.p0,self.d0,self.q0)=split(self.u0)
        #
        # self.c = TestFunction(self.D)
        # self.d0 = Function(self.D)
        # #
        # self.b = TestFunction(self.Q)
        # self.q0 = Function(self.Q)
        # #
        # self.dl1 = TrialFunction(self.D)
        # #dl1 = Function(D)
        # self.dl0 = Function(self.D)
        # self.dl = Function(self.D)
        # #
        # self.ql1 = TrialFunction(self.Q)
        # self.ql0 = Function(self.Q)
        
        # self.ql = Function(self.Q)

        self.init_functions = [self.u0.sub(0),self.u0.sub(1),self.u0.sub(2),self.u0.sub(3)]

        #projection on L2 i still missing
        self.grad_d0_project = Function(self.TensorF) #solver_parameters={'linear_solver': 'mumps'})
        

        # Mass Lumping
        dxL = dx(scheme='vertex', degree=1, metadata={'representation': 'quadrature', 'degree': 1})


        #vl0, pl0, dl0, ql0, v0, p0, d0, q0 =  self.vl0, self.pl0, self.dl0, self.ql0, self.v0, self.p0, self.d0, self.q0 
        #grad_d0_project = self.grad_d0_project
        #vl1, pl1, dl1, ql1,
        #self.vl1, self.pl1, self.dl1, self.ql1,
        #print("- create UFL forms...")
        


        # problem might act. be missing boundary cond. of pressure
        #+ 1/Re * inner( grad(vl1) , grad(a) )*dx \
        #+ 0.5 * inner(div(v0)*vl1,a)*dx \
        Fa =  inner( (self.vl1-self.v0) , self.a )*dx \
            + Constant(nu)*dt*inner(grad(self.vl1), grad(self.a))*dx \
            + dt*inner(dot(self.v0, nabla_grad(self.vl1)), self.a)*dx + dt*0.5*div(self.v0)*inner(self.vl1, self.a)*dx \
            - dt*v_el* inner( dot(cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim), dot(self.grad_d0_project,self.a)) ,  self.ql1 )*dxL\
            -dt*inner(self.pl1,div(self.a))*dx + div(self.vl1)*self.h*dx #\
            #+ T_L
            #

        # self.La = lhs(Fa)
        # self.Ra = rhs(Fa)

        #Fb = E*dx - inner(self.ql1,self.b)*dxL
        # Energy term
        E = const_A * inner( grad(Constant(0.5) *(self.dl1+self.d0)), grad(self.b))
        Fb = E*dx - inner(self.ql1,self.b)*dxL #+ dt*inner(self.c,dot(grad(self.dl),self.normal_vector))*ds
        # self.Lb = lhs(Fb)
        # self.Rb = rhs(Fb)

        #semi-implicit in dl1
        Fc = inner((self.dl1-self.d0), self.c)*dxL \
            + dt*v_el* inner( dot(cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim), dot(self.grad_d0_project,self.vl1)) ,  self.c )*dxL \
            + dt*inner(dot(cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.ql1), self.c)*dxL 
            #+ dt*inner(self.c,dot(self.normal_vector,grad(self.dl1)))*ds


            #+ Constant(lam)*inner(self.c,dot(cross_mat( 0.5 *(self.dl0+self.d0),self.dim), dot(grad_sym(self.vl0),0.5 *(self.dl0+self.d0))))*dxL\
            #- inner(dot(grad_skw(self.vl0),0.5 *(self.dl1+self.d0)),self.c)*dxL \
        # self.Lc = lhs(Fc)
        # self.Rc = rhs(Fc)
        
        self.L = lhs(Fa+Fb+Fc)
        self.R = rhs(Fa+Fb+Fc)


        #self.tmp_form = v_el* inner( dot(cross_mat(0.5 *(self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.grad_d0_project*self.vl0) ,  self.c )*dxL \
        #    + inner(dot(cross_mat(0.5 *(self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.ql0), self.c)*dxL

    def nodal_unit_norm(self):
        vertices = self.mesh.coordinates()
        abs_values = []
        for x in vertices:
            tmp = self.dl(x)
            abs_values.append(tmp[0]**2+tmp[1]**2)

        return abs_values

    def accuracy(self):
        """
        measures how precise the solution to our approximate problem is
        """
        dt = Constant(self.dt)
        # the following is fine, although its a copy since the constants dont change
        [v_el, const_A, nu, mu_1, mu_4, mu_5, mu_6, lam] = self.parameters
        # Mass Lumping
        dxL = dx(scheme='vertex', degree=1, metadata={'representation': 'quadrature', 'degree': 1})


        Fa =  inner( (self.vl-self.v0) , self.a )*dx \
            + Constant(nu)*dt*inner(grad(self.vl), grad(self.a))*dx \
            + dt*inner(dot(self.v0, nabla_grad(self.vl)), self.a)*dx + dt*0.5*div(self.v0)*inner(self.vl, self.a)*dx \
            - dt*v_el* inner( dot(cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim), dot(self.grad_d0_project,self.a)) ,  self.ql0 )*dxL \
            -dt*inner(self.pl,div(self.a))*dx # + div(self.vl)*self.h*dx


        E = const_A * inner( grad(Constant(0.5) *(self.dl+self.d0)), grad(self.b))


        Fb = E*dx - inner(self.ql,self.b)*dxL

        #semi-implicit in dl1
        Fc = inner((self.dl-self.d0), self.c)*dxL \
            + dt* v_el* inner( dot(cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.grad_d0_project*self.vl0) ,  self.c )*dxL \
            + dt*inner(dot(cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.ql0), self.c)*dxL #\
            #+ dt*inner(self.c,dot(nabla_grad(self.dl),self.normal_vector))*ds
         
        tmp = Function(self.F)
        
        tmp.vector()[0]=1
        (a,b,c,d)= split(tmp)
        test_form = inner( dot(cross_mat(0.5 *(self.dl0+self.d0), 0.5 *(self.dl0+self.d0),self.dim), dot(self.grad_d0_project,a)) ,  self.ql0 )*dxL
        #print("rhs ",assemble(test_form))


        tmp = assemble(Fa)
        #print(tmp)

        #print("N-BC ", np.max(np.abs(assemble(inner(self.c,dot(self.normal_vector,grad(self.dl)))*ds)[:] )))
        # for b in boundaries:
        #     b.homogenize() # (Sets Dirichlet data to 0)
        #     b.apply(Fa.vector())
        #for bc in self.bcs:
        self.bcs[0].apply(tmp, self.ul.vector())
        return [tmp,assemble(Fb),assemble(Fc), assemble(div(self.vl)*self.h*dx)]

    def consistency(self):
        """
        measures how precise the solution to our linearized approximate problem solves the original non-linear problem
        """
        dt = Constant(self.dt)
        # the following is fine, although its a copy since the constants dont change
        [v_el, const_A, nu, mu_1, mu_4, mu_5, mu_6, lam] = self.parameters
        # Mass Lumping
        dxL = dx(scheme='vertex', degree=1, metadata={'representation': 'quadrature', 'degree': 1})


        #+ 0.5 * inner(div(v0)*vl1,a)*dx \
        Fa =  inner( (self.vl-self.v0) , self.a )*dx \
            + Constant(nu)*dt* inner(grad(self.vl), grad(self.a))*dx \
            + dt* inner(grad(self.vl)*self.v0, self.a)*dx + dt*0.5*div(self.v0)*inner(self.vl, self.a)*dx \
            - dt* v_el* inner( dot(cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl+self.d0),self.dim), self.grad_d0_project*self.a) ,  self.ql )*dxL\
            - dt* inner(self.pl,div(self.a))*dx + div(self.vl)*self.h*dx #\


        E = const_A * inner( grad(Constant(0.5) *(self.dl+self.d0)), grad(self.b))


        Fb = E*dx - inner(self.ql,self.b)*dxL

        #semi-implicit in dl1
        Fc = inner((self.dl-self.d0), self.c)*dxL \
            + dt* v_el* inner( dot(cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl+self.d0),self.dim), self.grad_d0_project*self.vl) ,  self.c )*dxL \
            + dt*inner(dot(cross_mat(0.5 *(self.dl+self.d0), 0.5 *(self.dl0+self.d0),self.dim), self.ql), self.c)*dxL #\
            #- dt*inner(self.c,dot(grad(self.dl),self.normal_vector))*ds

        tmp = assemble(Fa)
        #print(tmp)
        # for b in boundaries:
        #     b.homogenize() # (Sets Dirichlet data to 0)
        #     b.apply(Fa.vector())
        #for bc in self.bcs:
        self.bcs[0].apply(tmp, self.ul.vector())
        return [tmp,assemble(Fb),assemble(Fc)]

