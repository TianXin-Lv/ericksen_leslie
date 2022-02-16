"""
jupyter notebook as file
runs experiment
"""
"""
testasdfa
"""
from fenics import *
import threading
from functools import partial
# necessary for parallelization of simulation runs
import numpy as np
import os, sys
from datetime import datetime
import time
from postprocess import *
from experiments import *
import EL_model_linear_projection as elp
import EL_model_coupled as elc
import EL_model as eld
from my_plots import *
from logger import log_class,summary
# shut off deprecation warning
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

# def perform(f):
#     f()

def run_experiment_fp(experiment, model, dt, dh,save_freq, max_iter=50, fp_tol= 1e-06, path_bias="", silent=False):
    
    # - init. unique identification of simulation
    timestamp = datetime.now()
    timestamp = str(timestamp.day)+"-"+str(timestamp.month)+"-"+str(timestamp.year)+"_"+str(timestamp.hour)+"-"+str(timestamp.minute)
    if not silent: print("The current timestamp is: ",timestamp)
    path = "simulations/"+path_bias+"_sim_"+timestamp
    os.makedirs(path,exist_ok=True)


    parameters["form_compiler"]["precision"] = 100

    # - init logs
    
    logs = log_class(path)
    param_log = logs.add_log("parameter_log")
    time_step_log =logs.add_log("time_scheme_log", sort_by=["time"])
    dim = experiment.dim
    t=0.0
    T = experiment.T
    Energies=[[],[]]
    # - how many decimal points to be rounded for saving
    roundpot = int(np.ceil(-1*np.log10(dt)))


    # - logging parameters
    param_log.add({**{"model name":model.modelname, "experiment": experiment.name ,"timestamp":timestamp, "maximum iterations" : max_iter, "tolerance of fixpoint solver": fp_tol, "T":T,"dt":dt,"dh":dh, "plot frequency":save_freq},**experiment.param_dict})
    param_log.save(path)
    # - init postprocess 
    postprocess_v = fem_postprocess( dim, ["xdmf","quiver"], path, "v")
    postprocess_d = fem_postprocess( dim, ["xdmf","quiver"], path, "d")
    # - save initial conditions
    (vl,pl,dl,ql)=model.get_functions()

    #WHY ERROR HERE?
    #postprocess_v.save(vl,experiment.mesh, round(0.0,roundpot))
    #postprocess_d.save(dl,experiment.mesh, round(0.0,roundpot))

    if not silent: print("Starting the time evolution")
    accumulated_time=0.0
    try:
        while t <= T:
            t += model.dt
            # Start of fixpoint-iteration
            iteration=0
            total_err = float("inf")
            while total_err>fp_tol and iteration<max_iter:
                iteration+= 1
                if not silent: print("time ",t, ", iteration ",iteration)
                time_meas_start = time.process_time()
                # Time of parts of postprocesssing need to be included since tolerance check is eminent for fixpoint method
                #------------------------------------------------
                (vl0,pl0,dl0,ql0)=model.get_functions(dc=True)
                
                
                model.iterate()

                # check whether this makes a difference
                #(vl,pl,dl,ql)=model.ul.split()

                model.update()
                
                
                (vl,pl,dl,ql)=model.get_functions(dc=True)

                e_v = vl.vector()[:]-vl0.vector()[:]
                e_d = dl.vector()[:]-dl0.vector()[:]
                e_q = ql.vector()[:]-ql0.vector()[:]
                
                               
                
                #consistency_err = [np.max(np.abs(c_err[0])), np.max(np.abs(c_err[1])), np.max(np.abs(c_err[2])) ]
                # accuracy_err = [np.max(np.abs(acc[0])), np.max(np.abs(acc[1])), np.max(np.abs(acc[2])) ]
                
                total_err = np.max([np.max(np.abs(e_v)), np.max(np.abs(e_d)) ,np.max(np.abs(e_q))])
                
                fp_err = [np.max(np.abs(e_v)), np.max(np.abs(e_d)) ,np.max(np.abs(e_q))] 
                #------------------------------------------------
                time_meas_end = time.process_time()
                accumulated_time += time_meas_end - time_meas_start
                if not silent: print("FP Tolerance: ",fp_err)
    
            
            
                # print("accuracy ", accuracy_err )
                # print("consistency error ", consistency_err )
                #nodal_norms = model.nodal_unit_norm()
                
            if total_err<=fp_tol:
                # d_tmp=Function(model. )
                # d_tmp.vector()[:] = model.dl.vector()[:]-model.d0.vector()[:]
                energy_tmp = assemble((vl**2)/2*dx + (Constant(model.parameters[1])*(grad(model.dl))**2)/2*dx)

                # - logging errors
                time_step_log.add({"time": t , "iterations": iteration , "energy": energy_tmp  , "fp_err":dict(zip(["v","d","q"], fp_err)) , "processing_time": accumulated_time }) # \
                #    ,  "acc_err":dict(zip(["v","d","q"], accuracy_err)),  "consistency_err":dict(zip(["v","d","q"], consistency_err))  })
                time_step_log.save(path)

                # - postprocessing
                if t< (dt*3/2) or ((t+dt)%save_freq) < (3*dt/2):
                    postprocess_v.save(vl,experiment.mesh, round(t,roundpot))
                    postprocess_d.save(dl,experiment.mesh, round(t,roundpot))
                    
                    Energies[0].append(t)
                    Energies[1].append(energy_tmp)
                    energy_plot(Energies, path)

                model.push_to_IC()
            elif iteration>=max_iter: 
                raise RuntimeError("Fixpoint scheme failed to converge in "+str(iteration)+" iterations..")
                #break
    except Exception as err:
        print("Warning: Ran into error", err)
        param_log.add({"Failed": err })
        param_log.save(path)
      
    logs.save_all()

def run_multiple(cluster, experiments, models, timepartition, spacepartition, save_freq, max_iter=50, fp_tol= 1e-06, test_run=False, no_of_threads=1):
    """
    The run_id gives all simulations the same characters in the beginning of the folder name for easier comparison.
    """
    print("no of threads", no_of_threads)
    tasks=[]
    threadlist=[]
    editing_thread = 0
    counter = 1
    for e in experiments:
        for dh in spacepartition:
            for m in models:
                for dt in timepartition:
                    # only one time iteration in test case
                    exp=e(dh,dt)
                    if test_run != False: 
                        exp.T=test_run
                        save_freq=dt
                    model = m(exp)
                    s="running: "+", ".join([model.modelname, exp.name,"dt", str(dt), "dh", str(dh)])
                    print(s)
                    t=partial(run_experiment_fp, exp, model, dt, dh,save_freq, max_iter, fp_tol, path_bias=(cluster+"_run"+str(counter)), silent=True,)
                    threadlist.append(threading.Thread(target=t))
                    print("editing thread ", editing_thread)
                    # mod no_of_threads
                    if editing_thread==(no_of_threads-1):
                        print("here")
                        for threadinstance in threadlist:
                            print("starting thread")
                            threadinstance.start()
                        for threadinstance in threadlist:
                            print("waiting for thread")
                            threadinstance.join()
                        threadlist=[]
                        editing_thread=0
                        try: 
                            summary(cluster)
                        except:
                            print("Warning: summary can currently not be accessed...")
                    tasks.append((t,s))
                    counter+=1
                    editing_thread+=1
     #[ for i in range(no_of_threads)]
    
    # for (t,s) in tasks:
        
    #     print(s)
    #     print("initializing thread ",editing_thread)
    #     #threadlist[editing_thread].start()
    #     editing_thread+=1
        

            

        
        

"""
-------------------------------------------------------------------------------------------
preliminaries over here.
start of experiment.
"""
def main():
    cluster_number=sys.argv[1]
    t_run=sys.argv[2]
    no_of_threads = int(sys.argv[3])
    t_run = False if t_run == "False" else float(t_run) 
    experiments = [annihilation_3d, velocity_driven_flow_3d]
    models = [eld.linear_fp_simple] #, eld.linear_fp]
    timepartition = [.01,.001,.0005,.0001]
    spacepartition = [2**5] #,2**6]
    cluster = "cluster"+str(cluster_number)

    run_multiple(cluster, experiments, models, timepartition, spacepartition, save_freq=.01, test_run=t_run, max_iter=50, no_of_threads=no_of_threads)

if __name__ == "__main__":
    main()