"""
Logs of unstructured data, saved as a csv.
To-Do:
- option to save it as JSON
"""
import pandas as pd
import json
import os

class log_class:
    """
    A class which allows to create several logs in which unstructured data can be added and saved.
    """
    def __init__(self, path):
        self.path=path
        self.logs=[]

    def add_log(self, name, sort_by=None):
        self.logs.append(log(name, sort_by))
        return self.logs[-1]

    def save_all(self):
        for log in self.logs:
            log.save(self.path)

class log:
    """
    A class which takes unstructured data, in particular nested dictionaries, and saves them as csv.
    Big advantage: almost no formatting necessary in main file. Saves a lot of lines and makes the code stay clear.
    """
    def __init__(self, name, sort_by):
        self.name=name
        self.json=[]
        self.sort_by=sort_by

    def __str__(self):
        return str(self.json)

    def add(self, object):
        self.json.append(object)

    def save(self, path):
        df=pd.DataFrame()
        for dic in self.json:
            df=pd.concat([df,pd.json_normalize(dic)])
        df.to_csv(path+"/"+self.name+".csv",index=False, encoding="utf-8")

def summary( id, directory='simulations/'):
    json =[]
    subfolders = next(os.walk(directory))[1]
    for spath in subfolders:
        if id in spath:
            df_param = pd.read_csv(directory+spath+"/parameter_log.csv")
            modelname,expname, dt,dh, timestamp = df_param["model name"][0], df_param["experiment"][0],df_param["dt"][0],df_param["dh"][0], df_param["timestamp"][0]
            success = False if ("Failed" in df_param.columns) else True
            err_msg = "" if success else ' '.join(df_param["Failed"].values[1:])
            try:
                df_evol = pd.read_csv(directory+spath+"/time_scheme_log.csv")
                total_runtime = df_evol["processing_time"].iloc[-1]
                total_iterations = df_evol["iterations"].sum() if success==True else ""
                json.append({"modelname": modelname,"experiment":expname,"timestamp":timestamp, "dt":dt,"dh":dh,"success":success, "Error": err_msg, "total_runtime":total_runtime, "total_iterations":total_iterations}) 
            except: 
                json.append({"modelname": modelname,"experiment":expname,"timestamp":timestamp, "dt":dt,"dh":dh,"success":success, "Error": err_msg, "total_runtime":"failed before first time step", "total_iterations":"failed before first time step"}) 
    df=pd.DataFrame()
    for dic in json:
        df=pd.concat([df,pd.json_normalize(dic)])
    df.to_csv(directory+"/summary_"+id+".csv",index=False, encoding="utf-8")

