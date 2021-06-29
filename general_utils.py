import labrad

import os
import inspect
import sys

import numpy as np


def labrad_hdf5_ndarray(dir_path, file_num, file_name):
    """Load a hdf5 file and return the data (numpy.array) and columns of labels (list)
    
    Parameters
    ----------
    dir_path : string
        Usually this is "vault" directory
    file_num : int
        hdf5 file number. ex. '000## - measurement_name.hdf5'
    file_name : string
        
    Returns
    -------
    data : ndarray
    variables : list
        list of parameters
    
    """
    # Load hdf5 file
    f_name = '0'*(5-len(str(file_num))) + str(file_num) + ' - ' + file_name + '.hdf5'
    f = h5py.File(dir_path + f_name,'r')['DataVault']
    raw_data = f.value
    attrs = f.attrs
    
    # Raw data to np.array
    data = np.array([list(d) for d in raw_data])
                
    # Get varialbles labels
    indep_keys = sorted([str(x) for x in list(attrs.keys()) if str(x).startswith('Independent') and str(x).endswith('label')])
    dep_keys = sorted([str(x) for x in list(attrs.keys()) if str(x).startswith('Dependent') and str(x).endswith('label')])
    indep_labels = [attrs[c] for c in indep_keys]
    dep_labels = [attrs[c] for c in dep_keys]
    variables = indep_labels + dep_labels
    
    return data, variables

  
def labrad_hdf5_get_parameters(dir_path, file_num, file_name):
    """Get parameter settings (e.g., ferquency, time constant added by DV.add_parameters()) from a labrad hdf5 file
    
    Parameters
    ----------
    dir_path : string
        Usually this is "vault" directory
    file_num : int
        hdf5 file number. ex. '00033 - measurement_name.hdf5'
    file_name : string
        
    Returns
    -------
    dictionary
        Pairs of paramter keys and values
        
    Notes
    -----
    The default parameter values are encoded by labrad format. 
    The prefix in endoded values is 'data:application/labrad;base64,'
    To decode these and get the raw value, we need to simply use DV.get_parameters()
    or change the backend script in datavault/backend.py
    This function works in the latter case.
    """
    # Load hdf5 file
    f_name = '0'*(5-len(str(file_num))) + str(file_num) + ' - ' + file_name + '.hdf5'
    f = h5py.File(dir_path + f_name,'r')['DataVault']
    attrs = f.attrs
                
    # Get parameters labels and values
    param_ukeys = sorted([str(x) for x in list(attrs.keys()) if str(x).startswith('Param')])
    param_keys = [c[6:] for c in param_ukeys]
    param_values = [attrs[c] for c in param_ukeys]
    
    return {k : v for k, v in zip(param_keys, param_values)}

  
def get_parameters_of_func(offset = None):
    """Get a dictionary of paramteres of the function.

    Parameters
    ----------
    offset : int
        default value is None
        
    Return
    ------
    dictionary
        The dictionary includes pairs of paremeter's name and the corresponding values.

    References
    ----------
    [1] https://tottoto.net/python3-get-args-of-current-function/
    """
    parent_frame = inspect.currentframe().f_back
    info = inspect.getargvalues(parent_frame)
    return {key: info.locals[key] for key in info.args[offset:]}

  
def create_labrad_hdf5file(DV, file_path, scan_name, scan_var, meas_var):
    """Create a labrad hdf5 file from ndarray.
    
    Parameters
    ----------
    DV : object
    file_path : string
    scan_name : string
    scan_var : list or tuple 
    meas_var : list or tuple
    
    Returns
    -------
    int
        The file number
        
    """
    DV.cd('')
    try:
        DV.mkdir(file_path)
        DV.cd(file_path)
    except Exception:
        DV.cd(file_path)

    file_name = file_path + '_' + scan_name
    dv_file = DV.new(file_name, scan_var, meas_var)
    print '\r',"new file created, file numer: ", int(dv_file[1][0:5])

    return int(dv_file[1][0:5])

  
def write_meas_parameters(DV, file_path, file_number, date, scan_name, meas_parameters, amplitude, sensitivity):
    """Write measurement parameters to txt file and labrad hdf5 file.
    
    Parameters
    ----------
    DV : object
    file_path : string
    file_number : int
    date : object
    scan_name : string
    meas_parameters : dict
    scan_var : list or tuple 
    meas_var : list or tuple
    amplitude : float
    sensitivity : float
    
    Returns
    -------
    None
        
    """
    
    if not os.path.isfile(meas_details_path+file_path+'.txt'):
        with open(meas_details_path+file_path+'.txt', "w+") as f: 
            pass
    with open(meas_details_path+file_path+'.txt', "a") as f:
        f.write("========"+ "\n")
        f.write("file_number: "+ str(file_number) + "\n" + "date: " + str(date) +"\n" + "measurement:" + str(scan_name) + "\n")
        for k, v in sorted(meas_parameters.items()):
            print(k, v)
            f.write(str(k) + ": "+ str(v) + "\n")
            DV.add_parameter(str(k), str(v))

        for i, LA in enumerate(LAs):
            tc = LA.time_constant()
            sens = LA.sensitivity()
            f.write("time_constant_" + str(i) + ' : ' + str(tc) + "\n")
            f.write("sensitivity_" + str(i) + ' : ' + str(sens) + "\n")
            DV.add_parameter("time_constant_" + str(i), tc)
            DV.add_parameter("sensitivity_" + str(i), sens)
        
    
def write_meas_parameters_end(date1, date2, file_path):
    
    with open(meas_details_path+file_path+'.txt', "a") as f:
        f.write("end date: " + str(date2) + "\n"+ "total time: " + str(date2-date1)+ "\n")

        
def get_variables(DV):
    """Get variables of a lablad hdf5 file

    Parameters
    ----------
    DV : object (datavault)

    Return
    ------
    list
        A variable of the a lablad hdf5 file
    """
    variables = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    return  variables
