import labrad
import numpy as np
from numpy import linalg as LA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from heapq import heappush, nlargest
import scipy.io as sio
from matplotlib.ticker import ScalarFormatter
import datetime
import time
import math
import os
import inspect
import sys
from tqdm import tqdm

import slackweb
import requests

meas_details_path = "C://Users//Young Lab//Young Lab Dropbox//Bruefors//Data//vault//Yu_Bruefors//meas_details//"

sns.set('talk', 'whitegrid', 'dark', font_scale=1.2,
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

cxn = labrad.connect()
cxn_2 = labrad.connect()
cxn_3 = labrad.connect()

tcdel = 3.0

ramp_step = 1000
ramp_delay = 2500

adc_channel_0 = 4
adc_channel_1 = 5
adc_channel_2 = 6
adc_channel_3 = 7

#Initialization
DV = cxn.data_vault
DAC = cxn.dac_adc.select_device()
sim = cxn.sim900.select_device()
MGz = cxn.ami_420.select_device()

LA1 = cxn.sr860.select_device('mildred GPIB Bus - GPIB0::8::INSTR')
LA2 = cxn_2.sr860.select_device('mildred GPIB Bus - GPIB0::4::INSTR')  
LA3 = cxn_3.sr860.select_device('mildred GPIB Bus - GPIB0::14::INSTR')
LAs = [LA1, LA2, LA3]



wait_time = 1e6*tcdel*LA1.time_constant()


def labrad_hdf5_dataloader(dir_path, file_num, file_name):
    """
    Load a hdf5 file and return the data (numpy.array) and columns of labels (list)
    
    Parameters
    ----------
    dir_path : string
        Usually this is "vault" directory
    file_num : int
        hdf5 file number. ex. '000## - measurement_name.hdf5'
    file_name : string
        
    Returns
    -------
    data : numpy.array
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
    """
    Get parameter settings from a labrad hdf5 file
    
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
    To decode these and get the raw value, we need to simply use DV.get_parameter()
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


def get_meas_parameters(offset = None):
    """
    Get a dictionary of paramteres of the function.

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


def set_lockin_parameters(amplitude = 0.01, frequency = 17.777):
    """
    Initialize the lockin amp for voltage source.
    
    Parameters
    ----------
    amplitude : int or float
        The ampitude value of the lockin.
    frequency : int or float
            
    Returns
    -------
    None
    """
    LA1.sine_out_amplitude(amplitude)
    LA1.frequency(frequency)
    time.sleep(1)
    print '\r',"parameters set done",


def create_file(DV, file_path, scan_name, scan_var, meas_var):
    """
    Create a measurment file.
    
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
    """
    Write measurement parameters to txt file and labrad hdf5 file.
    
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


        
def get_variables():
    variables = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))]

    return  variables


def plot_fig(file_name, file_num, data, cl, xsize, ysize, xaxis, yaxis, xscale, yscale, xname, yname, logy, var, unit):
        
    df = pd.DataFrame(data.T, columns=cl)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    df[yaxis[1]] = abs(df[yaxis[1]])
    
    df.plot(x=xaxis, y=yaxis[0], logy=logy[0], ax=ax1, figsize=(xsize, ysize))
    df.plot(x=xaxis, y=yaxis[1], logy=logy[1], ax=ax2, figsize=(xsize, ysize))
    df.plot(x=xaxis, y=yaxis[2], logy=logy[2], ax=ax3, figsize=(xsize, ysize))

    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname[0])
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname[1])
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname[2])
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])
    
    ax1.legend(bbox_to_anchor=(0.85, 1.11), loc='upper left', borderaxespad=0, fontsize=18)
    ax2.legend(bbox_to_anchor=(0.85, 1.11), loc='upper left', borderaxespad=0, fontsize=18)
    ax3.legend(bbox_to_anchor=(0.85, 1.11), loc='upper left', borderaxespad=0, fontsize=18)

    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)
    ax3.legend().set_visible(False)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.patch.set_facecolor('white')
    fig.tight_layout()

#     plt.text(0.01, 0.99, horizontalalignment='left', verticalalignment='top', family='monospace', transform=ax1.transAxes, fontsize=18)

    print '\r', 'searching folder...',
    flag = False
    try:
        os.mkdir(save_path+file_name)
    except Exception:
        pass

    print '\r','saving...',

    try:
        plt.savefig(save_path+file_name+'//'+file_name+'_'+str(file_num)+' at '+str(var, 4)+' ' + unit +'.png')
        flag = True
    except Exception:
        flag = False
        pass
    
    if flag:
        print '\r','slack plotting...',
        try:
            files = {'file': open(save_path+file_name+'//'+file_name+'_'+str(file_num)+' at '+str(var, 4)+' ' + unit +'.png', 'rb')}
            param = {
                'token':TOKEN,
                'channels':CHANNEL,
                'title': file_name+'_'+str(file_num)+' at '+str(var, 4)+' ' + unit
            }
            requests.post(url="https://slack.com/api/files.upload",params=param, files=files)
        except Exception:
            print '\r','slack plotting failed',
            pass



def slack_post(file_path, file_number, date, scan_name, sweep_values):

    print '\r','slack plotting...',
    try:
        files = {'file': open(save_path+file_name+'//'+file_name+'_'+str(file_num)+' at '+str(var, 4)+' ' + unit +'.png', 'rb')}
        param = {
            'token':TOKEN,
            'channels':CHANNEL,
            'title': file_name+'_'+str(file_num)+' at '+str(var, 4)+' ' + unit
        }
        requests.post(url="https://slack.com/api/files.upload",params=param, files=files)
    except Exception:
        print '\r','slack plotting failed',
        pass

def find_vt_vb(p0, n0, c_delta):    # input p0, n0, return vt, vb
    return 0.5 * (n0 + p0) / (1.0 + c_delta), 0.5 * (n0 - p0) / (1.0 - c_delta)

    
def sim_sweep(out_ch, vstart, vend, points, delay):
    vs = [[0] * points]
    vs = np.linspace(vstart, vend, num = points)
    d_tmp = None
    p1, p2, p3 = 0, 0, 0

    for jj in range(points):
        sim.dc_set_voltage(out_ch,float("{0:.4f}".format(vs[jj])))
        #time.sleep(delay*0.9)
        try:
        #line_data = [DAC.read_voltage(k) for k in in_dac_ch]
            p1, p2, p3 = LA1.x(), 1.0, 1.0
            line_data = [p1, p2, p3]
        except:
            line_data = [p1, p2, p3]
            
        if d_tmp is not None:
            d_tmp = np.vstack([d_tmp, line_data])
        else:
            d_tmp = line_data

    return  d_tmp.T



def sim_dual_sweep(out_ch_bottom, out_ch_top, vbg_start, vbg_end, vtg_start, vtg_end, points_vbg, points_vtg, delay):
    vbg_s = np.linspace(vbg_start, vbg_end, num = points_vbg)
    vtg_s = np.linspace(vtg_start, vtg_end, num = points_vtg)
    d_tmp = None
    p1, p2, p3 = 0, 0, 0

    for jj in range(points_vbg):
        sim.dc_set_voltage(out_ch_bottom,float("{0:.4f}".format(vbg_s[jj])))
        sim.dc_set_voltage(out_ch_top,float("{0:.4f}".format(vtg_s[jj])))
        time.sleep(delay*0.7)
        try:
        #line_data = [DAC.read_voltage(k) for k in in_dac_ch]
            p1, p2, p3 = LA1.x(), 1.0, 1.0
            line_data = [p1, p2, p3]
        except:
            line_data = [p1, p2, p3]
            
        if d_tmp is not None:
            d_tmp = np.vstack([d_tmp, line_data])
        else:
            d_tmp = line_data

    return  d_tmp.T




def sim_sweep_2omega(out_ch, vstart, vend, points, delay):
    vs = [[0] * points]
    vs = np.linspace(vstart, vend, num = points)
    d_tmp = None
    p1, p2, p3, p3, p5 = 0, 0, 0, 0, 0

    for jj in range(points):
        sim.dc_set_voltage(out_ch,float("{0:.4f}".format(vs[jj])))
        time.sleep(delay*0.8)
        #print vs[jj]
        try:
        #line_data = [DAC.read_voltage(k) for k in in_dac_ch]
            p1, p2, p3, p4, p5 = LA1.x(), LA2.x(), LA2.y(), LA3.x(), LA3.y()
            line_data = [p1, p2, p3, p4, p5]
        except:
            line_data = [p1, p2, p3, p4, p5]
            #print(0)
            
        if d_tmp is not None:
            d_tmp = np.vstack([d_tmp, line_data])
        else:
            d_tmp = line_data

    return  d_tmp.T


def sim_sweep_ave(out_ch, vstart, vend, points, delay):
    vs = [[0] * points]
    vs = np.linspace(vstart, vend, num = points)
    d_tmp = None
    p1, p2, p3, p4, p5 = 0, 0, 0, 0, 0

    for jj in range(points):
        sim.dc_set_voltage(out_ch,float("{0:.4f}".format(vs[jj])))
        time.sleep(delay*0.1)
        #flag2 = True
        print vs[jj]
        while flag2:
            try:
                p1, p2, p3, p4, p5 = 0.0, 0.0, 0.0, 0.0,0.0
                time.sleep(0.01)
                for _ in range(10):
                    p1 += LA1.x()
                    p2 += LA2.x()
                    p3 += LA2.y()
                    p4 += LA3.x()
                    p5 += LA3.y()
                    time.sleep(0.01)
                p1 /= 10.0
                p2 /= 10.0
                p3 /= 10.0
                p4 /= 10.0
                p5 /= 10.0
                flag2 = False
                line_data = [p1, p2, p3, p4, p5]
            except:
                print(0)
                pass
            
        if d_tmp is not None:
            d_tmp = np.vstack([d_tmp, line_data])
        else:
            d_tmp = line_data

    return  d_tmp.T


def sim_sweep_dac(out_ch, in_dac_ch, vstart, vend, points, delay):
    vs = [[0] * points]
    vs = np.linspace(vstart, vend, num = points)
    d_tmp = None

    for jj in range(points):
        sim.dc_set_voltage(out_ch,float("{0:.4f}".format(vs[jj])))
        time.sleep(delay*0.9)

        line_data = [DAC.read_voltage(k) for k in in_dac_ch]
        #line_data = [LA1.x(), LA2.x(), LA2.y()]
        if d_tmp is not None:
            d_tmp = np.vstack([d_tmp, line_data])
        else:
            d_tmp = line_data

    return  d_tmp.T



def set_Vg_nodac(voltage_source, voltage_channel, start_v, end_v):
    
    if voltage_source == "DAC":
        DAC.buffer_ramp([voltage_channel], [4,5,6,7],[start_v],[end_v],100, 500, 1)
        time.sleep(1)
        print '\r',"Voltage reached: ",end_v," V",
    
    elif voltage_source == "SIM":
        sim_sweep(voltage_channel, start_v, end_v, 100, 0.02)
        time.sleep(1)
        print '\r',"Voltage reached: ",end_v," V",
        
        
        
def set_Vg_dac(voltage_source, voltage_channel, start_v, end_v):
    
    if voltage_source == "DAC":
        DAC.buffer_ramp([voltage_channel], [adc_channel_0, adc_channel_1, adc_channel_2,adc_channel_3],[start_v],[end_v],100, 500, 1)
        time.sleep(1)
        print '\r',"Voltage reached: ",end_v," V",
    
    elif voltage_source == "SIM":
        sim_sweep2(voltage_channel, [adc_channel_0, adc_channel_1,adc_channel_2,adc_channel_3], start_v, end_v, 100, 500/1e6)
        time.sleep(1)
        print '\r',"Voltage reached: ",end_v," V",


def scan_Vg(voltage_source, meas_voltage_gain, voltage_channel, start_v, end_v, number_of_vg_points, wait_time):
#     time_constant_1 = LA1.time_constant()
#     sensitivity_1 = LA1.sensitivity()

#     time_constant_2 = LA2.time_constant()
#     sensitivity_2 = LA2.sensitivity()

#     time_constant_3 = LA3.time_constant()
#     sensitivity_3 = LA3.sensitivity()
    #vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
    vg = np.linspace(start_v, end_v, number_of_vg_points)   
    
    print '\r','Scanning Vg:  Start_V:', start_v, ' V; End_V:', end_v, ' V',
    if voltage_source == "DAC":
        res = DAC.buffer_ramp([voltage_channel], [adc_channel_0, adc_channel_1, adc_channel_2,adc_channel_3],[start_v],[end_v],number_of_vg_points, wait_time, 1)

        aux_1, aux_2, aux_3, _ = res
        
        res_1 = aux_1*sensitivity_1/10.0
        res_2 = aux_2*sensitivity_2/10.0/meas_voltage_gain
        res_3 = aux_3*sensitivity_3/10.0/meas_voltage_gain

        # Calculate resistance
        res_5 = np.float64(1.0)*res_2/res_1
        res_6 = np.float64(1.0)*res_3/res_1
        
        # Calculate conductance
        res_7 = np.float64(1.0)/res_5
        res_8 = np.float64(1.0)/res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])
    
    elif voltage_source == "SIM":
        res = sim_sweep(voltage_channel, start_v, end_v, number_of_vg_points, wait_time/1e6)
        
#         res = sim_sweep2(dc_channel_0, [adc_channel_0, adc_channel_1,adc_channel_2,adc_channel_3], start_v, end_v, number_of_vg_points, wait_time/1e6)

        res_1, res_2, res_3 = res

        # Calculate resistance
        res_5 = np.float64(1.0)*res_2/res_1/meas_voltage_gain
        res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain
        
        # Calculate conductance
        res_7 = np.float64(1.0)/res_5
        res_8 = np.float64(1.0)/res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])
    
    
    
def scan_Vg_one(voltage_source, meas_voltage_gain, amplitude, voltage_channel, start_v, end_v, number_of_vg_points, wait_time):
#     time_constant_1 = LA1.time_constant()
#     sensitivity_1 = LA1.sensitivity()

#     time_constant_2 = LA2.time_constant()
#     sensitivity_2 = LA2.sensitivity()

#     time_constant_3 = LA3.time_constant()
#     sensitivity_3 = LA3.sensitivity()
    #vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
    vg = np.linspace(start_v, end_v, number_of_vg_points)   
    
    print '\r','Scanning Vg:  Start_V:', start_v, ' V; End_V:', end_v, ' V',
    if voltage_source == "DAC":
        res = DAC.buffer_ramp([voltage_channel], [adc_channel_0, adc_channel_1, adc_channel_2,adc_channel_3],[start_v],[end_v],number_of_vg_points, wait_time, 1)

        aux_1, aux_2, aux_3, _ = res
        
        res_1 = aux_1*sensitivity_1/10.0
        res_2 = aux_2*sensitivity_2/10.0/meas_voltage_gain
        res_3 = aux_3*sensitivity_3/10.0/meas_voltage_gain

        # Calculate resistance
        res_5 = np.float64(1.0)*res_2/res_1
        res_6 = np.float64(1.0)*res_3/res_1
        
        # Calculate conductance
        res_7 = np.float64(1.0)/res_5
        res_8 = np.float64(1.0)/res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])
    
    elif voltage_source == "SIM":
        res = sim_sweep(voltage_channel, start_v, end_v, number_of_vg_points, wait_time/1e6)
        
#         res = sim_sweep2(dc_channel_0, [adc_channel_0, adc_channel_1,adc_channel_2,adc_channel_3], start_v, end_v, number_of_vg_points, wait_time/1e6)

        res_1, res_2, res_3 = res
    
        current = amplitude/1.0e8

        # Calculate resistance
        res_5 = np.float64(1.0)*res_1/current/meas_voltage_gain
        res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain
        
        # Calculate conductance
        res_7 = np.float64(1.0)/res_5
        res_8 = np.float64(1.0)/res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])
    
    
def scan_Vg_dual(voltage_source, meas_voltage_gain, out_ch_bottom, out_ch_top, vbg_start, vbg_end, vtg_start, vtg_end, points_vbg, points_vtg, wait_time):
#     time_constant_1 = LA1.time_constant()
#     sensitivity_1 = LA1.sensitivity()

#     time_constant_2 = LA2.time_constant()
#     sensitivity_2 = LA2.sensitivity()

#     time_constant_3 = LA3.time_constant()
#     sensitivity_3 = LA3.sensitivity()
    #vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
    #vg = np.linspace(start_v, end_v, number_of_vg_points)   
    
    #print '\r','Scanning Vg:  Start_V:', start_v, ' V; End_V:', end_v, ' V',
    if voltage_source == "DAC":
        res = DAC.buffer_ramp([voltage_channel], [adc_channel_0, adc_channel_1, adc_channel_2,adc_channel_3],[start_v],[end_v],number_of_vg_points, wait_time, 1)

        aux_1, aux_2, aux_3, _ = res
        
        res_1 = aux_1*sensitivity_1/10.0
        res_2 = aux_2*sensitivity_2/10.0/meas_voltage_gain
        res_3 = aux_3*sensitivity_3/10.0/meas_voltage_gain

        # Calculate resistance
        res_5 = np.float64(1.0)*res_2/res_1
        res_6 = np.float64(1.0)*res_3/res_1
        
        # Calculate conductance
        res_7 = np.float64(1.0)/res_5
        res_8 = np.float64(1.0)/res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])
    
    elif voltage_source == "SIM":
        res = sim_dual_sweep(out_ch_bottom, out_ch_top, vbg_start, vbg_end, vtg_start, vtg_end, points_vbg, points_vtg, wait_time/1e6)
        
#         res = sim_sweep2(dc_channel_0, [adc_channel_0, adc_channel_1,adc_channel_2,adc_channel_3], start_v, end_v, number_of_vg_points, wait_time/1e6)

        res_1, res_2, res_3 = res

        # Calculate resistance
        res_5 = np.float64(1.0)*res_2/res_1/meas_voltage_gain
        res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain
        
        # Calculate conductance
        res_7 = np.float64(1.0)/res_5
        res_8 = np.float64(1.0)/res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])

    
    
def scan_Vg_dual_one(voltage_source, meas_voltage_gain, amplitude, out_ch_bottom, out_ch_top, vbg_start, vbg_end, vtg_start, vtg_end, points_vbg, points_vtg, wait_time):
#     time_constant_1 = LA1.time_constant()
#     sensitivity_1 = LA1.sensitivity()

#     time_constant_2 = LA2.time_constant()
#     sensitivity_2 = LA2.sensitivity()

#     time_constant_3 = LA3.time_constant()
#     sensitivity_3 = LA3.sensitivity()
    #vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
    #vg = np.linspace(start_v, end_v, number_of_vg_points)   
    
    #print '\r','Scanning Vg:  Start_V:', start_v, ' V; End_V:', end_v, ' V',
    if voltage_source == "DAC":
        res = DAC.buffer_ramp([voltage_channel], [adc_channel_0, adc_channel_1, adc_channel_2,adc_channel_3],[start_v],[end_v],number_of_vg_points, wait_time, 1)

        aux_1, aux_2, aux_3, _ = res
        
        res_1 = aux_1*sensitivity_1/10.0
        res_2 = aux_2*sensitivity_2/10.0/meas_voltage_gain
        res_3 = aux_3*sensitivity_3/10.0/meas_voltage_gain

        # Calculate resistance
        res_5 = np.float64(1.0)*res_2/res_1
        res_6 = np.float64(1.0)*res_3/res_1
        
        # Calculate conductance
        res_7 = np.float64(1.0)/res_5
        res_8 = np.float64(1.0)/res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])
    
    elif voltage_source == "SIM":
        res = sim_dual_sweep(out_ch_bottom, out_ch_top, vbg_start, vbg_end, vtg_start, vtg_end, points_vbg, points_vtg, wait_time/1e6)
        
#         res = sim_sweep2(dc_channel_0, [adc_channel_0, adc_channel_1,adc_channel_2,adc_channel_3], start_v, end_v, number_of_vg_points, wait_time/1e6)

        res_1, res_2, res_3 = res
    
        current = amplitude/1.0e8

        # Calculate resistance
        res_5 = np.float64(1.0)*res_1/current/meas_voltage_gain
        res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain
        
        # Calculate conductance
        res_7 = np.float64(1.0)/res_5
        res_8 = np.float64(1.0)/res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])

    
def scan_Vg_2omega(voltage_source, meas_voltage_gain, dc_channel_0, start_v, end_v, number_of_vg_points, wait_time):
#     time_constant_1 = LA1.time_constant()
#     sensitivity_1 = LA1.sensitivity()

#     time_constant_2 = LA2.time_constant()
#     sensitivity_2 = LA2.sensitivity()

#     time_constant_3 = LA3.time_constant()
#     sensitivity_3 = LA3.sensitivity()
    #vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
    vg = np.linspace(start_v, end_v, number_of_vg_points)   
    
    print '\r','Scanning Vg:  Start_V:', start_v, ' V; End_V:', end_v, ' V',
    if voltage_source == "DAC":
        res = DAC.buffer_ramp([dc_channel_0], [adc_channel_0, adc_channel_1, adc_channel_2,adc_channel_3],[start_v],[end_v],number_of_vg_points, wait_time, 1)

        aux_1, aux_2, aux_3, _ = res
        
        res_1 = aux_1*sensitivity_1/10.0
        res_2 = aux_2*sensitivity_2/10.0/meas_voltage_gain
        res_3 = aux_3*sensitivity_3/10.0/meas_voltage_gain

        # Calculate resistance
        res_5 = np.float64(1.0)*res_2/res_1
        res_6 = np.float64(1.0)*res_3/res_1
        
        # Calculate conductance
        res_7 = np.float64(1.0)/res_5
        res_8 = np.float64(1.0)/res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])
    
    elif voltage_source == "SIM":
        res = sim_sweep_2omega(dc_channel_0, start_v, end_v, number_of_vg_points, wait_time/4/1e6)
        
#         res = sim_sweep2(dc_channel_0, [adc_channel_0, adc_channel_1,adc_channel_2,adc_channel_3], start_v, end_v, number_of_vg_points, wait_time/1e6)

        res_1, res_2, res_2b, res_3, res_3b = res

        # Calculate resistance
        res_5 = np.float64(1.0)*res_2/res_1/meas_voltage_gain
        res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain
        res_5b = np.float64(1.0)*res_2b/res_1/meas_voltage_gain
        res_6b = np.float64(1.0)*res_3b/res_1/meas_voltage_gain       
        # Calculate conductance
        res_7 = np.float64(1.0)/res_5
        res_8 = np.float64(1.0)/res_6

        return np.array([res_1, res_2, res_2b, res_3, res_3b, res_5, res_5b, res_6, res_6b, res_7, res_8])


def set_Bz(MGz, target_bz):
    #MGz.conf_ramp_rate_field(1, 0.0606, 9.0)
    MGz.conf_field_targ(float(target_bz))
    MGz.ramp()
    print '\r',"Ramping magnetic field to ",target_bz," T",
            
    flag = True
    while flag:
        try:
            actual_field = float(MGz.get_field_mag())
            flag = False
        except:
            flag = True

    while abs(float(target_bz)- float(MGz.get_field_mag())) >1.0e-4:
        continue

    time.sleep(1)
    print '\r',"Magnetic field reached: ",target_bz," T",
    return float(MGz.get_field_mag())


def set_BxByBz(MGx, MGy, MGz, target_bx, target_by, target_bz):
    MGz.conf_ramp_rate_field(1, 0.0606, 9.0)
    #MG.conf_ramp_rate_field(1, 0.0806, 12.0)
    time.sleep(0.2)
    MGz.conf_field_targ(float(target_bz))
    MGz.ramp()
    time.sleep(0.2)
    MGy.conf_field_targ(float(target_by))
    MGy.ramp()
    time.sleep(0.2)
    MGx.conf_field_targ(float(target_bx))
    MGx.ramp()
    time.sleep(0.2)
    print '\r',"Ramping magnetic field (Bx, By, Bz): ",target_bx, target_by, target_bz, " T",
            
    flag = True
    while flag:
        try:
            actual_fieldz = float(MGz.get_field_mag())
            time.sleep(0.2)
            actual_fieldy = float(MGy.get_field_mag())
            time.sleep(0.2)
            actual_fieldx = float(MGx.get_field_mag())
            time.sleep(0.2)
            flag = False
        except:
            flag = True

    while abs(float(target_bz)- actual_fieldz) >1.0e-4 or abs(float(target_by)- actual_fieldy) >1.0e-4 or abs(float(target_bx)- actual_fieldx) >1.0e-4:
        actual_fieldz = float(MGz.get_field_mag())
        time.sleep(0.2)
        actual_fieldy = float(MGy.get_field_mag())
        time.sleep(0.2)
        actual_fieldx = float(MGx.get_field_mag())
        time.sleep(0.2)
        print '\r',"current magnetic field (Bx, By, Bz): ",actual_fieldx, actual_fieldy, actual_fieldz, " T", "target magnetic field: (Bx, By, Bz): ",target_bx, target_by, target_bz, " T", 

    time.sleep(0.5)
    print '\r',"Magnetic field reached: (Bx, By, Bz): ",target_bx, target_by, target_bz, " T", 
    return float(MGx.get_field_mag()), float(MGy.get_field_mag()), float(MGz.get_field_mag())

  
    
def sweep_Bz(b_start, b_end, points):
    bs = np.linspace(b_start, b_end, num = points)
    d_tmp = None
    p1, p2, p3 = 0, 0, 0

    set_Bz(MGz, b_start)
    #time.sleep(180)
    for jj in range(points):
        set_Bz(MGz, bs[jj])
        #time.sleep(delay*0.9)
        try:
        #line_data = [DAC.read_voltage(k) for k in in_dac_ch]
            p1, p2, p3 = LA1.x(), 1.0, 1.0
            line_data = [p1, p2, p3]
        except:
            line_data = [p1, p2, p3]
            
        if d_tmp is not None:
            d_tmp = np.vstack([d_tmp, line_data])
        else:
            d_tmp = line_data

    return  d_tmp.T


def scan_B_dual_one(meas_voltage_gain, amplitude, b_start, b_end, points_b):

    res = sweep_Bz(b_start = b_start, b_end = b_end, points = points_b)

    res_1, res_2, res_3 = res

    current = amplitude/1.0e8

    # Calculate resistance
    res_5 = np.float64(1.0)*res_1/current/meas_voltage_gain
    res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain

    # Calculate conductance
    res_7 = np.float64(1.0)/res_5
    res_8 = np.float64(1.0)/res_6

    return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])
    


def read_T():
    reconnected = False
    while not reconnected:
        try:
            time.sleep(1.0)
            print '\r', "Connecting to temperature server ..."
            #cxn4 = labrad.connect('evaporator-PC', 7682, password='pass')
            cxn4 = labrad.connect()
            tc = cxn4.lakeshore_372
            tc.select_device()
            Tmc, T_p = tc.mc(), tc.probe()
            print '\r', 'MXC: ', Tmc, 'probe: ', T_p
            time.sleep(1.0)
            print '\r', 'Reconnected successfully',
            return Tmc, T_p
        except Exception as e:
            print '\r', str(e),
            print '\r', 'Could not reconnect to temperature server',
            time.sleep(2.0)
                    
                    
def set_T(setpoint):
    setpoint_updated = False
    while not setpoint_updated:
        try:
            expression = "SETP 0, %03.2E\n"%setpoint
            #print '\r', expression
            tc.write(expression)  # set the power for the current zone
            Tmc, T_p = tc.mc(), tc.probe()
            print '\r', 'MXC: ', Tmc, 'probe: ', T_p,
            if setpoint < 1.0:
                temperature_error = min(setpoint*0.03, 0.01)
            elif setpoint > 6.5 and setpoint < 10.0:
                temperature_error = 0.3
            elif setpoint > 10.0: temperature_error = 1.0 
            else: temperature_error = 0.1
            

            while abs(Tmc - setpoint) > temperature_error:
                time.sleep(2.0)
                Tmc, T_p = tc.mc(), tc.probe()
                print '\r', 'current MXC: ', Tmc, 'current probe: ', T_p,
                
            print '\r', 'Target temperature reached.',
            return Tmc, T_p
            setpoint_updated = True

        except Exception as e:
            print '\r', str(e),
            print '\r', 'Failed to update setpoint',
            reconnected = False
            while not reconnected:
                try:
                    print '\r', "Connecting to temperature server ...",
                    #cxn4 = labrad.connect('evaporator-PC', 7682, password='pass')
                    cxn4 = labrad.connect()
                    tc = cxn4.lakeshore_372
                    tc.select_device()
                    Tmc, T_p = tc.mc(), tc.probe()
                    print '\r', 'MXC: ', Tmc, 'probe: ', T_p,
                    print '\r', 'Reconnected successfully',
                    reconnected = True
                except Exception as e:
                    print '\r', str(e),
                    print '\r', 'Could not reconnect to temperature server',
                    time.sleep(2.0)
                    
    
    
                    
def scan_R_vs_Vg_Bz(
    file_path,
    voltage_source,
    voltage_channel,
    amplitude = 0.01,
    frequency = 17.777,
    gate_gain = 1.0,
    meas_voltage_gain = 1.0,
    bz_range = [0.0, 0.0],
    vg_range = [-1.0, 1.0],
    number_of_bz_lines = 1,
    number_of_vg_points = 200,
    wait_time = wait_time,
    misc = "misc"
):
    
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg_ind', 'Vg', 'Bz_ind', 'Bz', 'Bz_ac', 'Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    b_lines = np.linspace(bz_range[0], bz_range[1], number_of_bz_lines)
    
    t_mc0, t_p0 = 0, 0
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg_range[0])

    for ind, val in enumerate(b_lines, 1):
        actual_B = set_Bz(MGz, val)
        print '\r',"Field Line:", ind, "out of ", number_of_bz_lines
        
        vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
        vg = gate_gain* np.linspace(vg_range[0], vg_range[1], number_of_vg_points)  
        b_ind = np.linspace(ind, ind, number_of_vg_points)
        b_val = val * np.ones(number_of_vg_points)
        b_ac = actual_B * np.ones(number_of_vg_points)
        t_mc = t_mc0 * np.ones(number_of_vg_points)
        t_p = t_p0 * np.ones(number_of_vg_points)
        
        data1 = np.array([vg_ind, vg, b_ind, b_val, b_ac, t_mc, t_p])
        # Scan Vg and acquire data
        data2 = scan_Vg(voltage_source, meas_voltage_gain, voltage_channel, vg_range[0], vg_range[1], number_of_vg_points, wait_time)
        
        data = np.vstack((data1, data2))
        DV.add(data.T)
        
        plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "Vg", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 0, 1000, 0, 1000], xname = "Vg", yname = ['Ix', 'R1', 'R2'], logy = [False, False, False], var = 0, unit = "T")
        # go to next gate voltage
        if ind < number_of_bz_lines: set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], vg_range[0])
        

    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
    

    
def scan_R_vs_n_Bz_at_fixedD(
    file_path,
    voltage_source,
    voltage_channel_bottom,
    voltage_channel_top,
    amplitude = 0.01,
    frequency = 17.777,
    gate_gain = 1.0,
    meas_voltage_gain = 1.0,
    displacement_field = 0.0,
    bz_range = [0.0, 0.0],
    n_range = [-1.0, 1.0],
    number_of_bz_lines = 1,
    number_of_n_points = 200,
    wait_time = wait_time,
    c_delta = 0.0,
    misc = "misc"
):
    
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('n_ind', 'n', 'Bz_ind', 'iBz', 'Bz','Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    b_lines = np.linspace(bz_range[0], bz_range[1], number_of_bz_lines)
    
    t_mc0, t_p0 = 0, 0
    D_val = displacement_field
    ##### Measurements start #####
    # go to initial gate volatge
    vtg_last, vbg_last = 0, 0
    for ind, val in enumerate(b_lines, 1):
        d3 = datetime.datetime.now()
        actual_B = set_Bz(MGz, val)
        print '\r',"Field Line:", ind, "out of ", number_of_bz_lines
        vtg_s, vbg_s = find_vt_vb(D_val, n_range[0], c_delta)
        vtg_e, vbg_e = find_vt_vb(D_val, n_range[1], c_delta)
        sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                       out_ch_top = voltage_channel_top, 
                       vbg_start = vbg_last, vbg_end = vbg_s, 
                       vtg_start = vtg_last, vtg_end = vtg_s, 
                       points_vbg = 40, points_vtg = 40, delay = 0.005)        
        n_ind = np.linspace(1, number_of_n_points, number_of_n_points)
        n = gate_gain* np.linspace(n_range[0], n_range[1], number_of_n_points)  
        b_ind = np.linspace(ind, ind, number_of_n_points)
        b_val = val * np.ones(number_of_n_points)
        b_ac = actual_B * np.ones(number_of_n_points)
        ib_val = np.float64(1.0/actual_B) * np.ones(number_of_n_points)
        t_mc = t_mc0 * np.ones(number_of_n_points)
        t_p = t_p0 * np.ones(number_of_n_points)
        
        data1 = np.array([n_ind, n, b_ind, ib_val, b_val, t_mc, t_p])
        # Scan Vg and acquire data
        
        data2 = scan_Vg_dual_one(voltage_source, meas_voltage_gain,
                                 amplitude = amplitude,
                             out_ch_bottom = voltage_channel_bottom, 
                             out_ch_top = voltage_channel_top, 
                             vbg_start = vbg_s, vbg_end = vbg_e,
                             vtg_start = vtg_s, vtg_end = vtg_e, 
                             points_vbg = number_of_n_points, 
                             points_vtg = number_of_n_points, 
                             wait_time = wait_time)
        t = datetime.datetime.now()-d3
        print '\r', "one epoch time:", t, "estimated finish time:", datetime.datetime.now()+(len(b_lines)-ind)*t 
        vtg_last, vbg_last = vtg_e, vbg_e
        data = np.vstack((data1, data2))
        DV.add(data.T)
        
    
        
        plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "n", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 100, 100000, 0, 1000], xname = "n", yname = ['Ix', 'R1', 'R2'], logy = [False, True, False], var = 0, unit = "T")
        
        #go to next gate voltage
        
    # go to 0 V
    sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                   out_ch_top = voltage_channel_top, 
                   vbg_start = vbg_last, vbg_end = 0.0, 
                   vtg_start = vtg_last, vtg_end = 0.0, 
                   points_vbg = 40, points_vtg = 40, delay = 0.005)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
    
def scan_R_vs_n_Bz_at_fixedD_diagonal(
    file_path,
    voltage_source,
    voltage_channel_bottom,
    voltage_channel_top,
    amplitude = 0.01,
    frequency = 17.777,
    gate_gain = 1.0,
    meas_voltage_gain = 1.0,
    displacement_field = [0.0,0.0],
    bz_range = [0.0, 0.0],
    n_range = [-1.0, 1.0],
    number_of_bz_lines = 1,
    number_of_n_points = 200,
    wait_time = wait_time,
    c_delta = 0.0,
    misc = "misc"
):
    
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('n_ind', 'n', 'Bz_ind', 'iBz', 'Bz','Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    b_lines = np.linspace(bz_range[0], bz_range[1], number_of_bz_lines)
    
    t_mc0, t_p0 = 0, 0
    D_val = displacement_field
    ##### Measurements start #####
    # go to initial gate volatge
    vtg_last, vbg_last = 0, 0
    for ind, val in enumerate(b_lines, 1):
        d3 = datetime.datetime.now()
        actual_B = set_Bz(MGz, val)
        print '\r',"Field Line:", ind, "out of ", number_of_bz_lines
        vtg_s, vbg_s = find_vt_vb(D_val[0], n_range[0], c_delta)
        vtg_e, vbg_e = find_vt_vb(D_val[1], n_range[1], c_delta)
        sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                       out_ch_top = voltage_channel_top, 
                       vbg_start = vbg_last, vbg_end = vbg_s, 
                       vtg_start = vtg_last, vtg_end = vtg_s, 
                       points_vbg = 40, points_vtg = 40, delay = 0.005)        
        n_ind = np.linspace(1, number_of_n_points, number_of_n_points)
        n = gate_gain* np.linspace(n_range[0], n_range[1], number_of_n_points)  
        b_ind = np.linspace(ind, ind, number_of_n_points)
        b_val = val * np.ones(number_of_n_points)
        b_ac = actual_B * np.ones(number_of_n_points)
        ib_val = np.float64(1.0/actual_B) * np.ones(number_of_n_points)
        t_mc = t_mc0 * np.ones(number_of_n_points)
        t_p = t_p0 * np.ones(number_of_n_points)
        
        data1 = np.array([n_ind, n, b_ind, ib_val, b_val, t_mc, t_p])
        # Scan Vg and acquire data
        
        data2 = scan_Vg_dual_one(voltage_source, meas_voltage_gain,
                                 amplitude = amplitude,
                             out_ch_bottom = voltage_channel_bottom, 
                             out_ch_top = voltage_channel_top, 
                             vbg_start = vbg_s, vbg_end = vbg_e,
                             vtg_start = vtg_s, vtg_end = vtg_e, 
                             points_vbg = number_of_n_points, 
                             points_vtg = number_of_n_points, 
                             wait_time = wait_time)
        t = datetime.datetime.now()-d3
        print '\r', "one epoch time:", t, "estimated finish time:", datetime.datetime.now()+(len(b_lines)-ind)*t 
        vtg_last, vbg_last = vtg_e, vbg_e
        data = np.vstack((data1, data2))
        DV.add(data.T)
        
    
        
#         plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "n", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 0, 15000, 0, 1000], xname = "n", yname = ['Ix', 'R1', 'R2'], logy = [False, False, False], var = 0, unit = "T")
        
        #go to next gate voltage
        
    # go to 0 V
    sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                   out_ch_top = voltage_channel_top, 
                   vbg_start = vbg_last, vbg_end = 0.0, 
                   vtg_start = vtg_last, vtg_end = 0.0, 
                   points_vbg = 40, points_vtg = 40, delay = 0.005)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
def scan_R_vs_n_invBz_Bz_mixed_at_fixedD(
    file_path,
    voltage_source,
    voltage_channel_bottom,
    voltage_channel_top,
    amplitude = 0.01,
    frequency = 17.777,
    gate_gain = 1.0,
    meas_voltage_gain = 1.0,
    displacement_field = 0.0,
    ibz_range = [10, 10],
    bz_range = [0, 0],
    n_range = [-1.0, 1.0],
    number_of_ibz_lines = 1,
    number_of_bz_lines = 1,
    number_of_n_points = 200,
    wait_time = wait_time,
    c_delta = 0.0,
    reverse = False,
    misc = "misc"
):
    
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('n_ind', 'n', 'iBz_ind', 'iBz', 'Bz','Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    ib_lines = np.linspace(ibz_range[0], ibz_range[1], number_of_ibz_lines)
    b_lines = np.linspace(bz_range[0], bz_range[1], number_of_bz_lines)
    
    mixed_lines = sorted(np.append(b_lines, 1.0/ib_lines), reverse = reverse)
    t_mc0, t_p0 = 0, 0
    D_val = displacement_field
    ##### Measurements start #####
    # go to initial gate volatge
    vtg_last, vbg_last = 0, 0
    for ind, val in enumerate(mixed_lines, 1):
        d3 = datetime.datetime.now()
        actual_B = set_Bz(MGz, val)
        print '\r',"Field Line:", ind, "out of ", len(mixed_lines)
        vtg_s, vbg_s = find_vt_vb(D_val, n_range[0], c_delta)
        vtg_e, vbg_e = find_vt_vb(D_val, n_range[1], c_delta)
        sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                       out_ch_top = voltage_channel_top, 
                       vbg_start = vbg_last, vbg_end = vbg_s, 
                       vtg_start = vtg_last, vtg_end = vtg_s, 
                       points_vbg = 40, points_vtg = 40, delay = 0.005)        
        n_ind = np.linspace(1, number_of_n_points, number_of_n_points)
        n = gate_gain* np.linspace(n_range[0], n_range[1], number_of_n_points)  
        ib_ind = np.linspace(ind, ind, number_of_n_points)
        if abs(val) <= 0.001:
            rval = 1000
        else:
            rval = 1.0/val
        ib_val = rval * np.ones(number_of_n_points)
        b_ac = actual_B * np.ones(number_of_n_points)
        t_mc = t_mc0 * np.ones(number_of_n_points)
        t_p = t_p0 * np.ones(number_of_n_points)
        
        data1 = np.array([n_ind, n, ib_ind, ib_val, b_ac, t_mc, t_p])
        # Scan Vg and acquire data
        
        data2 = scan_Vg_dual_one(voltage_source, meas_voltage_gain,
                                 amplitude = amplitude,
                             out_ch_bottom = voltage_channel_bottom, 
                             out_ch_top = voltage_channel_top, 
                             vbg_start = vbg_s, vbg_end = vbg_e,
                             vtg_start = vtg_s, vtg_end = vtg_e, 
                             points_vbg = number_of_n_points, 
                             points_vtg = number_of_n_points, 
                             wait_time = wait_time)
        vtg_last, vbg_last = vtg_e, vbg_e
        data = np.vstack((data1, data2))
        DV.add(data.T)
        t = datetime.datetime.now()-d3
        print '\r', "one epoch time:", t, "estimated finish time:", datetime.datetime.now()+(len(mixed_lines)-ind)*t 
        
    
        
#         plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "n", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 0, 1000, 0, 1000], xname = "n", yname = ['Ix', 'R1', 'R2'], logy = [False, False, False], var = 0, unit = "T")
        # go to next gate voltage
        
    # go to 0 V
    sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                   out_ch_top = voltage_channel_top, 
                   vbg_start = vbg_last, vbg_end = 0.0, 
                   vtg_start = vtg_last, vtg_end = 0.0, 
                   points_vbg = 40, points_vtg = 40, delay = 0.005)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
    
def scan_R_vs_n_invBz2_at_fixedD(
    file_path,
    voltage_source,
    voltage_channel_bottom,
    voltage_channel_top,
    amplitude = 0.01,
    frequency = 17.777,
    gate_gain = 1.0,
    meas_voltage_gain = 1.0,
    displacement_field = 0.0,
    ibz_range = [10, 10],
    n_range = [-1.0, 1.0],
    number_of_ibz_lines = 1,
    number_of_n_points = 200,
    wait_time = wait_time,
    c_delta = 0.0,
    misc = "misc"
):
    
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('n_ind', 'n', 'iBz_ind', 'iBz', 'Bz','Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    ib_lines = np.linspace(ibz_range[0], ibz_range[1], number_of_ibz_lines)

    t_mc0, t_p0 = 0, 0
    D_val = displacement_field
    ##### Measurements start #####
    # go to initial gate volatge
    vtg_last, vbg_last = 0, 0
    for ind, val in enumerate(ib_lines, 1):
        d3 = datetime.datetime.now()
        actual_B = set_Bz(MGz, 1.0/val)
        print '\r',"Field Line:", ind, "out of ", len(ib_lines)
        vtg_s, vbg_s = find_vt_vb(D_val, n_range[0], c_delta)
        vtg_e, vbg_e = find_vt_vb(D_val, n_range[1], c_delta)
        sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                       out_ch_top = voltage_channel_top, 
                       vbg_start = vbg_last, vbg_end = vbg_s, 
                       vtg_start = vtg_last, vtg_end = vtg_s, 
                       points_vbg = 40, points_vtg = 40, delay = 0.005)        
        n_ind = np.linspace(1, number_of_n_points, number_of_n_points)
        n = gate_gain* np.linspace(n_range[0], n_range[1], number_of_n_points)  
        ib_ind = np.linspace(ind, ind, number_of_n_points)
        ib_val = val * np.ones(number_of_n_points)
        b_ac = actual_B * np.ones(number_of_n_points)
        t_mc = t_mc0 * np.ones(number_of_n_points)
        t_p = t_p0 * np.ones(number_of_n_points)
        
        data1 = np.array([n_ind, n, ib_ind, ib_val, b_ac, t_mc, t_p])
        # Scan Vg and acquire data
        
        data2 = scan_Vg_dual_one(voltage_source, meas_voltage_gain,
                                 amplitude = amplitude,
                             out_ch_bottom = voltage_channel_bottom, 
                             out_ch_top = voltage_channel_top, 
                             vbg_start = vbg_s, vbg_end = vbg_e,
                             vtg_start = vtg_s, vtg_end = vtg_e, 
                             points_vbg = number_of_n_points, 
                             points_vtg = number_of_n_points, 
                             wait_time = wait_time)
        t = datetime.datetime.now()-d3
        print '\r', "one epoch time:", t, "estimated finish time:", datetime.datetime.now()+(len(ib_lines)-ind)*t 
        vtg_last, vbg_last = vtg_e, vbg_e
        data = np.vstack((data1, data2))
        DV.add(data.T)
        
    
        
#         plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "n", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 0, 1000, 0, 1000], xname = "n", yname = ['Ix', 'R1', 'R2'], logy = [False, False, False], var = 0, unit = "T")
        # go to next gate voltage
        
    # go to 0 V
    sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                   out_ch_top = voltage_channel_top, 
                   vbg_start = vbg_last, vbg_end = 0.0, 
                   vtg_start = vtg_last, vtg_end = 0.0, 
                   points_vbg = 40, points_vtg = 40, delay = 0.005)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    

def scan_R_vs_n_D(misc, file_path, amplitude, frequency, gate_gain, meas_voltage_gain, voltage_source, voltage_channel_bottom, voltage_channel_top, n_range, D_range, number_of_D_lines, number_of_n_points, c_delta, wait_time):
    
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('n_ind', 'n', 'D_ind', 'D', 'Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)
    DV.add_parameter('live_plots', [('n', 'D', 'R1')])

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    D_lines = np.linspace(D_range[0], D_range[1], number_of_D_lines)
    
    t_mc0, t_p0 = 0, 0
#     vt_start, vg_start = find_vt_vb(D_range[0], n_range[0], c_delta)
#     vt_end, vg_end = find_vt_vb(D_range[1], n_range[1], c_delta)
    ##### Measurements start #####
    # go to initial gate volatge
#     set_Vg_nodac(voltage_source, voltage_channel_bottom, 0.0, vg_start)
    vtg_last, vbg_last = 0, 0
    for ind, D_val in enumerate(D_lines):
        d3 = datetime.datetime.now()
        vtg_s, vbg_s = find_vt_vb(D_val, n_range[0], c_delta)
        vtg_e, vbg_e = find_vt_vb(D_val, n_range[1], c_delta)
        sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                       out_ch_top = voltage_channel_top, 
                       vbg_start = vbg_last, vbg_end = vbg_s, 
                       vtg_start = vtg_last, vtg_end = vtg_s, 
                       points_vbg = 40, points_vtg = 40, delay = 0.005)    
#         if ind == 0: set_Vg_nodac(voltage_source, voltage_channel_top, 0.0, vt_start)
#         else: set_Vg_nodac(voltage_source, voltage_channel_top, vgt_lines[ind-1], val)
        print '\r',"D Line:", ind + 1, "out of ", number_of_D_lines
        
        n_ind = np.linspace(1, number_of_n_points, number_of_n_points)
        n = gate_gain* np.linspace(n_range[0], n_range[1], number_of_n_points)  
        D_ind = np.linspace(ind, ind, number_of_n_points)
        D = D_val * np.ones(number_of_n_points)
        t_mc = t_mc0 * np.ones(number_of_n_points)
        t_p = t_p0 * np.ones(number_of_n_points)
        
        data1 = np.array([n_ind, n, D_ind, D, t_mc, t_p])
        # Scan Vg and acquire data
        time.sleep(1.0)
        data2 = scan_Vg_dual_one(voltage_source, meas_voltage_gain,
                                 amplitude = amplitude,
                             out_ch_bottom = voltage_channel_bottom, 
                             out_ch_top = voltage_channel_top, 
                             vbg_start = vbg_s, vbg_end = vbg_e,
                             vtg_start = vtg_s, vtg_end = vtg_e, 
                             points_vbg = number_of_n_points, 
                             points_vtg = number_of_n_points, 
                             wait_time = wait_time)
        vtg_last, vbg_last = vtg_e, vbg_e
        
        t = datetime.datetime.now()-d3
        print '\r', "one epoch time:", t, "estimated finish time:", datetime.datetime.now()+(len(D_lines)-ind)*t 
        
        data = np.vstack((data1, data2))
        DV.add(data.T)
        
        
#         plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "n", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 10, 10000, 0, 4000], xname = "n", yname = ['Ix', 'R1', 'R2'], logy = [False, True, False], var = 0, unit = "T")
        # go to next gate voltage
#         if ind < number_of_vgt_lines: 
#             set_Vg_nodac(voltage_source, voltage_channel_bottom, vgb_range[1], vgb_range[0])
    # go to 0 V
    sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                   out_ch_top = voltage_channel_top, 
                   vbg_start = vbg_last, vbg_end = 0.0, 
                   vtg_start = vtg_last, vtg_end = 0.0, 
                   points_vbg = 40, points_vtg = 40, delay = 0.005)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    

def scan_R_vs_n_D_fixedBz(misc, file_path, amplitude, frequency, gate_gain, meas_voltage_gain, voltage_source, voltage_channel_bottom, voltage_channel_top, magnetic_field, n_range, D_range, number_of_D_lines, number_of_n_points, c_delta, wait_time):
    
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('n_ind', 'n', 'D_ind', 'D', 'Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)
    DV.add_parameter('live_plots', [('n', 'D', 'R1')])

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    D_lines = np.linspace(D_range[0], D_range[1], number_of_D_lines)
    
    t_mc0, t_p0 = 0, 0
#     vt_start, vg_start = find_vt_vb(D_range[0], n_range[0], c_delta)
#     vt_end, vg_end = find_vt_vb(D_range[1], n_range[1], c_delta)
    ##### Measurements start #####
    # go to initial gate volatge
#     set_Vg_nodac(voltage_source, voltage_channel_bottom, 0.0, vg_start)
    set_Bz(MGz, magnetic_field)
    vtg_last, vbg_last = 0, 0
    for ind, D_val in enumerate(D_lines):
        d3 = datetime.datetime.now()
        vtg_s, vbg_s = find_vt_vb(D_val, n_range[0], c_delta)
        vtg_e, vbg_e = find_vt_vb(D_val, n_range[1], c_delta)
        sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                       out_ch_top = voltage_channel_top, 
                       vbg_start = vbg_last, vbg_end = vbg_s, 
                       vtg_start = vtg_last, vtg_end = vtg_s, 
                       points_vbg = 40, points_vtg = 40, delay = 0.005)    
#         if ind == 0: set_Vg_nodac(voltage_source, voltage_channel_top, 0.0, vt_start)
#         else: set_Vg_nodac(voltage_source, voltage_channel_top, vgt_lines[ind-1], val)
        print '\r',"D Line:", ind + 1, "out of ", number_of_D_lines
        
        n_ind = np.linspace(1, number_of_n_points, number_of_n_points)
        n = gate_gain* np.linspace(n_range[0], n_range[1], number_of_n_points)  
        D_ind = np.linspace(ind, ind, number_of_n_points)
        D = D_val * np.ones(number_of_n_points)
        t_mc = t_mc0 * np.ones(number_of_n_points)
        t_p = t_p0 * np.ones(number_of_n_points)
        
        data1 = np.array([n_ind, n, D_ind, D, t_mc, t_p])
        # Scan Vg and acquire data
        
        data2 = scan_Vg_dual_one(voltage_source, meas_voltage_gain,
                                 amplitude = amplitude,
                             out_ch_bottom = voltage_channel_bottom, 
                             out_ch_top = voltage_channel_top, 
                             vbg_start = vbg_s, vbg_end = vbg_e,
                             vtg_start = vtg_s, vtg_end = vtg_e, 
                             points_vbg = number_of_n_points, 
                             points_vtg = number_of_n_points, 
                             wait_time = wait_time)
        vtg_last, vbg_last = vtg_e, vbg_e
        
        t = datetime.datetime.now()-d3
        print '\r', "one epoch time:", t, "estimated finish time:", datetime.datetime.now()+(len(D_lines)-ind)*t 
        
        data = np.vstack((data1, data2))
        DV.add(data.T)
        
        
#         plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "n", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 0, 1000, 0, 4000], xname = "n", yname = ['Ix', 'R1', 'R2'], logy = [False, False, False], var = 0, unit = "T")
        # go to next gate voltage
#         if ind < number_of_vgt_lines: 
#             set_Vg_nodac(voltage_source, voltage_channel_bottom, vgb_range[1], vgb_range[0])
    # go to 0 V
    
    sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                   out_ch_top = voltage_channel_top, 
                   vbg_start = vbg_last, vbg_end = 0.0, 
                   vtg_start = vtg_last, vtg_end = 0.0, 
                   points_vbg = 40, points_vtg = 40, delay = 0.005)
    set_Bz(MGz, 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
def scan_R_vs_dual_gates(misc, file_path, amplitude, frequency, gate_gain, meas_voltage_gain, voltage_source, voltage_channel_bottom, voltage_channel_top,vgt_range, vgb_range, number_of_vgt_lines, number_of_vgb_points, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vgb_ind', 'Vgb', 'Vgt_ind', 'Vgt', 'Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    vgt_lines = np.linspace(vgt_range[0], vgt_range[1], number_of_vgt_lines)
    
    t_mc0, t_p0 = 0, 0
    
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel_bottom, 0.0, vgb_range[0])

    for ind, val in enumerate(vgt_lines):
        if ind == 0: set_Vg_nodac(voltage_source, voltage_channel_top, 0.0, val)
        else: set_Vg_nodac(voltage_source, voltage_channel_top, vgt_lines[ind-1], val)
        print '\r',"Vgt Line:", ind + 1, "out of ", number_of_vgt_lines
        
        vgb_ind = np.linspace(1, number_of_vgb_points, number_of_vgb_points)
        vgb = gate_gain* np.linspace(vgb_range[0], vgb_range[1], number_of_vgb_points)  
        vgt_ind = np.linspace(ind, ind, number_of_vgb_points)
        vgt_val = val * np.ones(number_of_vgb_points)
        t_mc = t_mc0 * np.ones(number_of_vgb_points)
        t_p = t_p0 * np.ones(number_of_vgb_points)
        
        data1 = np.array([vgb_ind, vgb, vgt_ind, vgt_val, t_mc, t_p])
        # Scan Vg and acquire data
        data2 = scan_Vg(voltage_source, meas_voltage_gain, voltage_channel_bottom, vgb_range[0], vgb_range[1], number_of_vgb_points, wait_time/4)
        
        data = np.vstack((data1, data2))
        DV.add(data.T)
        
#         plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "Vgb", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 0, 4000, 0, 4000], xname = "Vg", yname = ['Ix', 'R1', 'R2'], logy = [False, False, False], var = 0, unit = "T")
        # go to next gate voltage
        if ind < number_of_vgt_lines: 
            set_Vg_nodac(voltage_source, voltage_channel_bottom, vgb_range[1], vgb_range[0])
        

    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel_bottom, vgb_range[1], 0.0)
    set_Vg_nodac(voltage_source, voltage_channel_top, vgt_range[1], 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
    

    

def scan_R_vs_Vg_theta(misc, file_path, amplitude, frequency, gate_gain, meas_voltage_gain, voltage_source, voltage_channel,theta_range, vg_range, number_of_theta_lines, number_of_vg_points, fixed_Bz,fixed_Br,wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg_ind', 'Vg', 'theta_ind', 'theta', 'Bx_ac', 'By_ac', 'Bz_ac', 'Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    theta_lines = np.linspace(theta_range[0], theta_range[1], number_of_theta_lines)
    rot_mat = calc_rotation_matrix(0,0,1,0,0,1)
    t_mc0, t_p0 = read_T()
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg_range[0])

    for ind, theta in enumerate(theta_lines, 1):
        x0, y0 = fixed_Br*np.cos(np.radians(theta)), fixed_Br*np.sin(np.radians(theta)) 
        x, y, z = rotate_vector(rot_mat, x0, y0, fixed_Bz)
        if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
            actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, x, y, z)
            print '\r',"Theta Line:", ind, "out of ", number_of_theta_lines

            vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
            vg = gate_gain* np.linspace(vg_range[0], vg_range[1], number_of_vg_points)  
            theta_ind = np.linspace(ind, ind, number_of_vg_points)
            theta_val = theta * np.ones(number_of_vg_points)
            bx_ac = actual_Bx * np.ones(number_of_vg_points)
            by_ac = actual_By * np.ones(number_of_vg_points)
            bz_ac = actual_Bz * np.ones(number_of_vg_points)
            t_mc = t_mc0 * np.ones(number_of_vg_points)
            t_p = t_p0 * np.ones(number_of_vg_points)

            data1 = np.array([vg_ind, vg, theta_ind, theta_val, bx_ac, by_ac, bz_ac,t_mc, t_p])
            # Scan Vg and acquire data
            
            data2 = scan_Vg(voltage_source, meas_voltage_gain, voltage_channel, vg_range[0], vg_range[1], number_of_vg_points, wait_time/4)

            data = np.vstack((data1, data2))
            DV.add(data.T)

            #plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "Vg", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 1, 50000, -1000, 1000], xname = "Vg", yname = ['Ix', 'R1', 'R2'], logy = [False, True, False], var = 0, unit = "degree")
            # go to next gate voltage
            if ind < number_of_theta_lines: set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], vg_range[0])
        

    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    

def scan_R_vs_Vg_theta2(misc, file_path, amplitude, frequency, gate_gain, meas_voltage_gain, voltage_source, voltage_channel,theta_range, vg_range, number_of_theta_lines, number_of_vg_points, fixed_Bz,fixed_Br,alpha,wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg_ind', 'Vg', 'theta_ind', 'theta', 'Bx_ac', 'By_ac', 'Bz_ac', 'Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    theta_lines = np.linspace(theta_range[0], theta_range[1], number_of_theta_lines)
    rot_mat = calc_rotation_matrix(0,0,1,-alpha*0.342,-alpha*0.9396,1)
    t_mc0, t_p0 = read_T()
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg_range[0])

    for ind, theta in enumerate(theta_lines, 1):
        x0, y0 = fixed_Br*np.cos(np.radians(theta)), fixed_Br*np.sin(np.radians(theta)) 
        x, y, z = rotate_vector(rot_mat, x0, y0, fixed_Bz)
        if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
            actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, x, y, z)
            print '\r',"Theta Line:", ind, "out of ", number_of_theta_lines

            vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
            vg = gate_gain* np.linspace(vg_range[0], vg_range[1], number_of_vg_points)  
            theta_ind = np.linspace(ind, ind, number_of_vg_points)
            theta_val = theta * np.ones(number_of_vg_points)
            bx_ac = actual_Bx * np.ones(number_of_vg_points)
            by_ac = actual_By * np.ones(number_of_vg_points)
            bz_ac = actual_Bz * np.ones(number_of_vg_points)
            t_mc = t_mc0 * np.ones(number_of_vg_points)
            t_p = t_p0 * np.ones(number_of_vg_points)

            data1 = np.array([vg_ind, vg, theta_ind, theta_val, bx_ac, by_ac, bz_ac,t_mc, t_p])
            # Scan Vg and acquire data
            
            data2 = scan_Vg(voltage_source, meas_voltage_gain, voltage_channel, vg_range[0], vg_range[1], number_of_vg_points, wait_time/4)

            data = np.vstack((data1, data2))
            DV.add(data.T)

            #plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "Vg", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 1, 50000, -1000, 1000], xname = "Vg", yname = ['Ix', 'R1', 'R2'], logy = [False, True, False], var = 0, unit = "degree")
            # go to next gate voltage
            if ind < number_of_theta_lines: set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], vg_range[0])
        

    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)

    
def scan_R_vs_Vg_theta2_2omega(misc, file_path, amplitude, frequency, gate_gain, meas_voltage_gain, voltage_source, voltage_channel,theta_range, vg_range, number_of_theta_lines, number_of_vg_points, fixed_Bz,fixed_Br,alpha,wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg_ind', 'Vg', 'theta_ind', 'theta', 'Bx_ac', 'By_ac', 'Bz_ac', 'Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V1b', 'V2', 'V2b', 'R1', 'R1b', 'R2', 'R2b', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    theta_lines = np.linspace(theta_range[0], theta_range[1], number_of_theta_lines)
    rot_mat = calc_rotation_matrix(0,0,1,-alpha*0.342,-alpha*0.9396,1)
    t_mc0, t_p0 = read_T()
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg_range[0])

    for ind, theta in enumerate(theta_lines, 1):
        x0, y0 = fixed_Br*np.cos(np.radians(theta)), fixed_Br*np.sin(np.radians(theta)) 
        x, y, z = rotate_vector(rot_mat, x0, y0, fixed_Bz)
        if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
            actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, x, y, z)
            print '\r',"Theta Line:", ind, "out of ", number_of_theta_lines

            vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
            vg = gate_gain* np.linspace(vg_range[0], vg_range[1], number_of_vg_points)  
            theta_ind = np.linspace(ind, ind, number_of_vg_points)
            theta_val = theta * np.ones(number_of_vg_points)
            bx_ac = actual_Bx * np.ones(number_of_vg_points)
            by_ac = actual_By * np.ones(number_of_vg_points)
            bz_ac = actual_Bz * np.ones(number_of_vg_points)
            t_mc = t_mc0 * np.ones(number_of_vg_points)
            t_p = t_p0 * np.ones(number_of_vg_points)

            data1 = np.array([vg_ind, vg, theta_ind, theta_val, bx_ac, by_ac, bz_ac,t_mc, t_p])
            # Scan Vg and acquire data
            
            data2 = scan_Vg_2omega(voltage_source, meas_voltage_gain, voltage_channel, vg_range[0], vg_range[1], number_of_vg_points, wait_time/4)

            data = np.vstack((data1, data2))
            DV.add(data.T)

            #plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "Vg", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 1, 50000, -1000, 1000], xname = "Vg", yname = ['Ix', 'R1', 'R2'], logy = [False, True, False], var = 0, unit = "degree")
            # go to next gate voltage
            if ind < number_of_theta_lines: set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], vg_range[0])
        

    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)


def scan_R_vs_Bz_at_fixed_Vg(misc, file_path, amplitude, frequency, meas_voltage_gain, voltage_source, voltage_channel, bz_range, number_of_bz_points, vg, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg','Tmc', 'Tp', 'Bz', 'acBx', 'acBy', 'acBz')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    Bz_points = np.linspace(bz_range[0], bz_range[1], number_of_bz_points)
    
    t_mc0, t_p0 = read_T()
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg)
     
    d_tmp = None
    p1, p2, p3 = 0.0, 0.0, 0.0
    for z in Bz_points:
        actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, 0, 0, z)
        time.sleep(0.2)
        try:
            p1, p2, p3 = LA1.x(), LA2.x(), LA3.x()
            line_data = [vg, t_mc0, t_p0, z, actual_Bx, actual_By, actual_Bz, p1, p2, p3]
        except: line_data = [vg, t_mc0, t_p0, z, actual_Bx, actual_By, actual_Bz, p1, p2, p3]

        if d_tmp is not None: d_tmp = np.vstack([d_tmp, line_data])
        else: d_tmp = line_data
            
    d = d_tmp.T
    vg1, T_mc, T_p, target_Bz, actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3 = d
    res_5 = np.float64(1.0)*res_2/res_1/meas_voltage_gain
    res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain
    res_7 = np.float64(1.0)/res_5
    res_8 = np.float64(1.0)/res_6

    data = np.array([vg1, T_mc, T_p, target_Bz,  actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3, res_5, res_6, res_7, res_8])
    DV.add(data.T)

    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg, 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    


def scan_R_vs_By_at_fixed_Vg(misc, file_path, amplitude, frequency, meas_voltage_gain, voltage_source, voltage_channel, by_range, number_of_by_points, vg, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg','Tmc', 'Tp', 'By', 'acBx', 'acBy', 'acBz')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    By_points = np.linspace(by_range[0], by_range[1], number_of_by_points)
    
    t_mc0, t_p0 = read_T()
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg)
     
    d_tmp = None
    p1, p2, p3 = 0.0, 0.0, 0.0
    for y in By_points:
        y = min(max(y, -1.0),1.0)
        actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, 0, y, 0)
        time.sleep(0.2)
        try:
            p1, p2, p3 = LA1.x(), LA2.x(), LA3.x()
            line_data = [vg, t_mc0, t_p0, y, actual_Bx, actual_By, actual_Bz, p1, p2, p3]
        except: line_data = [vg, t_mc0, t_p0, y, actual_Bx, actual_By, actual_Bz, p1, p2, p3]

        if d_tmp is not None: d_tmp = np.vstack([d_tmp, line_data])
        else: d_tmp = line_data
            
    d = d_tmp.T
    vg1, T_mc, T_p, target_By, actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3 = d
    res_5 = np.float64(1.0)*res_2/res_1/meas_voltage_gain
    res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain
    res_7 = np.float64(1.0)/res_5
    res_8 = np.float64(1.0)/res_6

    data = np.array([vg1, T_mc, T_p, target_By,  actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3, res_5, res_6, res_7, res_8])
    DV.add(data.T)

    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg, 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
def scan_R_vs_BzBy_at_fixed_Vg_at_rotated_axis(misc, file_path, amplitude, frequency, meas_voltage_gain, voltage_source, voltage_channel,bx_range, by_range, number_of_bx_points, number_of_by_points, fixed_Bz, vg, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg','Tmc', 'Tp', 'Bx', 'By', 'acBx', 'acBy', 'acBz')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    t_mc0, t_p0 = read_T()
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg)
    
    rot_mat = calc_rotation_matrix(0,0,1,0.001,-0.005,0.2)
    print '\r', rot_mat
    
    Bx_points = np.round_(np.linspace(bx_range[0], bx_range[1], number_of_bx_points),6)
    #By_points = np.linspace(by_range[0], by_range[1], number_of_by_points)
    
    for i, x0 in enumerate(Bx_points):
        d_tmp = None
        p1, p2, p3 = 0.0, 0.0, 0.0
        flag = False
        if i%2 == 0: By_points = np.round_(np.linspace(by_range[0], by_range[1], number_of_by_points),6)
        else: By_points = np.round_(np.linspace(by_range[1], by_range[0], number_of_by_points),6)
        for y0 in By_points:
            if x0**2 + y0**2 <= 1.0:
                x, y, z = rotate_vector(rot_mat, x0, y0, fixed_Bz)
                if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
                    flag = True
                    actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, x, y, z)
                    flag2 = True
                    while flag2:
                        try:
                            p1, p2, p3 = 0.0, 0.0, 0.0
                            time.sleep(0.3)
                            for _ in range(20):
                                p1 += LA1.x()
                                p2 += LA2.x()
                                p3 += LA3.x()
                                time.sleep(0.1)
                            p1 /= 20.0
                            p2 /= 20.0
                            p3 /= 20.0
                            flag2 = False
                            line_data = [vg, t_mc0, t_p0, x0, y0, actual_Bx, actual_By, actual_Bz, p1, p2, p3]
                        except:
                            pass

                    if d_tmp is not None: d_tmp = np.vstack([d_tmp, line_data])
                    else: d_tmp = line_data
        if flag and d_tmp is not None:
            d = d_tmp.T
            vg1, T_mc, T_p, target_Bx, target_By, actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3 = d
            res_5 = np.float64(1.0)*res_2/res_1/meas_voltage_gain
            res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain
            res_7 = np.float64(1.0)/res_5
            res_8 = np.float64(1.0)/res_6
            data = np.array([vg1, T_mc, T_p, target_Bx, target_By, actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3, res_5, res_6, res_7, res_8])
            DV.add(data.T)
        
    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg, 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
    
def scan_R_vs_BzBy_at_fixed_Vg_at_rotated_axis_rtheta(misc, file_path, amplitude, frequency, meas_voltage_gain, voltage_source, voltage_channel,br_range, theta_range, number_of_br_points, number_of_theta_points, fixed_Bz,vg, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg','Tmc', 'Tp', 'Br', 'theta', 'acBx', 'acBy', 'acBz')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    t_mc0, t_p0 = read_T()
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg)
    
    rot_mat = calc_rotation_matrix(0,0,1,-0.034*0.342,-0.034*0.9396,1)
    print '\r', rot_mat
    
    Br_points = np.round_(np.linspace(br_range[0], br_range[1], number_of_br_points),6)
    theta_points = np.linspace(theta_range[0], theta_range[1], number_of_theta_points)
    
    for i, br in enumerate(Br_points):
        d_tmp = None
        flag = False
        p1, p2, p3 = 0.0, 0.0, 0.0
        for theta in theta_points:
            x0, y0 = br*np.cos(np.radians(theta)), br*np.sin(np.radians(theta)) 
            x, y, z = rotate_vector(rot_mat, x0, y0, fixed_Bz)
            if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
                flag = True
                actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, x, y, z)
                flag2 = True
                while flag2:
                    try:
                        p1, p2, p3 = 0.0, 0.0, 0.0
                        time.sleep(0.1)
                        for _ in range(20):
                            p1 += LA1.x()
                            p2 += LA2.x()
                            p3 += LA3.x()
                            time.sleep(0.05)
                        p1 /= 20.0
                        p2 /= 20.0
                        p3 /= 20.0
                        flag2 = False
                        line_data = [vg, t_mc0, t_p0, br, theta, actual_Bx, actual_By, actual_Bz, p1, p2, p3]
                    except: pass
                if d_tmp is not None: d_tmp = np.vstack([d_tmp, line_data])
                else: d_tmp = line_data
        if flag and d_tmp is not None:
            d = d_tmp.T
            vg1, T_mc, T_p, target_Br, target_theta, actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3 = d
            res_5 = np.float64(1.0)*res_2/res_1/meas_voltage_gain
            res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain
            res_7 = np.float64(1.0)/res_5
            res_8 = np.float64(1.0)/res_6
            data = np.array([vg1, T_mc, T_p, target_Br, target_theta, actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3, res_5, res_6, res_7, res_8])
            DV.add(data.T)
        
    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg, 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
def scan_R_vs_BrBtheta_at_fixed_Vg(misc, file_path, amplitude, frequency, meas_voltage_gain, voltage_source, voltage_channel,br_range, theta_range, number_of_br_points, number_of_theta_points, fixed_Bz,vg, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg','Tmc', 'Tp', 'Br', 'theta', 'Bx', 'By', 'Bz', 'acBx', 'acBy', 'acBz')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    t_mc0, t_p0 = read_T()
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg)
    
    rot_mat = calc_rotation_matrix(0,0,1,-0.034*0.342,-0.034*0.9396,1)
    print '\r', rot_mat
    
    #Br_points = np.round_(np.linspace(br_range[0], br_range[1], number_of_br_points),6)
    theta_points = np.round_(np.linspace(theta_range[0], theta_range[1], number_of_theta_points))
    
    for i, theta in enumerate(theta_points):
        d_tmp = None
        flag = False
        p1, p2, p3 = 0.0, 0.0, 0.0
        if i%2 == 0: Br_points = np.round_(np.linspace(br_range[0], br_range[1], number_of_br_points),6)
        else: Br_points = np.round_(np.linspace(br_range[1], br_range[0], number_of_br_points),6)
        for br in Br_points:
            x0, y0 = br*np.cos(np.radians(theta)), br*np.sin(np.radians(theta)) 
            x, y, z = rotate_vector(rot_mat, x0, y0, fixed_Bz)
            if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
                flag = True
                actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, x, y, z)
                flag2 = True
                while flag2:
                    try:
                        # p1, p2, p3 = LA1.x(), LA2.x(), LA3.x()
                        p1, p2, p3 = 0.0, 0.0, 0.0
                        time.sleep(0.1)
                        for _ in range(20):
                            p1 += LA1.x()
                            p2 += LA2.x()
                            p3 += LA3.x()
                            time.sleep(0.05)
                        p1 /= 20.0
                        p2 /= 20.0
                        p3 /= 20.0
                        flag2 = False
                        line_data = [vg, t_mc0, t_p0, br, theta, x0, y0, fixed_Bz,actual_Bx, actual_By, actual_Bz, p1, p2, p3]
                    except: pass
                if d_tmp is not None: d_tmp = np.vstack([d_tmp, line_data])
                else: d_tmp = line_data
        if flag and d_tmp is not None:
            d = d_tmp.T
            vg1, T_mc, T_p, target_Br, target_theta, x0, y0, fixed_Bz0,actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3 = d
            res_5 = np.float64(1.0)*res_2/res_1/meas_voltage_gain
            res_6 = np.float64(1.0)*res_3/res_1/meas_voltage_gain
            res_7 = np.float64(1.0)/res_5
            res_8 = np.float64(1.0)/res_6
            data = np.array([vg1, T_mc, T_p, target_Br, target_theta, x0, y0, fixed_Bz0,actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3, res_5, res_6, res_7, res_8])
            DV.add(data.T)
        
    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg, 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
def scan_R_vs_BxBy_at_fixed_Bz_Vg_for_exploring_axis(misc, file_path, amplitude, frequency, meas_voltage_gain, voltage_source, voltage_channel,bx_range, by_range, number_of_bx_points, number_of_by_points, fixed_Bz, vg, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg','Tmc', 'Tp', 'Bx', 'By', 'acBx', 'acBy', 'acBz')
    meas_var = ('Ix', 'V1', 'V2', 'normR1', 'normR2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    t_mc0, t_p0 = read_T()
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg)
    
    Bx_points = np.round_(np.linspace(bx_range[0], bx_range[1], number_of_bx_points),6)
    #By_points = np.linspace(by_range[0], by_range[1], number_of_by_points)
    normalize = 1.0
    z = fixed_Bz
    for i, x in enumerate(Bx_points):
        d_tmp = None
        p1, p2, p3= 0.0, 0.0, 0.0
        flag = False
        if i%2 == 0: By_points = np.round_(np.linspace(by_range[0], by_range[1], number_of_by_points),6)
        else: By_points = np.round_(np.linspace(by_range[1], by_range[0], number_of_by_points),6)
        for y in By_points:
            if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
                flag = True
                actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, x, y, z)
                normalize = math.sqrt(actual_Bx**2 + actual_By**2 + actual_Bz**2)
                flag2 = True
                while flag2:
                    try:
                        p1, p2, p3 = 0.0, 0.0, 0.0
                        time.sleep(0.3)
                        for _ in range(20):
                            p1 += LA1.x()
                            p2 += LA2.x()
                            p3 += LA3.x()
                            time.sleep(0.1)
                        p1 /= 20.0
                        p2 /= 20.0
                        p3 /= 20.0
                        flag2 = False
                        r1_norm = np.float64(1.0)*p2/p1/normalize
                        r2_norm = np.float64(1.0)*p3/p1/normalize
                        line_data = [vg, t_mc0, t_p0, x, y, actual_Bx, actual_By, actual_Bz, p1, p2, p3, r1_norm, r2_norm]
                    except: pass
                if d_tmp is not None: d_tmp = np.vstack([d_tmp, line_data])
                else: d_tmp = line_data
        if flag and d_tmp is not None:
            d = d_tmp.T
            vg1, T_mc, T_p, target_Bx, target_By, actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3, res_5, res_6 = d
            data = np.array([vg1, T_mc, T_p, target_Bx, target_By, actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3, res_5, res_6])
            DV.add(data.T)
        
    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg, 0.0)
    
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)

            

def scan_R_vs_BxBy_at_fixed_Bz_Vg_for_exploring_axis2(misc, file_path, amplitude, frequency, meas_voltage_gain, voltage_source, voltage_channel,bx_range, by_range, number_of_bx_points, number_of_by_points, fixed_Bz, vg, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg','Tmc', 'Tp', 'Bx', 'By', 'acBx', 'acBy', 'acBz')
    meas_var = ('Ix', 'V1', 'V2', 'normR1', 'normR2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    t_mc0, t_p0 = read_T()
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg)
    
    #Bx_points = np.linspace(bx_range[0], bx_range[1], number_of_bx_points)
    By_points = np.round_(np.linspace(by_range[0], by_range[1], number_of_by_points),6)
    normalize = 1.0
    z = fixed_Bz
    for i, y in enumerate(By_points):
        flag = False
        d_tmp = None
        p1, p2, p3 = 0.0, 0.0, 0.0
        if i%2 == 0: Bx_points = np.round_(np.linspace(bx_range[0], bx_range[1], number_of_bx_points),6)
        else: Bx_points = np.round_(np.linspace(bx_range[1], bx_range[0], number_of_bx_points),6)
        for x in Bx_points:
            if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
                flag = True
                actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, x, y, z)
                normalize = math.sqrt(actual_Bx**2 + actual_By**2 + actual_Bz**2)
                flag2 = True
                while flag2:
                    try:
                        p1, p2, p3 = 0.0, 0.0, 0.0
                        time.sleep(0.3)
                        for _ in range(20):
                            p1 += LA1.x()
                            p2 += LA2.x()
                            p3 += LA3.x()
                            time.sleep(0.1)
                        p1 /= 20.0
                        p2 /= 20.0
                        p3 /= 20.0
                        flag2 = False
                        r1_norm = np.float64(1.0)*p2/p1/normalize
                        r2_norm = np.float64(1.0)*p3/p1/normalize
                        line_data = [vg, t_mc0, t_p0, x, y, actual_Bx, actual_By, actual_Bz, p1, p2, p3, r1_norm, r2_norm]
                    except: pass
                if d_tmp is not None: d_tmp = np.vstack([d_tmp, line_data])
                else: d_tmp = line_data
        if flag and d_tmp is not None:
            d = d_tmp.T
            vg1, T_mc, T_p, target_Bx, target_By, actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3, res_5, res_6 = d

            data = np.array([vg1, T_mc, T_p, target_Bx, target_By, actual_Bx, actual_By, actual_Bz, res_1, res_2, res_3, res_5, res_6])
            DV.add(data.T)
        
    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg, 0.0)
    
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)

    
def scan_R_vs_Vg_T(misc, file_path, amplitude, frequency, gate_gain, meas_voltage_gain, voltage_source, voltage_channel,t_range, vg_range, number_of_t_lines, number_of_vg_points, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vg_ind', 'Vg',  't_ind', 'Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg_range[0])

    for ind, val in enumerate(t_lines, 1):
        t_mc0, t_p0 = set_T(val)
        print '\r',"Temperature Line:", ind, "out of ", number_of_t_lines
        
        vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
        vg = gate_gain* np.linspace(vg_range[0], vg_range[1], number_of_vg_points)  
        t_ind = np.linspace(ind, ind, number_of_vg_points)
        t_mc = t_mc0 * np.ones(number_of_vg_points)
        t_p = t_p0 * np.ones(number_of_vg_points)
        
        data1 = np.array([vg_ind, vg, t_ind, t_mc, t_p])
        # Scan Vg and acquire data
        data2 = scan_Vg(voltage_source, meas_voltage_gain, voltage_channel, vg_range[0], vg_range[1], number_of_vg_points, wait_time/4)
        
        data = np.vstack((data1, data2))
        DV.add(data.T)
        # go to next gate voltage
        set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], vg_range[0])

    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
    
def scan_dIdV_vs_I_idc_fixedD_fixedB(
    file_path,
    voltage_source,
    voltage_channel_bottom,
    voltage_channel_top,
    voltage_channel_dc,
    amplitude = 0.01,
    frequency = 17.777,
    gate_gain = 1.0,
    meas_voltage_gain = 1.0,
    displacement_field = 0.0,
    magnetic_field = 0.0,
    n_range = [0, 0],
    idc_range = [-1.0, 1.0],
    number_of_n_lines = 1,
    number_of_idc_points = 200,
    wait_time = wait_time,
    c_delta = 0.0,
    misc = "misc"
):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('idc_ind', 'idc', 'n_ind', 'n', 'Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    #t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    n_lines = np.linspace(n_range[0], n_range[1], number_of_n_lines)

    t_mc0, t_p0 = 0, 0
    D_val = displacement_field
    ##### Measurements start #####
    # go to initial gate volatge
    set_Bz(MGz, magnetic_field)
    vtg_last, vbg_last = 0, 0
    vdc_last = 0
    for ind, val in enumerate(n_lines, 1):
        print '\r',"n Line:", ind, "out of ", len(n_lines)
        vtg_s, vbg_s = find_vt_vb(D_val, val, c_delta)
        sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                       out_ch_top = voltage_channel_top, 
                       vbg_start = vbg_last, vbg_end = vbg_s, 
                       vtg_start = vtg_last, vtg_end = vtg_s, 
                       points_vbg = 40, points_vtg = 40, delay = 0.005)
        sim_sweep(out_ch = voltage_channel_dc, 
                  vstart = vdc_last,
                  vend = idc_range[0],
                  points = 50,
                  delay = 0.005)
        idc_ind = np.linspace(1, number_of_idc_points, number_of_idc_points)
        idc = 1e-8 * np.linspace(idc_range[0], idc_range[1], number_of_idc_points)  
        n_ind = np.linspace(ind, ind, number_of_idc_points)
        n_val = val * np.ones(number_of_idc_points)
        t_mc = t_mc0 * np.ones(number_of_idc_points)
        t_p = t_p0 * np.ones(number_of_idc_points)
        
        data1 = np.array([idc_ind, idc, n_ind, n_val, t_mc, t_p])
        # Scan Vg and acquire data
        d3 = datetime.datetime.now()
        data2 = scan_Vg_one(voltage_source = voltage_source, 
                        meas_voltage_gain = meas_voltage_gain,
                        amplitude = amplitude,
                        voltage_channel = voltage_channel_dc, 
                        start_v = idc_range[0], 
                        end_v = idc_range[1], 
                        number_of_vg_points = number_of_idc_points, 
                        wait_time = 0)
        
        t = datetime.datetime.now()-d3
        print '\r', "one epoch time:", t, "estimated finish time:", datetime.datetime.now()+(len(n_lines)-ind)*t
        
        vdc_last = idc_range[1]
        vtg_last, vbg_last = vtg_s, vbg_s
        data = np.vstack((data1, data2))
        DV.add(data.T)
        
        
    
        
        plot_fig(file_name = scan_name, file_num = file_number, data = data, cl = list(scan_var) + list(meas_var), xsize = 12, ysize = 16, xaxis = "idc", yaxis = ['Ix', 'R1', 'R2'], xscale = [None, None], yscale = [None, None, 0,500, 0, 1000], xname = "n", yname = ['Ix', 'R1', 'R2'], logy = [False, False, False], var = 0, unit = "T")
        
        #go to next gate voltage
        
    # go to 0 V
    
    sim_sweep(out_ch = voltage_channel_dc, 
          vstart = vdc_last,
          vend = 0.0,
          points = 50,
          delay = 0.005)
    sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                   out_ch_top = voltage_channel_top, 
                   vbg_start = vbg_last, vbg_end = 0.0, 
                   vtg_start = vtg_last, vtg_end = 0.0, 
                   points_vbg = 40, points_vtg = 40, delay = 0.005)
    set_Bz(MGz, 0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    

def scan_dIdV_vs_I_Bz(misc, file_path, amplitude, frequency, gate_gain, meas_voltage_gain,bias_gain, voltage_source_DC, voltage_source_gate, voltage_channel_DC, voltage_channel_gate,vg, vdc_range,bz_range, number_of_bz_lines, number_of_vdc_points, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vdc_ind', 'Idc', 'Bz_ind', 'Bz', 'Bz_ac', 'Tmc', 'Tp','Vg')
    meas_var = ('Ix', 'dV1', 'dV2', 'dVdI1', 'dVdI2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    b_lines = np.linspace(bz_range[0], bz_range[1], number_of_bz_lines)
    
    t_mc0, t_p0 = read_T()
    
    ##### Measurements start #####
    # go to initial gate volatge and DC voltage
    set_Vg_nodac(voltage_source_gate, voltage_channel_gate, 0.0, vg)
    set_Vg_nodac(voltage_source_DC, voltage_channel_DC, 0.0, vdc_range[0])

    for ind, val in enumerate(b_lines, 1):
        actual_B = set_Bz(MGz, val)
        print '\r',"Field Line:", ind, "out of ", number_of_bz_lines
        # create index data etc.
        vdc_ind = np.linspace(1, number_of_vdc_points, number_of_vdc_points)
        Idc = bias_gain/1e-8 * np.linspace(vdc_range[0], vdc_range[1], number_of_vdc_points)  
        b_ind = np.linspace(ind, ind, number_of_vdc_points)
        b_val = val * np.ones(number_of_vdc_points)
        b_ac = actual_B * np.ones(number_of_vdc_points)
        vg0 = vg * np.ones(number_of_vdc_points)
        t_mc = t_mc0 * np.ones(number_of_vdc_points)
        t_p = t_p0 * np.ones(number_of_vdc_points)
        
        data1 = np.array([vdc_ind, Idc, b_ind, b_val, b_ac, t_mc, t_p, vg0])
        # Scan Vg and acquire data
        data2 = scan_Vg(voltage_source_DC, meas_voltage_gain, voltage_channel_DC, vdc_range[0], vdc_range[1], number_of_vdc_points, wait_time/4)
        
        data = np.vstack((data1, data2)) 
        DV.add(data.T)

        # go to next gate voltage
        if ind < number_of_bz_lines: set_Vg_nodac(voltage_source_DC, voltage_channel_DC, vdc_range[1], vdc_range[0])

    # go to 0 V both Vdc and Vg
    set_Vg_nodac(voltage_source_DC, voltage_channel_DC, vdc_range[1], 0.0)
    set_Vg_nodac(voltage_source_gate, voltage_channel_gate, vg, 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
    
    
    
def scan_dIdV_vs_I_Btheta_at_fixed_Br(misc, file_path, amplitude, frequency, gate_gain, meas_voltage_gain,bias_gain, voltage_source_DC, voltage_source_gate, voltage_channel_DC, voltage_channel_gate,vg, br, vdc_range,btheta_range, number_of_btheta_lines, number_of_vdc_points, wait_time):
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('Vdc_ind', 'Idc','B_ind', 'B_theta', 'B_r', 'Bx_ac','By_ac','Bz_ac', 'Tmc', 'Tp','Vg')
    meas_var = ('Ix', 'dV1', 'dV2', 'dVdI1', 'dVdI2', 'G1', 'G2')
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(DV, file_path, file_number, date1, scan_name, meas_parameters, amplitude, frequency)

    #Create meshes
    b_lines = np.linspace(bz_range[0], bz_range[1], number_of_bz_lines) 
    t_mc0, t_p0 = read_T()
    
    ##### Measurements start #####
    # go to initial gate volatge and DC voltage
    set_Vg_nodac(voltage_source_gate, voltage_channel_gate, 0.0, vg)
    set_Vg_nodac(voltage_source_DC, voltage_channel_DC, 0.0, vdc_range[0])

    for ind, theta in enumerate(theta_lines, 1):
        x0, y0 = Br*np.cos(theta), Br*np.sin(theta)
        x, y, z = rotate_vector(x0, y0)
        x, y = min(max(x, -1.0), 1.0), min(max(y, -1.0), 1.0)
        actual_Bx, actual_By, actual_Bz = set_BxByBz(MGx, MGy, MGz, x, y, z)
        print '\r',"Field Line:", ind, "out of ", number_of_bz_lines
        # create index data etc.
        vdc_ind = np.linspace(1, number_of_vdc_points, number_of_vdc_points)
        Idc = bias_gain/1e-8 * np.linspace(vdc_range[0], vdc_range[1], number_of_vdc_points)  
        b_ind = np.linspace(ind, ind, number_of_vdc_points)
        b_theta = theta * np.ones(number_of_vdc_points)
        b_r = Br*np.ones(number_of_vdc_points)
        bx_ac = actual_Bx * np.ones(number_of_vdc_points)
        by_ac = actual_By * np.ones(number_of_vdc_points)
        bz_ac = actual_Bz * np.ones(number_of_vdc_points)
        vg0 = vg * np.ones(number_of_vdc_points)
        t_mc = t_mc0 * np.ones(number_of_vdc_points)
        t_p = t_p0 * np.ones(number_of_vdc_points)
        
        data1 = np.array([vdc_ind, Idc, b_ind, b_theta, b_r, bx_ac, by_ac, bz_ac, t_mc, t_p, vg0])
        # Scan Vg and acquire data
        data2 = scan_Vg(voltage_source_DC, meas_voltage_gain, voltage_channel_DC, vdc_range[0], vdc_range[1], number_of_vdc_points, wait_time/4)
        
        data = np.vstack((data1, data2)) 
        DV.add(data.T)

        # go to next gate voltage
        if ind < number_of_theta_lines: set_Vg_nodac(voltage_source_DC, voltage_channel_DC, vdc_range[1], vdc_range[0])

    # go to 0 V both Vdc and Vg
    set_Vg_nodac(voltage_source_DC, voltage_channel_DC, vdc_range[1], 0.0)
    set_Vg_nodac(voltage_source_gate, voltage_channel_gate, vg, 0.0)
    print '\r',"measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)

    
    
def scan_R_vs_B_fixed_n_D(
    file_path,
    voltage_source,
    voltage_channel_bottom,
    voltage_channel_top,
    amplitude = 0.01,
    frequency = 17.777,
    gate_gain = 1.0,
    meas_voltage_gain = 1.0,
    n_range = [0, 0],
    D_range = [0, 0],
    b_range = [-0.1, 0.1],
    number_of_n_points = 1,
    number_of_D_points = 1,
    number_of_b_points = 100,
    wait_time = wait_time,
    c_delta = 0.0,
    misc = "misc"
):
    
    MGz.conf_ramp_rate_field(0.018060)
    #Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    #Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)
    
    #Create data file and save measurement parameters
    scan_var = ('B_ind', 'B', 'n_ind', 'D_ind', 'n', 'D', 'Tmc', 'Tp')
    meas_var = ('Ix', 'V1', 'V2', 'R1', 'R2', 'G1', 'G2')
    
    n_lines = np.linspace(n_range[0], n_range[1], number_of_n_points)
    D_lines = np.linspace(D_range[0], D_range[1], number_of_D_points)

    t_mc0, t_p0 = 0, 0
    ##### Measurements start #####
    # go to initial gate volatge
    vtg_last, vbg_last = 0, 0
    n_mesh, D_mesh = np.meshgrid(n_lines, D_lines, sparse=False, indexing='ij')
    for i in range(number_of_n_points):
        for j in range(number_of_D_points):
            

            n, D = n_mesh[i,j], D_mesh[i,j]
            
            print("n: ", n, "D: ", D)
        
            vtg_s, vbg_s = find_vt_vb(D, n, c_delta)

            sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                           out_ch_top = voltage_channel_top, 
                           vbg_start = vbg_last, vbg_end = vbg_s, 
                           vtg_start = vtg_last, vtg_end = vtg_s, 
                           points_vbg = 40, points_vtg = 40, delay = 0.005)

            b_ind = np.linspace(1, number_of_b_points, number_of_b_points)
            b = np.linspace(b_range[0], b_range[1], number_of_b_points)
            br = b[::-1]
            n_ind = i * np.ones(number_of_b_points)
            D_ind = j * np.ones(number_of_b_points)
            n_val = n * np.ones(number_of_b_points)
            D_val = D * np.ones(number_of_b_points)
            t_mc = t_mc0 * np.ones(number_of_b_points)
            t_p = t_p0 * np.ones(number_of_b_points)

            file_number1 = create_file(DV, file_path, scan_name + "_trace", scan_var, meas_var)
            write_meas_parameters(DV, file_path, file_number1, date1, scan_name + "_trace", meas_parameters, amplitude, frequency)

            data1 = np.array([b_ind, b, n_ind, D_ind,  n_val, D_val, t_mc, t_p])
            data2 = scan_B_dual_one(meas_voltage_gain = meas_voltage_gain, 
                                    amplitude = amplitude, 
                                    b_start = b_range[0], 
                                    b_end = b_range[1], 
                                    points_b = number_of_b_points)
            data = np.vstack((data1, data2))
            DV.add(data.T)


            file_number2 = create_file(DV, file_path, scan_name + "_retrace", scan_var, meas_var)
            write_meas_parameters(DV, file_path, file_number2, date1, scan_name + "_retrace", meas_parameters, amplitude, frequency)

            data1 = np.array([b_ind, br, n_ind, D_ind,  n_val, D_val, t_mc, t_p])
            data2 = scan_B_dual_one(meas_voltage_gain = meas_voltage_gain, 
                                    amplitude = amplitude, 
                                    b_start = b_range[1], 
                                    b_end = b_range[0], 
                                    points_b = number_of_b_points)
            data = np.vstack((data1, data2))
            DV.add(data.T)

            vtg_last, vbg_last = vtg_s, vbg_s
            
        
    sim_dual_sweep(out_ch_bottom = voltage_channel_bottom, 
                   out_ch_top = voltage_channel_top, 
                   vbg_start = vbg_last, vbg_end = 0.0, 
                   vtg_start = vtg_last, vtg_end = 0.0, 
                   points_vbg = 40, points_vtg = 40, delay = 0.005)

    print '\r',"measurement number: ", file_number2, scan_name, " done"
    ##### Measurements done #####
    MGz.conf_ramp_rate_field(0.05060)
    
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)