import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
import csv
import math
    
directory = "Air" # The directory we want to look at from where this code is
open("bestData"+str(directory)+".csv", mode = "w") # Opens in write only to clear file of all previous contents
curveFitTimes = pd.read_csv("startEndTimes.csv",names=("gas", "time", "10ml", "20ml", "30ml", "40ml", "50ml", "60ml", "70ml", "80ml", "90ml", "100ml"),skiprows=1)
print(curveFitTimes) # Displays table of general start and end times for curve fitting

def calcGamma(filename): # Given a file, will calc value of gamma from curve fitting

    if "Air" in directory:
        startTimeRow = 0
    elif "CO2" in directory:
        startTimeRow = 2
    elif "N2" in directory:
        startTimeRow = 4
    elif "He" in directory:
        startTimeRow = 6
    
    if "10ml" in filename: # Volume depends on which file we're looking at
        volume = (10+6.3)*10**(-6)
        lower = curveFitTimes["10ml"][startTimeRow]
        upper = curveFitTimes["10ml"][startTimeRow+1]
    elif "20ml" in filename:
        volume = (20+6.3)*10**(-6)
        lower = curveFitTimes["20ml"][startTimeRow]
        upper = curveFitTimes["20ml"][startTimeRow+1]
    elif "30ml" in filename:
        volume = (30+6.3)*10**(-6)
        lower = curveFitTimes["30ml"][startTimeRow]
        upper = curveFitTimes["30ml"][startTimeRow+1]
    elif "40ml" in filename:
        volume = (40+6.3)*10**(-6)
        lower = curveFitTimes["40ml"][startTimeRow]
        upper = curveFitTimes["40ml"][startTimeRow+1]
    elif "50ml" in filename:
        volume = (50+6.3)*10**(-6)
        lower = curveFitTimes["50ml"][startTimeRow]
        upper = curveFitTimes["50ml"][startTimeRow+1]
    elif "60ml" in filename:
        volume = (60+6.3)*10**(-6)
        lower = curveFitTimes["60ml"][startTimeRow]
        upper = curveFitTimes["60ml"][startTimeRow+1]
    elif "70ml" in filename:
        volume = (70+6.3)*10**(-6)
        lower = curveFitTimes["70ml"][startTimeRow]
        upper = curveFitTimes["70ml"][startTimeRow+1]
    elif "80ml" in filename:
        volume = (80+6.3)*10**(-6)
        lower = curveFitTimes["80ml"][startTimeRow]
        upper = curveFitTimes["80ml"][startTimeRow+1]
    elif "90ml" in filename:
        volume = (90+6.3)*10**(-6)
        lower = curveFitTimes["90ml"][startTimeRow]
        upper = curveFitTimes["90ml"][startTimeRow+1]
    elif "100ml" in filename:
        volume = (100+6.3)*10**(-6)
        lower = curveFitTimes["100ml"][startTimeRow]
        upper = curveFitTimes["100ml"][startTimeRow+1]
    
    if ("He" in directory): # If from Air data, these particular repeats of the data:
        lower = 0.0
        if "20ml" in filename:
            lower  = 0.001
    if ("Air" in directory):
        if ("10ml" in filename) or ("20ml" in filename) or ("30ml" in filename):
            lower = 0.0
    
    
    data = pd.read_csv("Gas Data/"+str(directory)+"/"+str(filename),names=("time","emf"),skiprows=2) # Read file we are looking at, ensuring to go through required directory
    clean_time = [] # Create list for time values
    clean_emf = [] # Create list for emf values
    
    # for i in range(0, len(data.emf)):
    #     if data.emf[i] == max(data.emf):
    #         lower = data.time[i]
    upper = 0.6*upper # Sloan goes till 2x the decay constant
    
    for i in range(0,len(data.time)-1):
        if data.time[i]>=lower and data.time[i]<=upper:
            clean_time.append(data.time[i])
            clean_emf.append(data.emf[i])
            
    
    
    time = np.arange(lower,upper,0.00001)
    
    def fit(time, A, beta, omegaStar, phi, c):    
        exp_term = A*np.exp(-beta*time)
        cos = np.sin(omegaStar*time + phi)
        return exp_term*(cos) + c
    
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    plt.tick_params(direction='in',
                    length=7,
                    bottom='on',
                    left='on',
                    top='on',
                    right='on')
    plt.xlabel('Time / s')
    plt.ylabel('EMF / V')
    plt.xlim(lower, 0.6)
    plt.plot(clean_time,clean_emf)
    try: # Will attempt to curve fit
        popt, pcov = curve_fit(fit, clean_time, clean_emf, maxfev=10000)
        plt.plot(time,fit(time,popt[0],popt[1],popt[2],popt[3], popt[4]),linestyle = '-', color = 'black')
        omega = np.sqrt(popt[2]**2+(popt[1])**2)
        omegaError = np.sqrt(((popt[2]**2+popt[1]**2)**-0.5*popt[2]*(pcov[2,2])**0.5)**2
                           +((popt[2]**2+popt[1]**2)**-0.5*popt[1]*(pcov[1,1])**0.5)**2)
    except RuntimeError or ValueError or RuntimeWarning: # If there is an error, abort curve fitting and continue on to next set of data
        print("Error: Curve fit could not be found for file " + str(filename))
        omega = 0
        omegaError = 0
        
    if math.isnan(omega) == True or math.isnan(omegaError) == True: # Replace any NaNs with 0
        omega = 0
        omegaError = 0
    
    plt.tick_params(direction='in',top='on',bottom='on',left='on',right='on')
    plt.rcParams.update({'font.size':15})
    
    mass = 106.68*10**(-3)
    if directory == "Air":
        pressure = 101400
    elif directory == "N2":
        pressure = 101100
    elif directory == "He":
        pressure = 101100
    elif directory == "CO2":
        pressure = 101700
    area = (np.pi*0.03416**2)/4
    
    gamma = (mass*volume*omega**2)/(pressure*area**2)
    print("For File " + filename + ", gamma = {:.3f}".format(gamma)+", omega = {:.3f}".format(omega)+" +- {:.3f}".format(omegaError)+", maxAmp = "+str(max(data.emf)))
    plt.title(str(filename)+" Data, Gamma = {:.3f}".format(gamma))
    
    with open("bestData"+str(directory)+".csv", mode = "a") as bestData: # Append data to .csv file
        writeToFile = csv.writer(bestData, delimiter=",")
        writeToFile.writerow([str(directory), str(volume*10**6), str(filename), str(omega), str(omegaError)])
    
    plt.show()
    
    


for filename in os.listdir("Gas Data/"+directory): # Looks at all files in this directory
    if filename.endswith(".csv"): # If any are .csv files
        calcGamma(filename) # Calculates gamma from this file
    else:
        continue