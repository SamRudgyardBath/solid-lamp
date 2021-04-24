import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit ## for fitting a line to our data
import pandas as pd
plt.rcParams.update({'font.size': 22})


directory = "Air" # The directory we want to look at from where this code is
data = pd.read_csv("bestData"+str(directory)+".csv", names=("gas", "volume", "filename", "omega", "omegaError"))

list10ml = list() # Lists to contain all values of omega for each V
list10mlError = list()
list20ml = list()
list20mlError = list()
list30ml = list()
list30mlError = list()
list40ml = list()
list40mlError = list()
list50ml = list()
list50mlError = list()
list60ml = list()
list60mlError = list()
list70ml = list()
list70mlError = list()
list80ml = list()
list80mlError = list()
list90ml = list()
list90mlError = list()
list100ml = list()
list100mlError = list()


for i in range(0, len(data)): # Loop through all data and assign to correct list depending on volume
    if 9+6.3 <= data.volume[i] <= 11+6.3:
        list10ml.append(data.omega[i])
        list10mlError.append(data.omegaError[i])
    elif 19+6.3 <= data.volume[i] <= 21+6.3:
        list20ml.append(data.omega[i])
        list20mlError.append(data.omegaError[i])
    elif 29+6.3 <= data.volume[i] <= 31+6.3:
        list30ml.append(data.omega[i])
        list30mlError.append(data.omegaError[i])
    elif 39+6.3 <= data.volume[i] <= 41+6.3:
        list40ml.append(data.omega[i])
        list40mlError.append(data.omegaError[i])
    elif 49+6.3 <= data.volume[i] <= 51+6.3:
        list50ml.append(data.omega[i])
        list50mlError.append(data.omegaError[i])
    elif 59+6.3 <= data.volume[i] <= 61+6.3:
        list60ml.append(data.omega[i])
        list60mlError.append(data.omegaError[i])
    elif 69+6.3 <= data.volume[i] <= 71+6.3:
        list70ml.append(data.omega[i])
        list70mlError.append(data.omegaError[i])
    elif 79+6.3 <= data.volume[i] <= 81+6.3:
        list80ml.append(data.omega[i])
        list80mlError.append(data.omegaError[i])
    elif 89+6.3 <= data.volume[i] <= 91+6.3:
        list90ml.append(data.omega[i])
        list90mlError.append(data.omegaError[i])
    elif 99+6.3 <= data.volume[i] <= 101+6.3:
        list100ml.append(data.omega[i])
        list100mlError.append(data.omegaError[i])

topThree10ml = sorted(zip(list10ml, list10mlError),reverse=True)[0:3]
topThree20ml = sorted(zip(list20ml, list20mlError),reverse=True)[0:3]
topThree30ml = sorted(zip(list30ml, list30mlError),reverse=True)[0:3]
topThree40ml = sorted(zip(list40ml, list40mlError),reverse=True)[0:3]
topThree50ml = sorted(zip(list50ml, list50mlError),reverse=True)[0:3]
topThree60ml = sorted(zip(list60ml, list60mlError),reverse=True)[0:3]
topThree70ml = sorted(zip(list70ml, list70mlError),reverse=True)[0:3]
topThree80ml = sorted(zip(list80ml, list80mlError),reverse=True)[0:3]
topThree90ml = sorted(zip(list90ml, list90mlError),reverse=True)[0:3]
topThree100ml = sorted(zip(list100ml, list100mlError),reverse=True)[0:3]

numerator = 0
denominator = 0
for i in range(0,3):
    if topThree10ml[i][1] != 0:
        weight = 1/(topThree10ml[i][1]**2) # [i][0] value is the uncertainty, weight = 1/sigma^2
    else:
        weight = 9E+99
    numerator += weight*topThree10ml[i][0] # Increase numerator by w_i*x_i
    denominator += weight
    mean10ml = numerator/denominator
    uncertainty10ml = 1/np.sqrt(denominator) # ∆y = 2<x>∆x
numerator = 0
denominator = 0
for i in range(0,3):
    if topThree20ml[i][1] != 0:
        weight = 1/(topThree20ml[i][1]**2) # i+1th value is the uncertainty, weight = 1/sigma^2
    else:
        weight = 9E+99
    numerator += weight*topThree20ml[i][0] # Increase numerator by w_i*x_i
    denominator += weight
    mean20ml = numerator/denominator
    uncertainty20ml = 1/np.sqrt(denominator) # ∆y = 2<x>∆x
numerator = 0
denominator = 0
for i in range(0,3):
    if topThree30ml[i][1] != 0:
        weight = 1/(topThree30ml[i][1]**2) # i+1th value is the uncertainty, weight = 1/sigma^2
    else:
        weight = 9E+99
    numerator += weight*topThree30ml[i][0] # Increase numerator by w_i*x_i
    denominator += weight
    mean30ml = numerator/denominator
    uncertainty30ml = 1/np.sqrt(denominator) # ∆y = 2<x>∆x
numerator = 0
denominator = 0
for i in range(0,3):
    if topThree40ml[i][1] != 0:
        weight = 1/(topThree40ml[i][1]**2) # i+1th value is the uncertainty, weight = 1/sigma^2
    else:
        weight = 9E+99
    numerator += weight*topThree40ml[i][0] # Increase numerator by w_i*x_i
    denominator += weight
    mean40ml = numerator/denominator
    uncertainty40ml = 1/np.sqrt(denominator) # ∆y = 2<x>∆x
numerator = 0
denominator = 0
for i in range(0,3):
    if topThree50ml[i][1] != 0:
        weight = 1/(topThree50ml[i][1]**2) # i+1th value is the uncertainty, weight = 1/sigma^2
    else:
        weight = 9E+99
    numerator += weight*topThree50ml[i][0] # Increase numerator by w_i*x_i
    denominator += weight
    mean50ml = numerator/denominator
    uncertainty50ml = 1/np.sqrt(denominator) # ∆y = 2<x>∆x
numerator = 0
denominator = 0
for i in range(0,3):
    if topThree60ml[i][1] != 0:
        weight = 1/(topThree60ml[i][1]**2) # i+1th value is the uncertainty, weight = 1/sigma^2
    else:
        weight = 9E+99
    numerator += weight*topThree60ml[i][0] # Increase numerator by w_i*x_i
    denominator += weight
    mean60ml = numerator/denominator
    uncertainty60ml = 1/np.sqrt(denominator) # ∆y = 2<x>∆x
numerator = 0
denominator = 0
for i in range(0,3):
    if topThree70ml[i][1] != 0:
        weight = 1/(topThree70ml[i][1]**2) # i+1th value is the uncertainty, weight = 1/sigma^2
    else:
        weight = 9E+99
    numerator += weight*topThree70ml[i][0] # Increase numerator by w_i*x_i
    denominator += weight
    mean70ml = numerator/denominator
    uncertainty70ml = 1/np.sqrt(denominator) # ∆y = 2<x>∆x
numerator = 0
denominator = 0
for i in range(0,3):
    if topThree80ml[i][1] != 0:
        weight = 1/(topThree80ml[i][1]**2) # i+1th value is the uncertainty, weight = 1/sigma^2
    else:
        weight = 9E+99
    numerator += weight*topThree80ml[i][0] # Increase numerator by w_i*x_i
    denominator += weight
    mean80ml = numerator/denominator
    uncertainty80ml = 1/np.sqrt(denominator) # ∆y = 2<x>∆x
numerator = 0
denominator = 0
for i in range(0,3):
    if topThree90ml[i][1] != 0:
        weight = 1/(topThree90ml[i][1]**2) # i+1th value is the uncertainty, weight = 1/sigma^2
    else:
        weight = 9E+99
    numerator += weight*topThree90ml[i][0] # Increase numerator by w_i*x_i
    denominator += weight
    mean90ml = numerator/denominator
    uncertainty90ml = 1/np.sqrt(denominator) # ∆y = 2<x>∆x
numerator = 0
denominator = 0
for i in range(0,3):
    if topThree100ml[i][1] != 0:
        weight = 1/(topThree100ml[i][1]**2) # i+1th value is the uncertainty, weight = 1/sigma^2
    else:
        weight = 9E+99
    numerator += weight*topThree100ml[i][0] # Increase numerator by w_i*x_i
    denominator += weight
    mean100ml = numerator/denominator
    uncertainty100ml = 1/np.sqrt(denominator) # ∆y = 2<x>∆x
    

plotOmega = [mean10ml, mean20ml, mean30ml, mean40ml, mean50ml, mean60ml, mean70ml, mean80ml, mean90ml, mean100ml]
plotOmegaError = [uncertainty10ml, uncertainty20ml, uncertainty30ml, uncertainty40ml, uncertainty50ml, uncertainty60ml, uncertainty70ml, uncertainty80ml, uncertainty90ml, uncertainty100ml]
plotVolume = [1/np.sqrt((10+6.3)*10**(-6)), 1/np.sqrt((20+6.3)*10**(-6)), 1/np.sqrt((30+6.3)*10**(-6)), 1/np.sqrt((40+6.3)*10**(-6)), 1/np.sqrt((50+6.3)*10**(-6)), 1/np.sqrt((60+6.3)*10**(-6)), 1/np.sqrt((70+6.3)*10**(-6)), 1/np.sqrt((80+6.3)*10**(-6)), 1/np.sqrt((90+6.3)*10**(-6)), 1/np.sqrt((100+6.3)*10**(-6))]

volume = [16.3*10**(-6), 26.3*10**(-6), 36.3*10**(-6), 46.3*10**(-6), 56.3*10**(-6), 66.3*10**(-6), 76.3*10**(-6), 86.3*10**(-6), 96.3*10**(-6), 106.3*10**(-6)]

    
if directory == "N2":
    plotOmega = [mean10ml, mean20ml, mean30ml, mean40ml, mean50ml, mean60ml, mean70ml, mean80ml, mean90ml]
    plotOmegaError = [uncertainty10ml, uncertainty20ml, uncertainty30ml, uncertainty40ml, uncertainty50ml, uncertainty60ml, uncertainty70ml, uncertainty80ml, uncertainty90ml]
    plotVolume = [1/np.sqrt((10+6.3)*10**(-6)), 1/np.sqrt((20+6.3)*10**(-6)), 1/np.sqrt((30+6.3)*10**(-6)), 1/np.sqrt((40+6.3)*10**(-6)), 1/np.sqrt((50+6.3)*10**(-6)), 1/np.sqrt((60+6.3)*10**(-6)), 1/np.sqrt((70+6.3)*10**(-6)), 1/np.sqrt((80+6.3)*10**(-6)), 1/np.sqrt((90+6.3)*10**(-6))]

plotVolumeError = []
for i in range(0, len(plotVolume)):
    plotVolumeError.append(abs(plotVolume[i] - 1/np.sqrt(volume[i]+10**(-6))))

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.tick_params(direction='in',
                length=7,
                bottom='on',
                left='on',
                top='on',
                right='on')
plt.xlabel('$1/{\sqrt{V}}$ / $m^{2/3}$')
plt.ylabel('$\omega$ / rads$^{-1}$')
plt.errorbar(plotVolume, plotOmega, markersize = 12, yerr = plotOmegaError, xerr = plotVolumeError, capsize = 5, fmt='ko')
plt.xlim(90, 200)
plt.ylim(75,225)
#plt.title(directory)

def line(x, slope, intercept):          # Set up the linear fitting - don't ammend
    return slope*x + intercept          # More set up, leave alone.

print("Gas: "+directory)

popt, pcov = curve_fit(line, plotVolume, plotOmega, sigma = plotOmegaError)
slope = popt[0]
intercept = popt[1]
err_slope = np.sqrt(float(pcov[0][0]))
err_intercept = np.sqrt(float(pcov[1][1]))
print('Slope: {0:.3f} +- {1:.3f}'.format(slope, err_slope))
print('Intercept: {0:.3f} +- {1:.3f}'.format(intercept, err_intercept))

x = np.arange(0, 210, 1)
ax.plot(x, x*slope+intercept, 
         linestyle='--',
         color='black',
         label='Best fit')

plt.savefig("plotOmegaVolume"+directory+".png", dpi=300)

plt.show()

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

# Calculate uncertainties
pressureU = 100/pressure
slopeU = err_slope/slope
massU = (0.01*10**(-3))/mass
areaU = (2*np.pi*0.03416/2*(0.01*10**(-3)))/area

gammaU = np.sqrt(slopeU**2 + massU**2 + pressureU**2 + areaU**2)

gamma = slope*mass/(pressure*area**2)
print("Method 1: Gamma = {:.4f} +- {:.4f}".format(gamma, gammaU))

pressureU = 100
massU = 0.01*10**(-3)
areaU = 2*np.pi*0.03416/2*(0.01*10**(-3))

deltaSlope = (2*slope*mass*err_slope)/(pressure*area**2)
deltaMass = (slope**2*massU)/(pressure*area**2)
deltaPressure = (slope**2*mass*pressureU)/(pressure**2*area**2)
deltaArea = (2*slope**2*mass*areaU)/(pressure*area**3)

gammaU = np.sqrt(deltaSlope**2 + deltaMass**2 + deltaPressure**2 + deltaArea**2)

print("Method 2: Gamma = {:.4f} +- {:.4f}".format(gamma, gammaU))