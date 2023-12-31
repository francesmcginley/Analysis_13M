"""
Code to analyse 13D simulation outputs.

CombineYearlyFiles - combines data from a single sim. Crops to Southern Europe and summer months.
            This uses the h1 yearly files, not the monthly ones.

SingleRun - reads .nc files output by CombineYearlyFiles
Ensemble -  combines data runs within an ensemble, by reading CombineYearlyFiles .nc output
HistogramAnalysis - bins data, fits gaussian/poisson/gev to binned data, computes threshold..

Our method:
0. Read in sim output files, crop to summer months and Southern Europe lat lon, and save as .nc file in ./Data directory (this all happens in CombineYearlyFiles) 
0.5.  Either - Open saved .nc file. Use Ensemble class to create an ensemble object that can be input to class HistogramAnalysis. 
      OR (if not looking at an ensemble) - Open saved .nc file. Use SingleRun class to create an SingleRun object that can be input to class HistogramAnalysis. 

(all other steps are in class HistogramAnalysis)
1. For each spatial gridpoint, calculate the Heat index
2. Take a weighted average over lat and lon. We do this by multiplying by land fractions and dividing by the sum of land fractions
3. Bin data into 1-day bins (may change?) to get a histogram
4. Fit a gaussian to the histogram


TODO:
- Get a sensible threshold...



@author : s1935349

"""

# imports
import matplotlib.pyplot as plt # matplotlib library
import xarray as xr# xarray
import numpy as np # numpy
from scipy import stats
import cartopy.crs as ccrs # cartopy projections
import os
import glob #for file searching 
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
import scipy

#makes plots prettier
sns.set_theme()
sns.set_style()

xr.set_options(keep_attrs=True) # keep attributes when doing arithmetic. 

# Southern Europe coordinates
min_lon = 15
min_lat = 34
max_lon = 40 
max_lat = 45

# This function is used for splicing to summer months, between March and September.
def is_summer(month):
    return (month >= 5) & (month < 10)

def gauss_old(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def poiss(x, *p):
    mu = p[0]
    return (np.exp(-mu)* mu**x / scipy.special.factorial(x))

def heatind(TK, RH):
    """
    should work fine with arrays, or single values, but make sure Temp and Humidity correspond properly
    Args:
        T (array?): Temperatures (Kelvin)
        RH (array?): Relative humidities (percentage) INTEGER PERCENTAGE
    """
    # define constants for the equation
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6    
    
    # convert Kelvin to Fahrenheit
    TF = (TK-273.15)*9/5 + 32
    HIF = c1 + c2*TF + c3*RH + c4*TF*RH + c5*(TF**2) + c6*(RH**2) + c7*(TF**2)*RH + c8*TF*(RH**2) + c9*(TF**2)*(RH**2)

    def adj(RH,TF,HIF):
        HIFadj = xr.where((RH<13) & (TF>80) & (TF<112), HIF-((13-RH)/4)*np.sqrt((17-np.abs(TF-95.))/17), HIF)
        HIFadj = xr.where((RH>85) & (TF>80) & (TF<87), HIF+((RH-85)/10)*((87-TF)/5), HIF)
        return HIFadj
    HIF = xr.where((RH>13) & (RH<85), HIF, adj(RH, TF,HIF))

    # convert heat index in fahrenheit to Kelvin
    HIK = (HIF-32)*5/9 + 273.15

    return HIK

class CombineYearlyFiles:

    def __init__(self,sim_dir_loc, output_path, name):
        """
        This takes a little while. So the xarray is saved at output_path.

        inputs:
        sim_dir_loc : (str) directory location where sim data is kept. ie. "/work/ta116/shared/users/eleanorsenior/cesm/archive/HistoricalNEW/atm/hist/"
        output_path : (str) dir loc to save output xarray.
        name : (str) what to call this sim, is how the output .nc xarray is saved

        returns:
        xarray dataset saved at output_path
        self.num_years : number of years used for simulation run. This is needed for threshold computation. 

        """

        year_files = sorted(glob.glob(sim_dir_loc + "*.cam.h1.*")) # these are the yearly files. We sort them because the last file (2015) has no summer months and breaks the code.

        # We open the first file and use this as the array we'll concatenate everything else to
        file0=year_files[0]
        ds=xr.open_dataset(file0) 

        ds_SE = ds.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon), time=is_summer(ds['time.month'])) #cropping to Southern Europe and summer

        for ind, year_f in enumerate(year_files[1:-1]): # The last file only contains January 2015 (not winter). And we've used the first file already.
            ds_year = xr.open_dataset(year_f)  # opening data from that year

            if np.all(is_summer(ds_year['time.month']) == False):
                raise Exception("One of the files doesn't contain any summer months... A workaround is needed.")
            ds_year_SE = ds_year.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon),  time=is_summer(ds_year['time.month']))
            ds_SE = xr.concat([ds_SE, ds_year_SE], dim="time") 
            print(f"Finished {ind + 1}/{len(year_files[1:-1]) } years!")

        ds_SE.sortby(ds_SE.time) # sorting the data so time is ascending. (Probably unnecessary, but kinda nice)

        ds = ds_SE.sortby(ds_SE.time)

        ds.to_netcdf(output_path + name + ".nc")

        print(f"Yearly files have been cropped, combined, and saved to {output_path + name}.nc")

        self.num_years = len(year_files)

        #return self.num_years

class SingleRun:
    """
    For looking at single runs
    """
    def __init__(self,sim_dir_loc, output_path, name):
        """
        reads in .nc file containing data for a single run
        """
        # loading in xarray we saved in combineYears
        self.ds = xr.open_dataset(output_path + name + ".nc", engine='h5netcdf') 

class Ensemble:
    """
    Combines data from different simulation sims 

    inputs:
       output_path : (str)  dir locs where .nc datasets (already cropped to summer and SE) are stored. ie ('./Data/Historical2023/')
       names : (lst) list of names of ensemble runs, without '.nc' ending (ie. ['jubauer_0', 's1935349_0',...])

    """
    def __init__(self, output_path, names):
        #lf = xr.open_dataset("/work/ta116/shared/users/tetts_ta/cesm/cesm_inputdata/atm/cam/topo/fv_1.9x2.5_nc3000_Nsw084_Nrs016_Co120_Fi001_ZR_GRNL_031819.nc") 
        #landfracs = lf.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon)).LANDFRAC
        ensemble_ds = xr.open_dataset(output_path + names[0] + ".nc", engine='h5netcdf') 

        for i, name in enumerate(names[1:]):
            sim_new = xr.open_dataset(output_path + names[i] + ".nc", engine='h5netcdf') #* landfracs / landfracs.sum()
            ensemble_ds = xr.concat([ensemble_ds, sim_new], dim = "run") 
        
        #self.ds = ensemble_ds
        self.ds = ensemble_ds#.mean(dim=("run"))  #averaging over runs
        self.no_members = len(names)

class HistogramAnalysis:
    """
    This is the class to use for fitting data. 

    1. For each spatial gridpoint, calculate how many days per 15-day period are above a HI-cutoff (This value hasn't been decided upon yet -  will change after bias correction. For now we use 305.372K)
    2. Take a weighted average over lat and lon. We do this by multiplying by land fractions and dividing by the sum of land fractions
    3. Bin data into 1-day bins (may change?) to get a histogram
    4. Fit a gaussian to the histogram
    
    inputs:

    ensemble : Ensemble OR SingleRun object, with .ds 
    """

    def __init__(self, ensemble):
        self.ds = ensemble.ds
        
        
        #The Heat Index cutoff (305.372K is the Extreme Caution category)
        #HI_cutoff =  305.372 # 312.594 (danger cutoff) 

        heatind_overLand = heatind(self.ds.TREFHTMX,  self.ds.RHREFHT) #calculate HI at all grid points

        #bimonthly = heatind_overLand.where(heatind_overLand > HI_cutoff).resample(time='15D').count() #counting number of days/15 of HI>extreme caution

        # Weighting by landfrac. Essentially a weighted average over lon and lat
        lf = xr.open_dataset("/work/ta116/shared/users/tetts_ta/cesm/cesm_inputdata/atm/cam/topo/fv_1.9x2.5_nc3000_Nsw084_Nrs016_Co120_Fi001_ZR_GRNL_031819.nc") 
        landfracs = lf.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon)).LANDFRAC

        #self.variable_ds = (bimonthly * landfracs).sum(dim=("lat","lon"))  / landfracs.sum().data # weighted spatial average
        #self.variable_ds.attrs["long_name"] = f"#Days per 15 with average HI>{HI_cutoff}" # for correct labelling

        spatially_averaged_HI = (heatind_overLand * landfracs).sum(dim=("lat","lon"))  / landfracs.sum().data # weighted spatial average
        self.variable_ds = spatially_averaged_HI.groupby('time.year').max() #taking the maximum value of the summer
        self.variable_ds.attrs["long_name"] = f"#Maximum SE-averaged HI per summer" # for correct labelling
        

    def binData(self, binspacing = 1, plot=False):
        
        # Creating bins for data. 
        min_val, max_val = self.variable_ds.min(), self.variable_ds.max()
        bins = np.arange(min_val, max_val, binspacing) #creating bins
        bin_centres = (bins[:-1] + bins[1:])/2

        # Putting data into our bins
        grouped_data =  self.variable_ds.groupby_bins(self.variable_ds, bins, labels=bin_centres) 
        
        counts = (grouped_data.count()).fillna(0)# filling in NaNs
        
        self.total = counts.sum().data # total number of counts

        probs = counts/self.total # converting to a probability, so we can get a PDF

        #if np.sum(probs).data != 1.0: # sanity check 
        #    raise Exception(f"The sum of probabilities is {np.sum(probs).data}! It should be 1.0.")
        
        if plot:
            plt.bar(bin_centres, probs, label = f"Simulated data", alpha = 0.8)
            #plt.show()
        
        self.bin_centres = bin_centres
        self.PDF = probs
    
    def fitBinnedData(self, fit_type = "Gaussian", plot=False, ensemble_name=None, color=None):
        """
        This has to be run after binData

        plt : (boolean) whether to gaussian fit to histogram
        """
        self.fit_type = fit_type
        if fit_type == "Gaussian":
            gauss = lambda x, loc, scale :  stats.norm.pdf(x, loc=loc, scale=scale) 
            
            mu = np.mean(self.variable_ds)
            p0 = [mu, 10.] #loc, scale (mu, sigma?)
            
            coeff, var_matrix = curve_fit(gauss, self.bin_centres, self.PDF , p0=p0)
            print(coeff)

            hist_fit = gauss(self.bin_centres, *coeff)

            coeff, var_matrix = curve_fit(gauss, self.bin_centres, self.PDF , p0=p0)
            hist_fit = gauss(self.bin_centres, *coeff)

            if plot:
                plt.bar(self.bin_centres, self.PDF, label =f"Simulated Data {ensemble_name}", color=color, alpha = 0.8)
                plt.plot(self.bin_centres, hist_fit, label =f"Gaussian fit {ensemble_name}", color=color)
            
            print('Fitted mean = ', coeff[0])
            print('Fitted standard deviation = ', coeff[1])
            #print('Fitted amp = ', coeff[0])

            mean  = coeff[0]
            stdev = coeff[1]
            #amp = coeff[0]

            self.coeff = coeff

        elif fit_type == "GEV":
            gev = lambda x, c, loc, scale : stats.genextreme.pdf(x, c, loc, scale)
            loc0 = np.mean(self.variable_ds)
            p0 = [-.8, loc0, 1]

            coeff, var_matrix = curve_fit(gev, self.bin_centres, self.PDF , p0=p0)

            hist_fit =  gev(self.bin_centres, *coeff)

            if plot:
                plt.plot(self.bin_centres, hist_fit, label =f"GEV fit Ensemble {ensemble_name}",color=color)
                plt.bar(self.bin_centres, self.PDF, label =f"Simulated Data  {ensemble_name}", color=color, alpha = 0.8)
            
            print('Fitted c = ', coeff[0])
            print('Fitted loc = ', coeff[1])
            print('Fitted scale = ', coeff[2])

            c  = coeff[0]
            loc = coeff[1]
            scale = coeff[2]

            self.coeff = coeff

        elif fit_type == "Poisson":
            poisson = lambda x, mu, loc : stats.poisson.pmf(x, mu, loc)

            p0 = [np.mean(self.variable_ds)]

            #coeff, var_matrix = curve_fit(poisson, self.bin_centres, self.PDF , p0=[3, np.mean(self.variable_ds)])
            #hist_fit =  poisson(self.bin_centres, *coeff)

            coeff, var_matrix = curve_fit(poiss, self.bin_centres, self.PDF , p0=p0)
            hist_fit =  poiss(self.bin_centres, *coeff)

            if plot:
                plt.bar(self.bin_centres, self.PDF, label =f"Simulated Data  {ensemble_name}", color=color, alpha = 0.8)
                plt.plot(self.bin_centres, hist_fit, label =f"Poisson fit Ensemble  {ensemble_name}", color=color)
            
            print('Fitted mu = ', coeff[0])
            #print('Fitted loc = ', coeff[1])

            mu = coeff[0]

            self.coeff = coeff
        
        elif fit_type == "Pareto":
            pareto = lambda x, b, loc, scale : stats.pareto.pdf(x, b, loc, scale)  

            loc0 = np.mean(self.variable_ds)
            p0 = [0.8, -0.1, 1]

            coeff, var_matrix = curve_fit(pareto, self.bin_centres, self.PDF , p0=p0)

            hist_fit =  pareto(self.bin_centres, *coeff)

            if plot:
                plt.plot(self.bin_centres, hist_fit, label =f"Pareto fit Ensemble  {ensemble_name}", color=color)
                plt.bar(self.bin_centres, self.PDF, label =f"Simulated Data  {ensemble_name}", color=color, alpha = 0.8)
            
            print('Fitted b = ', coeff[0])
            print('Fitted loc = ', coeff[1])
            print('Fitted scale = ', coeff[2])

            b  = coeff[0]
            loc = coeff[1]
            scale = coeff[2]

            self.coeff = coeff
        
        else:
            raise Exception("Sorry, you're gonna have to code this..")

    def getThreshold(self, threshold = 0.9, plot=False, ensemble_name = None,  color=None):
        """
        This has to be run after fitBinnedData

        """
        if self.fit_type == "Gaussian":
            thr = stats.norm.ppf(threshold, self.coeff[0], self.coeff[1])
            
        elif self.fit_type == "Poisson":
            thr = stats.poisson.ppf(threshold, self.coeff[0], self.coeff[1])

        elif self.fit_type == "GEV":   
            thr = stats.genextreme.ppf(threshold, self.coeff[0], self.coeff[1], self.coeff[2])

        elif self.fit_type == "Pareto":
            thr = stats.pareto.ppf(threshold, self.coeff[0], self.coeff[1], self.coeff[2])

        print(thr)
        if plot:
            plt.axvline(x=thr, linestyle="dashed", label = f"{ensemble_name} {threshold*100}% ", color=color)

        return thr




###INPUTS (Everything below is me trying to get results I needed, feel free to delete/ reformat to suit needs)




#output_path = "/home/ta116/ta116/s1935349/analysisCode/Data/Historical/"
output_path = "/work/ta116/shared/users/eleanorsenior/analysis/Data/Historical/"
ensemble_name = output_path.split('/')[-2]
fit_type = "Gaussian"
lst = [os.listdir(output_path)][0]
lst.sort()

names = []
color = 'darkseagreen'
for i, f in enumerate(lst):
    name = f.split('.')[0]
    names = names + [name]

Ens = Ensemble(output_path, names)
hist = HistogramAnalysis(Ens)
hist.binData(plot=False)
hist.fitBinnedData(fit_type = fit_type,plot=False, ensemble_name=ensemble_name, color=color)
thr = hist.getThreshold(threshold=0.9,plot=True, ensemble_name=ensemble_name, color=color)
coeffs = hist.coeff
#plt.title(f"Historical Ensemble")
#print(thr)

#xs = np.linspace(hist.variable_ds.min(), hist.variable_ds.max(), 100)
#ys = stats.norm.pdf(xs, coeffs[0], coeffs[1]) 
#plt.plot(xs, ys)

#print(f"Mean historical2023 value ({hist_2023.variable_ds.mean()}) is the ",stats.norm.cdf(hist_2023.variable_ds.mean(), coeffs[0], coeffs[1]), "percentile?" )


output_path = "/home/ta116/ta116/s1935349/analysisCode/Data/PI_2023/" #change this to analyse a different dataset in ./Data directory
#output_path = "/work/ta116/shared/users/eleanorsenior/analysis/Data/PI_2023/"
fit_type = "Gaussian"
ensemble_name = output_path.split('/')[-2]

lst = [os.listdir(output_path)][0]
lst.sort()

names = []
for i, f in enumerate(lst):
    name = f.split('.')[0]
    names = names + [name]
color='goldenrod'
Ens = Ensemble(output_path, names)
hist_PI = HistogramAnalysis(Ens)
hist_PI.binData(plot=False)
hist_PI.fitBinnedData(fit_type = fit_type,plot=False, ensemble_name=ensemble_name, color=color)
#thr_PI = hist_PI.getThreshold(threshold=0.9, plot=False, ensemble_name=ensemble_name, color=color)
#coeffs = hist_PI.coeff
#plt.title(f"{ensemble_name}")
xs = np.linspace(hist_PI.variable_ds.min(), hist_PI.variable_ds.max(), 100)
ys = stats.norm.pdf(xs, hist_PI.coeff[0], hist_PI.coeff[1]) 
plt.plot(xs, ys, color = color, label = f"{ensemble_name + fit_type} fit")
plt.fill_between(xs, ys, where=(xs>thr), color=color, alpha=0.5)

output_path = "/home/ta116/ta116/s1935349/analysisCode/Data/Historical2023/"
#output_path = "/work/ta116/shared/users/eleanorsenior/analysis/Data/Historical2023/"
ensemble_name = output_path.split('/')[-2]

lst = [os.listdir(output_path)][0]
lst.sort()

names = []
color = 'cornflowerblue'
for i, f in enumerate(lst):
    name = f.split('.')[0]
    names = names + [name]

Ens = Ensemble(output_path, names)
hist_2023 = HistogramAnalysis(Ens)
hist_2023.binData(plot=False)
hist_2023.fitBinnedData(fit_type = fit_type,plot=False, ensemble_name=ensemble_name, color=color)

xs = np.linspace(hist_2023.variable_ds.min(), hist_2023.variable_ds.max(), 100)
ys = stats.norm.pdf(xs, hist_2023.coeff[0], hist_2023.coeff[1]) 
plt.plot(xs, ys, color = color, label = f"{ensemble_name + fit_type} fit")
plt.fill_between(xs, ys, where=(xs>thr), color=color, alpha=0.5)
plt.title(f"Comparing Historical2023 and PI_2023")



hist_2023_percentile = stats.norm.cdf(thr, hist_2023.coeff[0], hist_2023.coeff[1])
PI_2023_percentile = stats.norm.cdf(thr, hist_PI.coeff[0], hist_PI.coeff[1])

print(f"Our threshold value is: {thr}" )
print(f"PI_2023 percentage above threshold: {100*(1-PI_2023_percentile)}%")
print(f"Historical2023 percentage above threshold: {100*(1-hist_2023_percentile)}%")


plt.legend()
plt.show()





####################################################### EVERYTHING FROM HERE ON IS OUTDATED ##################################################################################

"""

output_path = '/home/ta116/ta116/s1935349/analysisCode/Data/Historical2023/'
sim_parent_path = '/work/ta116/shared/users/jubauer/cesm/archive/'
lst = [os.listdir('/work/ta116/shared/users/jubauer/cesm/archive/')][0]
lst.sort()
names = []
sim_paths = []
sim_name = []
for i in range(9,25):
    sim_paths += [sim_parent_path+f"Historical2023_{i}/atm/hist/"]
    names = names+ ["jub_"+str(i)]

#broken jubauer sims: 5,8
#for i, sim_path in enumerate(lst):
#        sim_dir_loc = sim_parent_path +'/'+ sim_name[i] + "/atm/hist/"
#        sim_paths = sim_paths+[sim_dir_loc]
        

#print(names, sim_paths)

#sim_paths = ['/work/ta116/shared/users/tetts_ta/cesm/archive/PI_2023/atm/hist/']
for ind, name in enumerate(names):
    CombineYearlyFiles(sim_paths[ind], output_path, name)


output_path = "/home/ta116/ta116/s1935349/analysisCode/Data/PI_2023/"

sim_parent_path = '/work/ta116/shared/users/s1946411/cesm/archive/'
lst = [os.listdir('/work/ta116/shared/users/s1946411/cesm/archive/')][0]
lst.sort()
names = []
sim_paths = []

for i in range(1,26):
    sim_paths += [sim_parent_path+f"PI_2023_{i}/atm/hist/"]
    names = names+ ["tom_"+str(i)]

for ind, name in enumerate(names):
    CombineYearlyFiles(sim_paths[ind], output_path, name)

output_path = '/home/ta116/ta116/s1935349/analysisCode/Data/Historical_2023/'
sim_parent_path = '/work/ta116/shared/users/jubauer/cesm/archive/'
lst = [os.listdir('/work/ta116/shared/users/jubauer/cesm/archive/')][0]
lst.sort()
names = []
sim_paths = []
sim_name = []
for i in range(1,25):
    sim_paths += [sim_parent_path+f"Historical2023_{i}/atm/hist/"]
    names = names+ ["jubauer_"+str(i)]

#for i, sim_path in enumerate(lst):
#        sim_dir_loc = sim_parent_path +'/'+ sim_name[i] + "/atm/hist/"
#        sim_paths = sim_paths+[sim_dir_loc]
        

print(names, sim_paths)

#sim_paths = ['/work/ta116/shared/users/tetts_ta/cesm/archive/PI_2023/atm/hist/']
for ind, name in enumerate(names):
    CombineYearlyFiles(sim_paths[ind], output_path, name)


output_path = "/home/ta116/ta116/s1935349/analysisCode/Data/Historical/"

## We're missing one! also sorry for the clunkiness
eleanor_sim_path = "/work/ta116/shared/users/eleanorsenior/cesm/archive/HistoricalNEW/atm/hist/"
tom_sim_path = "/work/ta116/shared/users/s1946411/cesm/archive/Historical/atm/hist/"
simon1_sim_path = '/work/ta116/shared/users/tetts_ta/cesm/archive/FHIST_1982_2014_OSTIA/atm/hist/'
simon2_sim_path = '/work/ta116/shared/users/tetts_ta/cesm/archive/FHIST_1982_2014_OSTIA_a/atm/hist/'

names = ["eleanorMAYSUMMER","tomMAYSUMMER","simon1MAYSUMMER","simon2MAYSUMMER"]
#sim_paths = [eleanor_sim_path, tom_sim_path, simon1_sim_path, simon2_sim_path ] 

Ens = Ensemble(output_path, names)
#Ens.ds.TREFHTMX.isel(lat=2, lon=2).plot.line()
hist = HistogramAnalysis(Ens)
hist.binData(plot=False)
hist.getThreshold()
hist.fitBinnedData(fit_type = "Poisson",plot=True)
plt.legend()
plt.show()
"""

"""
output_path = '/home/ta116/ta116/s1935349/analysisCode/Data/PI_2023/'
sim_parent_path = '/work/ta116/shared/users/eleanorsenior/cesm/archive/PI_2023_ensemble'
lst = [os.listdir('/work/ta116/shared/users/eleanorsenior/cesm/archive/PI_2023_ensemble')][0]
lst.sort()
names = []
sim_paths = []
for i, sim_path in enumerate(lst):
        sim_dir_loc = sim_parent_path +'/'+ sim_path + "/atm/hist/"
        sim_paths = sim_paths+[sim_dir_loc]
        names = names+ ["eleanor_"+str(i)]
#print(names, sim_paths)
names = ['tetts_ta_0']
sim_paths = ['/work/ta116/shared/users/tetts_ta/cesm/archive/PI_2023/atm/hist/']
for ind, name in enumerate(names):
    CombineYearlyFiles(sim_paths[ind], output_path, name)

names = []
sim_paths = []
base = "/work/ta116/shared/users/s1946411/cesm/archive/Historical2023_"
for i in range(1,26):
    sim_paths = sim_paths + [base+str(i)+"/atm/hist/" ]
    names = names + [f'tom_{i}']

print(sim_paths[1])
for ind, name in enumerate(names):
    CombineYearlyFiles(sim_paths[ind], output_path, name)


means = np.zeros(len(names))
stdevs = np.zeros(len(names))
amplitudes = np.zeros(len(names))

for ind, name in enumerate(names):
    sim = SingleRun(sim_paths[ind], output_path, str(name + "MAYSUMMER"))
    ## UNCOMMENT LINE BELOW IF WANTING TO GENERATE A NEW .nc FILE.. takes a while
    #sim.combineYears()
    sim.maxTempsHistogram(plot = False)
    means[ind], stdevs[ind], amplitudes[ind] = sim.fitMaxTemps()

ensemble_mean = np.mean(means)
ensemble_stdev = np.mean(stdevs)
ensemble_amp = np.mean(amplitudes)

Ts = np.arange(0,45,0.1)
ensemble_fit = gauss_old(Ts, ensemble_amp, ensemble_mean, ensemble_stdev)

plt.plot(Ts, ensemble_fit, label ="Gaussian fit ENSEMBLE")


plt.legend()
plt.show()
"""
"""
class Ensemble:

#    Combines data from different simulation sims

 #   inputs:
 #       SingleRuns : (lst) list of SingleRun datasets.

 #   output(?) is an xarray dataset(?)

    def __init__(self, SingleRuns):

        for sim_ds in SingleRuns:


"""       

"""
    def maxTempsHistogram(self, plot=False):

        Bins temperatures into bins of 0.1ºC

        plt : (boolean) whether to plot histograM

        # loading in xarray we saved in combineYears
        self.ds=xr.open_dataset(self.output_path + self.name + ".nc", engine='h5netcdf') 

        # Here we're converting to Celsius. I just found this easier to sanity check than Kelvin
        self.maxtemps = self.ds.TREFHTMX - 273.15 
        self.maxtemps.attrs = self.ds.TREFHTMX.attrs 
        self.maxtemps.attrs["units"] = "ºC"

        #Now we create bins for our temperatures. We're using bins of width 0.1ºC
        mintemp = self.maxtemps.min()
        maxtemp = self.maxtemps.max()
        bins = np.arange(mintemp, maxtemp, 0.1) #creating bins
        bin_centres = (bins[:-1] + bins[1:])/2

        #Grouping the maxTemps into our bins
        self.grouped_temps  = self.maxtemps.groupby_bins(self.maxtemps, bins, labels=bin_centres) 

        counts = (self.grouped_temps.count()).fillna(0)# filling in NaNs

        total = np.sum(counts).data # total number of counts

        counts_10years = total/self.num_years*10 
        self.prob1_10yr = 1/(counts_10years) #probability for 1 in 10 years

        print(self.maxtemps.quantile(1-self.prob1_10yr))

        probs = counts/total # converting to a probability, so we can get a PDF

        if np.sum(probs).data != 1.0: # sanity check 
            raise Exception(f"The sum of probabilities is {np.sum(probs).data}! It should be 1.0.")
        
        if plot:
            plt.bar(bin_centres, probs, label = f"Sim data {self.name}")
            #plt.show()
        
        self.bin_centres = bin_centres
        self.maxTempPDF = probs
    
    def fitMaxTemps(self, fit_type = "Gaussian", plot=False):

        This has to be run after maxTempsHistogram ... 

        plt : (boolean) whether to gaussian fit to histogram

        if fit_type == "Gaussian":
            mu = np.mean(self.maxtemps)
            p0 = [1., mu, 10.] #A, mu, sigma
            coeff, var_matrix = curve_fit(gauss, self.bin_centres, self.maxTempPDF , p0=p0)

            hist_fit = gauss(self.bin_centres, *coeff)

            if plot:
                plt.plot(self.bin_centres, hist_fit, label =f"Gaussian fit {self.name}")
            
            print('Fitted mean = ', coeff[1])
            print('Fitted standard deviation = ', coeff[2])
            print('Fitted amp = ', coeff[0])

            mean  = coeff[1]
            stdev = coeff[2]
            amp = coeff[0]

            return mean, stdev, amp
            
        else:
            raise Exception("Sorry, you're gonna have to code this..")


    def combineYears(self):

        This takes a little while. So the xarray is saved at output_path.


        year_files = sorted(glob.glob(self.sim_dir_loc + "*.cam.h1.*")) # these are the yearly files. We sort them because the last file (2015) has no summer months and breaks the code.
        self.num_years = len(year_files)
        # We open the first file and use this as the array we'll concatenate everything else to
        file0=year_files[0]
        ds=xr.open_dataset(file0) 

        ds_SE = ds.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon), time=is_summer(ds['time.month'])) #cropping to Southern Europe and summer

        for ind, year_f in enumerate(year_files[1:-1]): # The last file only contains January 2015 (not winter). And we've used the first file already.
            ds_year = xr.open_dataset(year_f)  # opening data from that year

            if np.all(is_summer(ds_year['time.month']) == False):
                raise Exception("One of the files doesn't contain any summer months... A workaround is needed.")
            ds_year_SE = ds_year.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon),  time=is_summer(ds_year['time.month']))
            ds_SE = xr.concat([ds_SE, ds_year_SE], dim="time") 
            print(f"Finished {ind + 1}/{len(year_files[1:-1]) } years!")

        ds_SE.sortby(ds_SE.time) # sorting the data so time is ascending. (Probably unnecessary, but kinda nice)

        ds = ds_SE.sortby(ds_SE.time)

        ds.to_netcdf(self.output_path + self.name + "MAYSUMMER.nc")
### OLD CODE FOR PLOTTING GLOBES AND OTHER PRETTY THINGS #####


eleanor_sim



maxtemps = (ds_SE.TREFHTMX - 273.15).sel(lat=37, lon=20, method='nearest')
maxtemps.attrs = ds_SE.TREFHTMX.attrs 
maxtemps.attrs["units"] = "ºC"

#cropping for area, time of year
#cropped_ds = ds.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon), time=slice(start_time, end_time))
#cropped_ds = ds.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
#We're looking at max temperature
temps = (ds_SE.TREFHTMX - 273.15).sel(lat=37, lon=20, method='nearest')

# copy attributes to get nice figure labels and change Kelvin to Celsius
temps.attrs = ds_SE.TREFHTMX.attrs 

temps.attrs["units"] = "ºC"

#nbin=100
#hist, edges=np.histogram(temps,bins=nbin) #makes two arrays

#xr.plot.hist(temps, xlim=np.array([260,325]))
temps.plot()
plt.show()


levels = np.arange(250,320.,2) # levels for plotting. 2x normal pressure diffs
"""
"""
fig = plt.figure(figsize=[11,12],clear=True,num='MaxTemp')

time_index = [0,-1] # 1st and 10th days
ntime= len(time_index)
for count_time,time_indx in enumerate(time_index):  # iterate ove time

    dataArray=cropped_ds.TREFHTMX.isel(time=time_indx).load() 
    # get a max temp field. load -- means read it

    projections=[ccrs.PlateCarree(),ccrs.NorthPolarStereo()] # projections to use.
    nproj = len(projections)
    for count,proj in enumerate(projections):
        ax=fig.add_subplot(ntime,2,(count+1)+count_time*nproj,projection=proj)
        if isinstance(proj,ccrs.NorthPolarStereo):
        # Limit the map to 30 degrees latitude and above
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], ccrs.PlateCarree())
        cm=(dataArray).plot(levels=levels, ax=ax,transform=ccrs.PlateCarree(),
                                 cbar_kwargs=dict(orientation='horizontal'),
                                 # make colour bar horizontal
                             ) 
        #cs=(dataArray).plot.contour(levels=levels, 
        #                                 linestyles='solid', colors='pink',
        #                                 ax=ax,transform=ccrs.PlateCarree(),)
        # make colour plot
        # overlay contours
        #ax.clabel(cs,cs.levels,inline=True,fmt="%d",fontsize=8) # add contour labels
        ax.gridlines()
        ax.coastlines(linewidth=2,color='white')
fig.show()
fig.savefig("maxTemp.png")


p = (cropped_ds.TREFHTMX.isel(time=0)-273.15).plot(
    subplot_kws=dict(projection=ccrs.Orthographic(min_lat, max_lat), facecolor="gray"),
    transform=ccrs.PlateCarree(),
)


p.axes.set_global()

p.axes.coastlines()
plt.draw()

plt.show()

maxtemp  = Ens.ds.where(landfracs.LANDFRAC, drop=True).TREFHTMX.mean(dim="run")
.isel(time=0)
p = Ens.ds.where(landfracs.LANDFRAC,  drop=True).TREFHTMX.isel(time=0).mean(dim="run").plot(
    subplot_kws=dict(projection=ccrs.Orthographic(min_lat, max_lat), facecolor="gray"),
    transform=ccrs.PlateCarree(),
)
p.axes.set_global()

p.axes.coastlines()
plt.draw()

plt.show()



        #)* landfracs).sum(dim=("lat","lon"))  / landfracs.sum().data  
        #print(heatind_overLand)
        #self.variable_ds = heatind_overLand
        #self.no_members = ensemble.no_members 

        #rolling_average = self.variable_ds.rolling(time=3).mean()
        #self.variable_ds = self.rolling_average
        
        # Group by year and calculate the maximum rolling average for each year
        #max_rolling_per_year = rolling_average.groupby('time.year').max()
        #self.variable_ds = max_rolling_per_year

        #print("Maximum rolling average per year:", max_rolling_per_year)
        
        #self.ds.TREFHTMX  #CHANGE THIS IF WANTING TO LOOK AT A DIFFERENT VARIABLE
         
        # Here we're converting to Celsius. I just found this easier to sanity check than Kelvin
        #self.variable_ds = self.ds.TREFHTMX - 273.15 
        #self.variable_ds.attrs = self.ds.TREFHTMX.attrs 
        #self.variable_ds.attrs["units"] = "ºC"

        #num_years = self.ds["time.year"][-1] - self.ds["time.year"][0] #number of years in simulation

        #counts_10years = self.total / num_years * 10 # number of data points per 10 years
        #prob1_10yr = 1/(counts_10years) # probability for 1 in 10 years
        self.threshold = self.variable_ds.quantile(1-.95).data
        print(f"The 1/10 year value is {self.threshold}")

"""
