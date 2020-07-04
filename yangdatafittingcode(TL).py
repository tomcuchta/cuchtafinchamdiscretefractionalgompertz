import scipy
import scipy.stats
import scipy.optimize as so
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import math
import csv 

##########################
##########################
#Peculiarities to data set
##########################
##########################

global t0logfrac
global t0yfrac
global t0yfraclogfrac
global t0atici

###########################
### P.elegansTL INFORMATION
###########################
csvfilename="P.elegansMaleTL.csv"
graphsavename="pelegansTL.eps"
csv_xlabel="Weeks"
csv_ylabel="Tail length (centimeters)"
### Initial conditions for each model
t0logfrac=1 # choose the smallest value of the independent variable here
## DONT CHANGE THESE THREE
t0yfrac=t0logfrac-1
t0yfraclogfrac=t0logfrac-1
t0atici=t0logfrac-1 #atici model uses same initial condition as logfrac

### Initial parameter guesses for the models
logfrac_params_guess=(0.8,-0.6,0.09,3)
yfrac_params_guess=(0.9,-3.5,0.035,3)
yfrac_logfrac_params_guess=(0.9,-1.9,0.2,3)
aticiguessparams=[1,0.1,0.02,4]

######################
######################
# Functions and models
######################
######################

################################
# Define nabla monomial function
################################
def hnu(t,t0,nu):
    if isinstance(t-t0,int) and t-t0<0:
        return 0
    else:
        return gamma(t-t0+nu)/(gamma(nu+1)*gamma(t-t0))
    
###########################################################################################
# Define discrete fractional Gompertz with y non-fractional, logarithm fractional, aka (12)
###########################################################################################
zsoln_memo={}
def zsoln(t,nu,r,a):
    if t==t0logfrac:
        return -r*a
    elif t>t0logfrac:
        currentKey=str(t)+":"+str(nu)+":"+str(r)+":"+str(a)
        if currentKey in zsoln_memo:
            return zsoln_memo[currentKey]
        else:
            sumresult=-r*a - (1.0-r)*sum([(gamma(t-k-nu)/(gamma(-nu)*gamma(t-k+1)))*zsoln(k,nu,r,a) for k in range(int(t0logfrac),int(t))])
            zsoln_memo[currentKey]=sumresult
            return sumresult
def p(t,nu,r,a):
    return (-r/(1.0-r))*(a+zsoln(t,nu,r,a))

def gompertz_logfrac(t,nu,a,r,y0):
    if t==t0logfrac:
        return y0
    if t>t0logfrac:
        return y0*np.prod([1/(1-p(k,nu,r,a)) for k in range(int(t0logfrac),int(math.floor(t))+1)])

def gompertz_logfrac_fit(x,nu,a,r,y0):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i]=gompertz_logfrac(x[i],nu,a,r,y0)
    return y

###########################################################################################
# Define discrete fractional Gompertz with y fractional, logarithm non-fractional, aka (13)
###########################################################################################
gompertz_yfrac_memo={}
def gompertz_yfrac(t,nu,a,r,y0):
    if t==t0yfrac:
        return 0
    if t==t0yfrac+1:
        return y0
    if t>t0yfrac+1:
        currentKey=str(t)+":"+str(nu)+":"+str(a)+":"+str(r)+":"+str(y0)
        if currentKey in gompertz_yfrac_memo:
            return gompertz_yfrac_memo[currentKey]
        else:
            ominusr=-r/(1-r)
            numeratorint=sum(hnu(t,k-1,-nu-1)*gompertz_yfrac(k,nu,a,r,y0) for k in range(int(t0yfrac)+1,int(t)))
            numerator=(ominusr-1)*numeratorint
            denomint=sum((1/gompertz_yfrac(tau,nu,a,r,y0))*sum(hnu(tau,s-1,-nu-1)*gompertz_yfrac(s,nu,a,r,y0) for s in range(int(t0yfrac)+1,int(tau)+1)) for tau in range(int(t0yfrac)+1,int(t)))
            denominator=1-ominusr*(a+1+denomint)
            gompertz_yfrac_memo[currentKey]=numerator/denominator
            return numerator/denominator

def gompertz_yfrac_fit(x,nu,a,r,y0):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i]=gompertz_yfrac(x[i],nu,a,r,y0)
    return y

#######################################################################################
# Define discrete fractional Gompertz with y fractional, logarithm fractional, aka (14)
#######################################################################################
gompertz_yfrac_logfrac_memo={}
def gompertz_yfrac_logfrac(t,nu,a,r,y0):
    if t==t0yfraclogfrac:
        return 0
    if t==t0yfraclogfrac+1:
        return y0
    if t>t0yfraclogfrac+1:
        currentKey=str(t)+":"+str(nu)+":"+str(a)+":"+str(r)+":"+str(y0)
        if currentKey in gompertz_yfrac_logfrac_memo:
            return gompertz_yfrac_logfrac_memo[currentKey]
        else:
            ominusr=-r/(1-r)
            numeratorint=sum(hnu(t,k-1,-nu-1)*gompertz_yfrac_logfrac(k,nu,a,r,y0) for k in range(int(t0yfraclogfrac)+1,int(t)))
            numerator=(ominusr-1)*numeratorint
            denominatorint=sum((1/gompertz_yfrac_logfrac(tau,nu,a,r,y0))*hnu(t,tau-1,nu-1)*sum(hnu(tau,s-1,-nu-1)*gompertz_yfrac_logfrac(s,nu,a,r,y0) for s in range(int(t0yfraclogfrac)+1,tau+1)) for tau in range(int(t0yfraclogfrac)+1,int(t)))
            denominator=1-ominusr*(1+a+denominatorint)
            gompertz_yfrac_logfrac_memo[currentKey]=numerator/denominator
            return numerator/denominator

def gompertz_yfrac_logfrac_fit(x,nu,a,r,y0):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i]=gompertz_yfrac_logfrac(x[i],nu,a,r,y0)
    return y


################################
# Define Atici Gompertz, aka (3)
################################
atici_memo={}
def aticigompertz(t,c,b,nu,y0):
    if(t==t0atici):
        return (c-b*np.log(y0))*y0
    else:
        currentKey=str(t)+":"+str(c)+":"+str(b)+":"+str(nu)+":"+str(y0)
        if currentKey in atici_memo:
            return atici_memo[currentKey]
        else:
            sumresult=(c-b*np.log(aticigompertz(t-1,c,b,nu,y0)))*aticigompertz(t-1,c,b,nu,y0)-sum([(gamma(t-1-nu-s)*aticigompertz(s+1,c,b,nu,y0))/(gamma(t-s)*gamma(-nu)) for s in range(0,int(t-1))])
            atici_memo[currentKey]=sumresult
            return sumresult
           
def aticifitfunc_vec_self(x,c,b,nu,y0):
  y = np.zeros(x.shape)
  for i in range(len(y)):
    y[i]=aticigompertz(x[i],c,b,nu,y0)
  return y

##########################
# Define Yang ctn Gompertz 
##########################
def ctngompertz(t,A,B,K):
    return A*np.exp(-B*np.exp(-K*t))

#######################################################
#######################################################
#######################################################
#######################################################

#############
# Import data
#############
logfrac_params_guess=(0.8,-0.6,0.09,3)
yfrac_params_guess=(0.9,-3.5,0.035,3)
yfrac_logfrac_params_guess=(0.9,-1.9,0.2,3)
aticiguessparams=(1,0.1,0.02,4)

datarows = [] 
with open(csvfilename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    for row in csvreader: 
        datarows.append(row) 

xdata=np.array([int(row[0]) for row in datarows])
ydata=np.array([float(row[1]) for row in datarows])

##########################
#Basic stats from the data
##########################
denom_of_rsquare=sum(y**2 for y in ydata)-(1/len(ydata))*sum(y for y in ydata)**2


###########################
###########################   
#Curve fitting and plotting
###########################
###########################

###########
# Plot data
###########
plt.scatter(xdata,ydata,color='black',s=5,label='male data')

###############
###############
#Fits and plots
###############
###############

#######################
# Yang ctn fit and plot
#######################
ctn_popt, ctn_pcov = so.curve_fit(ctngompertz,xdata,ydata)
plt.plot(xdata,[ctngompertz(x,*ctn_popt) for x in xdata],label="ctn",color="cyan")
print('ctn parameters found:',[format(param,'.2E') for param in ctn_popt])

####################
# First fit and plot
####################
logfrac_popt, logfrac_pcov = so.curve_fit(gompertz_logfrac_fit, xdata, ydata,p0=logfrac_params_guess, maxfev=10000, bounds=((0,-np.inf,-np.inf,-np.inf),(1.5,np.inf,np.inf,np.inf))) 
plt.plot(xdata,[gompertz_logfrac(x,*logfrac_popt) for x in xdata],label='(12)',color='blue')
print("gompertz_logfrac PARAMETERS FOUND:",[format(param,'.2E') for param in logfrac_popt])

#####################
# Second fit and plot
#####################
yfrac_popt, yfrac_pcov = so.curve_fit(gompertz_yfrac_fit, xdata, ydata,p0=yfrac_params_guess) 
plt.plot(xdata,[gompertz_yfrac(x,*yfrac_popt) for x in xdata],label='(13)',color='yellow')
print("gompertz_yfrac PARAMETERS FOUND:",[format(param,'.2E') for param in yfrac_popt])

####################
# Third fit and plot
####################
yfrac_logfrac_popt, yfrac_logfrac_pcov = so.curve_fit(gompertz_yfrac_logfrac_fit, xdata, ydata,p0=yfrac_logfrac_params_guess, bounds=((0,-np.inf,-np.inf,-np.inf),(1.5,np.inf,np.inf,np.inf))) 
plt.plot(xdata,[gompertz_yfrac_logfrac(x,*yfrac_logfrac_popt) for x in xdata],label='(14)',color='green')
print("gompertz_yfrac_logfrac PARAMETERS FOUND:",[format(param,'.2E') for param in yfrac_logfrac_popt])

#########################
# Atici fit and plot
#########################
atici_popt, atici_pcov = so.curve_fit(aticifitfunc_vec_self, xdata, ydata,p0=aticiguessparams)
#atici_popt, atici_pcov = so.curve_fit(aticifitfunc_vec_self, xdata, ydata,p0=(aticiguessparams), bounds=((-np.inf,-np.inf,0,-np.inf),(np.inf,np.inf,1.5,np.inf)))
plt.plot(xdata,[aticigompertz(x,*atici_popt) for x in xdata],label="(3)",color='red')
print("atici parameters found:",[format(param,'.2E') for param in atici_popt])

######################
######################
#Goodness of fit tests
######################
######################

##########################
# RSS, SE, R^2 for logfrac
##########################
logfrac_RSS=sum([(ydata[int(k)-1]-gompertz_logfrac(k,*logfrac_popt))**2 for k in xdata])
print("logfrac RSS=", format(logfrac_RSS,'.2E'))
logfrac_standerr=(logfrac_RSS/(len(ydata)-4))**(0.5)
print("logfrac_standerr=",format(logfrac_standerr,'.2E'))
logfrac_rsquare=1-logfrac_RSS/denom_of_rsquare
print("logfrac_rsquare=",format(logfrac_rsquare,'.2E'))
logfrac_adjrsquare=1-((len(ydata)-1)/(len(ydata)-4))*(1-logfrac_rsquare)
print("logfrac_adjrsquare=",format(logfrac_adjrsquare,'.2E'))

########################
# RSS, SE, R^2 for yfrac
########################
yfrac_RSS=sum([(ydata[int(k)-1]-gompertz_yfrac(k,*yfrac_popt))**2 for k in xdata])
print("yfrac RSS=", format(yfrac_RSS,'.2E'))
yfrac_standerr=(yfrac_RSS/(len(ydata)-4))**(0.5)
print("yfrac_standerr=",format(yfrac_standerr,'.2E'))
yfrac_rsquare=1-yfrac_RSS/denom_of_rsquare
print("yfrac_rsquare=",format(yfrac_rsquare,'.2E'))
yfrac_adjrsquare=1-((len(ydata)-1)/(len(ydata)-4))*(1-yfrac_rsquare)
print("yfrac_adjrsquare=",format(yfrac_adjrsquare,'.2E'))

################################
# RSS, SE, R^2 for yfrac_logfrac
################################
yfrac_logfrac_RSS=sum([(ydata[int(k)-1]-gompertz_yfrac_logfrac(k,*yfrac_logfrac_popt))**2 for k in xdata])
print("yfrac_logfrac RSS=" + format(yfrac_logfrac_RSS,'.2E'))
yfrac_logfrac_standerr=(yfrac_logfrac_RSS/(len(ydata)-4))**(0.5)
print("yfrac_logfrac_standerr=",format(yfrac_logfrac_standerr,'.2E'))
yfrac_logfrac_rsquare=1-yfrac_logfrac_RSS/denom_of_rsquare
print("yfrac_logfrac_rsquare=",format(yfrac_logfrac_rsquare,'.2E'))
yfrac_logfrac_adjrsquare=1-((len(ydata)-1)/(len(ydata)-4))*(1-yfrac_logfrac_rsquare)
print("yfrac_logfrac_adjrsquare=",format(yfrac_logfrac_adjrsquare,'.2E'))



########################
# RSS, SE, R^2 for atici
########################
atici_RSS=sum([(ydata[int(k)-1]-aticigompertz(k,*atici_popt))**2 for k in xdata])
print("atici RSS=",format(atici_RSS,'.2E'))
atici_standerr=(atici_RSS/(len(ydata)-4))**(0.5)
print("atici_standerr=",format(atici_standerr,'.2E'))
atici_rsquare=1-atici_RSS/denom_of_rsquare
print("atici_rsquare=",format(atici_rsquare,'.2E'))
atici_adjrsquare=1-((len(ydata)-1)/(len(ydata)-4))*(1-atici_rsquare)
print("atici_adjrsquare=",format(atici_adjrsquare,'.2E'))

######################
# RSS, SE, R^2 for ctn
######################
ctn_RSS=sum([(ydata[int(k)-1]-ctngompertz(k,*ctn_popt))**2 for k in xdata])
print("ctn RSS=",format(ctn_RSS,'.2E'))
ctn_standerr=(ctn_RSS/(len(ydata)-3))**(0.5)
print("ctn_standerr=",format(ctn_standerr,'.2E'))


################################################################
###############################################################
##############################################################
###############################################################
################################################################
csvfilename="P.elegansFemaleTL.csv"
#csvfilename="P.elegansFemaleSVL.csv"

#############
# Import data
#############
datarows = [] 
with open(csvfilename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    for row in csvreader: 
        datarows.append(row) 

xdata=np.array([int(row[0]) for row in datarows])
ydata=np.array([float(row[1]) for row in datarows])

##########################
#Basic stats from the data
##########################
denom_of_rsquare=sum(y**2 for y in ydata)-(1/len(ydata))*sum(y for y in ydata)**2


###########################
###########################   
#Curve fitting and plotting
###########################
###########################

###########
# Plot data
###########
plt.scatter(xdata,ydata,color='grey',s=5,label='female data')

##################
# Ctn fit and plot
##################
ctn_popt, ctn_pcov = so.curve_fit(ctngompertz,xdata,ydata)
plt.plot(xdata,[ctngompertz(x,*ctn_popt) for x in xdata],'--',color="cyan")
print('ctn parameters found:',[format(param,'.2E') for param in ctn_popt])

####################
# First fit and plot
####################
logfrac_popt, logfrac_pcov = so.curve_fit(gompertz_logfrac_fit, xdata, ydata,p0=logfrac_params_guess, maxfev=10000, bounds=((0,-np.inf,-np.inf,-np.inf),(1.5,np.inf,np.inf,np.inf))) 
plt.plot(xdata,[gompertz_logfrac(x,*logfrac_popt) for x in xdata],'--',color='blue')
print("gompertz_logfrac PARAMETERS FOUND:",[format(param,'.2E') for param in logfrac_popt])

#####################
# Second fit and plot
#####################
yfrac_popt, yfrac_pcov = so.curve_fit(gompertz_yfrac_fit, xdata, ydata,p0=yfrac_params_guess) 
plt.plot(xdata,[gompertz_yfrac(x,*yfrac_popt) for x in xdata],'--',color='yellow')
print("gompertz_yfrac PARAMETERS FOUND:",[format(param,'.2E') for param in yfrac_popt])

####################
# Third fit and plot
####################
yfrac_logfrac_popt, yfrac_logfrac_pcov = so.curve_fit(gompertz_yfrac_logfrac_fit, xdata, ydata,p0=yfrac_logfrac_params_guess, bounds=((0,-np.inf,-np.inf,-np.inf),(1.5,np.inf,np.inf,np.inf))) 
plt.plot(xdata,[gompertz_yfrac_logfrac(x,*yfrac_logfrac_popt) for x in xdata],'--',color='green')
print("gompertz_yfrac_logfrac PARAMETERS FOUND:",[format(param,'.2E') for param in yfrac_logfrac_popt])

#########################
# Atici fit and plot
#########################
atici_popt, atici_pcov = so.curve_fit(aticifitfunc_vec_self, xdata, ydata,p0=aticiguessparams)
#atici_popt, atici_pcov = so.curve_fit(aticifitfunc_vec_self, xdata, ydata,p0=(aticiguessparams), bounds=((-np.inf,-np.inf,0,-np.inf),(np.inf,np.inf,1.5,np.inf)))
plt.plot(xdata,[aticigompertz(x,*atici_popt) for x in xdata],'--',color='red')
print("atici parameters found:",[format(param,'.2E') for param in atici_popt])

######################
######################
#Goodness of fit tests
######################
######################

##########################
# RSS, SE, R^2 for logfrac
##########################
logfrac_RSS=sum([(ydata[int(k)-1]-gompertz_logfrac(k,*logfrac_popt))**2 for k in xdata])
print("logfrac RSS=", format(logfrac_RSS,'.2E'))
logfrac_standerr=(logfrac_RSS/(len(ydata)-4))**(0.5)
print("logfrac_standerr=",format(logfrac_standerr,'.2E'))
logfrac_rsquare=1-logfrac_RSS/denom_of_rsquare
print("logfrac_rsquare=",format(logfrac_rsquare,'.2E'))
logfrac_adjrsquare=1-((len(ydata)-1)/(len(ydata)-4))*(1-logfrac_rsquare)
print("logfrac_adjrsquare=",format(logfrac_adjrsquare,'.2E'))

########################
# RSS, SE, R^2 for yfrac
########################
yfrac_RSS=sum([(ydata[int(k)-1]-gompertz_yfrac(k,*yfrac_popt))**2 for k in xdata])
print("yfrac RSS=", format(yfrac_RSS,'.2E'))
yfrac_standerr=(yfrac_RSS/(len(ydata)-4))**(0.5)
print("yfrac_standerr=",format(yfrac_standerr,'.2E'))
yfrac_rsquare=1-yfrac_RSS/denom_of_rsquare
print("yfrac_rsquare=",format(yfrac_rsquare,'.2E'))
yfrac_adjrsquare=1-((len(ydata)-1)/(len(ydata)-4))*(1-yfrac_rsquare)
print("yfrac_adjrsquare=",format(yfrac_adjrsquare,'.2E'))

################################
# RSS, SE, R^2 for yfrac_logfrac
################################
yfrac_logfrac_RSS=sum([(ydata[int(k)-1]-gompertz_yfrac_logfrac(k,*yfrac_logfrac_popt))**2 for k in xdata])
print("yfrac_logfrac RSS=" + format(yfrac_logfrac_RSS,'.2E'))
yfrac_logfrac_standerr=(yfrac_logfrac_RSS/(len(ydata)-4))**(0.5)
print("yfrac_logfrac_standerr=",format(yfrac_logfrac_standerr,'.2E'))
yfrac_logfrac_rsquare=1-yfrac_logfrac_RSS/denom_of_rsquare
print("yfrac_logfrac_rsquare=",format(yfrac_logfrac_rsquare,'.2E'))
yfrac_logfrac_adjrsquare=1-((len(ydata)-1)/(len(ydata)-4))*(1-yfrac_logfrac_rsquare)
print("yfrac_logfrac_adjrsquare=",format(yfrac_logfrac_adjrsquare,'.2E'))



########################
# RSS, SE, R^2 for atici
########################
atici_RSS=sum([(ydata[int(k)-1]-aticigompertz(k,*atici_popt))**2 for k in xdata])
print("atici RSS=",format(atici_RSS,'.2E'))
atici_standerr=(atici_RSS/(len(ydata)-4))**(0.5)
print("atici_standerr=",format(atici_standerr,'.2E'))
atici_rsquare=1-atici_RSS/denom_of_rsquare
print("atici_rsquare=",format(atici_rsquare,'.2E'))
atici_adjrsquare=1-((len(ydata)-1)/(len(ydata)-4))*(1-atici_rsquare)
print("atici_adjrsquare=",format(atici_adjrsquare,'.2E'))

######################
# RSS, SE, R^2 for ctn
######################
ctn_RSS=sum([(ydata[int(k)-1]-ctngompertz(k,*ctn_popt))**2 for k in xdata])
print("ctn RSS=",format(ctn_RSS,'.2E'))
ctn_standerr=(ctn_RSS/(len(ydata)-3))**(0.5)
print("ctn_standerr=",format(ctn_standerr,'.2E'))

########
# Legend
########
plt.xlabel(csv_xlabel)
plt.ylabel(csv_ylabel)
plt.legend()
plt.savefig(graphsavename,format='eps')