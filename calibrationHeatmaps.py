import QuantLib as ql
import matplotlib.pyplot as plt


import numpy as np
from scipy.stats import norm
import datetime
from py_vollib.black_scholes.implied_volatility import implied_volatility
import seaborn as sns




import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

from rbergomi.rbergomi import rBergomi

def heston_pricer(lambd, vbar, eta, rho, v0, r, q, tau, S0, K):
    """Computes European Call price under Heston dynamics with closedform solution.

    """
    today = datetime.date.today()
    ql_date = ql.Date(today.day, today.month, today.year)
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = ql_date
    
    # option data
    option_type = ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, K)
    maturity_date = ql_date + int(round(tau * 365))
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)
    
    # Heston process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, r, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, q, day_count))
    heston_process = ql.HestonProcess(flat_ts, dividend_yield, spot_handle, v0, lambd, vbar, eta, rho)
    
    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process), 1e-15, int(1e6))
    european_option.setPricingEngine(engine)
    
    # check numerical stability
    try:
        price = european_option.NPV()
        if price <= 0 or price + K < S0:
            iv = np.nan
            logging.debug("NumStabProblem: Price {}. Intrinsic {}. Time {}. Strike {}.".format(price, S0-K, tau, K))
        else:
            logging.debug("Success: Price {} > intrinsic {}".format(price, S0-K))
            iv = implied_volatility(price, S0, K, tau, r, 'c')
    except RuntimeError:
        logging.info("RuntimeError: Intrinsic {}. Time {}. Strike {}.".format(S0-K, tau, K))
        price = np.nan
        iv = np.nan
    return price

def rBergomi_pricer(H, eta, rho, v0, tau, K, S0, MC_samples=40000):
    
    try:
        rB = rBergomi(n=365, N=MC_samples, T=tau, a=H-0.5)
        dW1, dW2 = rB.dW1(), rB.dW2()
        Y = rB.Y(dW1)
        dB = rB.dB(dW1, dW2, rho)
        xi = v0
        V = rB.V(Y, xi, eta)
        S = rB.S(V, dB)
        ST = S[:, -1]
        price = np.mean(np.maximum(ST-K, 0))
    except:
        return np.nan, np.nan
    
    # check numerical stability
    if price <= 0 or price + K < S0:
        iv = np.nan
        logging.debug("NumStabProblem: Price {}. Intrinsic {}. Time {}. Strike {}.".format(price, S0-K, tau, K))
    else:
        logging.debug("Success: Price {} > intrinsic {}".format(price, S0-K))
        iv = implied_volatility(price, S0, K, tau, 0, 'c')
    return price
def BS_CALL(S, K, T, r, sigma):
    N = norm.cdf
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)
def merton_jump_call(S, K, T, r, sigma, m , v, lam):
    p = 0
    for k in range(40):
        r_k = r - lam*(m-1) + (k*np.log(m) ) / T
        sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
        k_fact = np.math.factorial(k)
        p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact))  * BS_CALL(S, K, T, r_k, sigma_k)  
    return p

S0=1.0
# K=1.0
# strike=K
r=0
q=0
#T=30/365
#Tf = 45/365
N=30
M=50000

strikes = [0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
maturities = [20/365,21/365,22/365,23/365,24/365,25/365,26/365,27/365,28/365,29/365,30/365]
maturitiesDays = [20,21,22,23,24,25,26,27,28,29,30]

#rBergomi parameters
aBerg = -0.4        #H-0.5 (Hurst param)
HBerg = 0.1
xiBerg = 0.235**2   
etaBerg = 1.9
rhoBerg = -0.9 

#Heston parameters
lamHest = 3.9810393221890084
vHest = 0.09327160815810764
etaHest = 1.704203041025321 
rhoHest = -0.716492483305834
v0Hest = 0.05126161966697793 

#merton parameters 
sigmaMert = 0.198
vMert = 0.048924488315176824
mMert = 0.0
lamMert = 2.0829758475877087
cMert = np.exp(mMert+vMert**2*0.5)

berg_hest = np.zeros((len(maturities),len(strikes)))
berg_mert = np.zeros((len(maturities),len(strikes)))
hest_mert = np.zeros((len(maturities),len(strikes)))

for i, T in enumerate(maturities):
    for j, K in enumerate(strikes):
        priceBerg = rBergomi_pricer(HBerg, etaBerg, rhoBerg, xiBerg, T, K, S0)
        priceHest = heston_pricer(lamHest, vHest, etaHest, rhoHest, v0Hest, r, q, T, S0, K)
        priceMert = merton_jump_call(S0, K, T, r, sigmaMert, cMert , vMert, lamMert)
        berg_hest[i,j] = 100*np.abs((priceBerg-priceHest)/priceBerg)
        berg_mert[i,j] = 100*np.abs((priceBerg-priceMert)/priceBerg)
        hest_mert[i,j] = 100*np.abs((priceHest-priceMert)/priceHest)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(19,5))
fig.subplots_adjust(wspace=0.15)

ax1=plt.subplot(1,3,1)
sns.heatmap(berg_hest, ax=ax1, xticklabels=strikes, yticklabels=maturitiesDays, cbar_kws={'format': '%.0f%%'})
plt.title('rBergomi Heston error', fontsize = 20) 
plt.xlabel('Strike', fontsize = 15) 
plt.ylabel('Maturity in Days', fontsize = 15) 

ax2=plt.subplot(1,3,2)
sns.heatmap(berg_mert, ax=ax2, xticklabels=strikes, yticklabels=maturitiesDays, cbar_kws={'format': '%.0f%%'})
plt.title('rBergomi Merton error', fontsize = 20)
plt.xlabel('Strike', fontsize = 15) 
plt.ylabel('Maturity in Days', fontsize = 15) 
ax3=plt.subplot(1,3,3)

sns.heatmap(hest_mert, ax=ax3, xticklabels=strikes, yticklabels=maturitiesDays, cbar_kws={'format': '%.0f%%'})
plt.title('Heston Merton error', fontsize=20)
plt.xlabel('Strike', fontsize = 15) 
plt.ylabel('Maturity in Days', fontsize = 15) 




















