# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 18:49:40 2022

@author: Theodor Munteanu
"""
"""
INVENTORY of functions/methods:
    VaR_stocks
    
    ES_stocks
    
    ES_stocks_vanilla_options
    
    VaR_stocks_vanilla_options
    
    VaR_stocks_baskets
    
    VaR_basket_vanilla
"""

import sys
sys.path.append(r'C:\Users\XYZW\Documents\Python Scripts\Equity exotics')
sys.path.append(r'C:\Users\XYZW\Documents\Python Scripts\Quantlib_tests\Helpers')
import european_option_price as EOP
import basket_option_class as BOSpec 
import price_BS as opt_BS
import numpy as np
import scipy.stats as stats
#import multivariate_gbms as mgbms
import multivariate_basket as MB
#%%
def VaR_stocks(S,n,sigs,corr,exp_rets = None,alpha = 0.95,h = 10/252,details = 'no'):
    """
    Parameters
    ----------
    S : array/list
        Represents stock prices.
    n : no. of stocks
        list/array
    sigs : list/array
        Array of volatilities
    corr : Estimated correlation of stock returns. TYPE
        DESCRIPTION.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.95.
    h : TYPE, optional
        DESCRIPTION. The default is 10/252

    Returns
    -------
    Value at Risk of a stock portfolio. Float TYPE

    """
    def std_dev_port(exposures,cov_mat):
        return np.sqrt(np.dot(np.dot(exposures,cov_mat),exposures.T))
    exposures = np.array(S)*np.array(n)
    cov_mat = np.array(sigs,ndmin = 2).T*np.array(sigs,ndmin = 2)*corr
    #print(cov_mat)
    std_dev = std_dev_port(exposures,cov_mat)
    if exp_rets==None:
        exp_rets = [0]*len(S)
    if details == 'no':
        VaR = stats.norm.isf(1-alpha)*np.sqrt(h)*std_dev-np.dot(exposures,(1+np.array(exp_rets))**h-1)
        return VaR
    else:
        VaR = stats.norm.isf(1-alpha)*np.sqrt(h)*std_dev-np.dot(exposures,(1+np.array(exp_rets))**h-1)
        RC = exposures*np.dot(cov_mat,exposures.T)/std_dev * stats.norm.isf(1-alpha)*np.sqrt(h)
        if sum(exposures)!=0:
            weights = exposures/sum(exposures)
            return {'VaR':VaR,'Risk Contribs':RC,'Exposures':exposures,
                'Standard deviation':std_dev,
                'alpha':alpha,'horizon': h,'Expected return':np.dot(weights,exp_rets)}
        else:
            return {'VaR':VaR,'Risk Contribs':RC,'Exposures':exposures,
                'Standard deviation':std_dev,
                'alpha':alpha,'horizon': h}


#%%

def ES_stocks(S,n,sigs,corr,alpha = 0.975,h = 10/252,RC = False):
    """
    Parameters
    ----------
    S : array/list
        Represents stock prices.
    n : no. of stocks
        list/array
    sigs : list/array
        Array of volatilities
    corr : Estimated correlation of stock returns. TYPE
        DESCRIPTION.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.975.
    h : TYPE, optional
        DESCRIPTION. The default is 10/252.

    Returns
    -------
    Expected Shortfall of a stock portfolio. Float TYPE

    """
    def std_dev_port(exposures,cov_mat):
        return np.sqrt(np.dot(np.dot(exposures,cov_mat),exposures.T))
    exposures = np.array(S)*np.array(n)
    cov_mat = np.array(sigs,ndmin = 2).T*np.array(sigs,ndmin = 2)*corr
    std_dev = std_dev_port(exposures,cov_mat)
    if RC==False:
        return stats.norm.pdf(stats.norm.isf(1-alpha))/(1-alpha)*np.sqrt(h)*std_dev
    else:
        ES = stats.norm.pdf(stats.norm.isf(1-alpha))/(1-alpha)*np.sqrt(h)*std_dev
        return {'Exp Shortfall':ES,'Risk Contribs':np.dot(exposures,np.dot(cov_mat,exposures.T))/std_dev * ES/std_dev}
    
#%%
def VaR_options(S,stds,rf,no_options,option_types,underlyings,Ks,Ts):
    """
    S: share prices / underlyings asset values
    
    
    Ks: strike prices of options (list)
    """
    pass


def ES_stocks_gaps_options(S,sigs,qs,corr,Ks1,Ks2,expiries,underlyings,option_types,
                    no_stocks,no_options,rf,h = 10/252,alpha = 0.95,details = 'no'):
    """
    Parameters:
        S: share prices
        
        sigs: underlying volatilities
        
        qs: dividend rates
        
        corr: correlation matrix
        
        Ks1: 
    Returns:
        
        
    """
    values = np.array([opt_BS.gap_option(S[underlyings[i]-1], Ks1[i], Ks2[i], 
                        rf, sigs[underlyings[i]-1], expiries[i],qs[underlyings[i]-1],
                        option_types[i]) for i in range(len(expiries))],ndmin = 2)
    delta_BS = values[:,1]
    
    indexes = [[i for (i,x) in enumerate(underlyings) if x==item] for item in set(underlyings)]
    
    exposures_options = [sum(np.array(delta_BS)[indexes[i]]*np.array(no_options)[indexes[i]]) 
                        for i in range(len(indexes))]
    
    total_exposures = np.array(no_stocks) + exposures_options
    ES_portfolio  = ES_stocks(S,total_exposures,sigs,corr,alpha,h)
    return ES_portfolio
    


#%%

def risk_vanilla_options(S,sigs,qs,corr,Ks,expiries,underlyings,option_types,
                    no_stocks,no_options,rf,h = 10/252,alpha = 0.95):
    """
    Parameters:
        S, sigs,qs: asset values and annual vols, cost of opportunities/dividends
        
        corr: correlation matrix of returns
        
        Ks,expiries: strike prices and option lifetimes (lists of floats )
        
        
    Returns:
        Expected shortfall of portfolio
        
        Value at Risk of portfolio
        
        Value of the portfolio
    """
    price_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, sigs[underlyings[i]-1], 
                expiries[i],q = qs[underlyings[i]-1],option = option_types[i]) for i in range(len(expiries))]
    value_portfolio = np.dot(price_BS,no_options)
    deltas_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, 
            sigs[underlyings[i]-1], expiries[i],q = qs[underlyings[i]-1],option = option_types[i],
            greeks = 'yes')[1] for i in range(len(expiries))]
    indexes = [[i for (i,x) in enumerate(underlyings) if x==item] for item in set(underlyings)]
    
    exposures_options = [sum(np.array(deltas_BS)[indexes[i]] * np.array(no_options)[indexes[i]]) 
                        for i in range(len(indexes))]
    
    total_exposures = np.array(no_stocks) + exposures_options
    ES_portfolio  = ES_stocks(S,total_exposures,sigs,corr,alpha = alpha,h = h)
    VaR_portfolio = VaR_stocks(S,total_exposures,sigs,corr,alpha = alpha,h = h)
    return ES_portfolio, VaR_portfolio,value_portfolio



def ES_stocks_vanilla_options(S,sigs,qs,corr,Ks,expiries,underlyings,option_types,
                    no_stocks,no_options,rf,h = 10/252,alpha = 0.95,details = 'no'):
    """
    Parameters:
        underlyings: list of numbers.
        
    
    Example: 
        underlyings = [1,1,2,3,4,1] means options on the 1st asset (twice)
    second asset, third asset,fourth asset and first asset again. 
    """
    
    price_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, 
            sigs[underlyings[i]-1], expiries[i],q = qs[underlyings[i]-1],option = option_types[i])
                for i in range(len(expiries))]
    
    delta_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, 
            sigs[underlyings[i]-1], expiries[i],q = qs[underlyings[i]-1],option = option_types[i],
            greeks = 'yes')[1] for i in range(len(expiries))]
    
    indexes = [[i for (i,x) in enumerate(underlyings) if x==item] for item in set(underlyings)]
    
    exposures_options = [sum(np.array(delta_BS)[indexes[i]] * np.array(no_options)[indexes[i]]) 
                        for i in range(len(indexes))]
    
    total_exposures = np.array(no_stocks) + exposures_options
    
    ES_shares  = ES_stocks(S,no_stocks,sigs,corr,alpha,h)
    ES_portfolio  = ES_stocks(S,total_exposures,sigs,corr,alpha,h)
    cost_of_hedging = np.dot(price_BS,no_options)
    if details in ['No','no']:
        return ES_portfolio,cost_of_hedging
    else:
        return {'ES + options':ES_portfolio,
                'ES - options': ES_shares,
                'Cost of hedging':cost_of_hedging}

def RC_VaR_ES(S,n,sigs,corr,alpha = 0.975,h = 10/252,measure = 'VaR'):
    #ES = ES_stocks(S, n, sigs, corr,alpha = alpha,h = h)
    weights = np.array(S)*np.array(n)/np.dot(S,n)
    cov_mat = np.array(sigs,ndmin = 2).T*np.array(sigs,ndmin = 2)*corr
    std_dev = np.sqrt(np.dot(weights,np.dot(cov_mat,weights.T)))
    risk_contribs_std = weights*np.dot(cov_mat,weights.T)/std_dev
    if measure=='VaR':
        return {'Risk Contribs':stats.norm.isf(1-alpha)*np.sqrt(h)*risk_contribs_std,
                'VaR prc':stats.norm.isf(1-alpha)*np.sqrt(h)*std_dev,
                'Portfolio Value':np.dot(S,n),
                'Money VaR': stats.norm.isf(1-alpha)*np.sqrt(h)*std_dev*np.dot(S,n)}
    else:
        return {'Risk Contribs':stats.norm.pdf(stats.norm.isf(1-alpha))/(1-alpha)*np.sqrt(h)*risk_contribs_std,
                'ES prc':stats.norm.pdf(stats.norm.isf(1-alpha))/(1-alpha)*np.sqrt(h)*std_dev,
                'Portfolio Value':np.dot(S,n),
                'Money ES': stats.norm.pdf(stats.norm.isf(1-alpha))/(1-alpha)*np.sqrt(h)*std_dev*np.dot(S,n)}
    
def ES_stocks_vanilla_options2(S,sigs,qs,corr,Ks,expiries,underlyings,option_types,
                    no_stocks,no_options,rf,h = 10/252,alpha = 0.95,details = 'no'):
    """
    underlyings: list of numbers.
    
    Example: underlyings = [1,1,2,3,4,1] means options on the 1st asset (twice)
    second asset, third asset,fourth asset and first asset again. 
    """
    
    
    price_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, 
            sigs[underlyings[i]-1], expiries[i],q = qs[underlyings[i]-1],
            option = option_types[i]) for i in range(len(expiries))]
    
    delta_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, 
            sigs[underlyings[i]-1], expiries[i],q = qs[underlyings[i]-1],
            option = option_types[i],greeks = 'yes')[1] for i in range(len(expiries))]
    
    
    indexes2 = [[i for (i,x) in enumerate(underlyings) if x==item] 
                                    for item in range(1,len(S)+1)]
    
    exposures_options2 = [sum(np.array(delta_BS)[indexes2[i]]*
                            np.array(no_options)[indexes2[i]]) 
                             if len(indexes2[i])>0 else 0 for i in range(len(indexes2))]
    
    total_exposures = np.array(no_stocks) + exposures_options2
    ES_portfolio  = ES_stocks(S,total_exposures,sigs,corr,alpha,h)
    cost_of_hedging = np.dot(price_BS,no_options)
    if details in ['No','no']:
        return ES_portfolio,cost_of_hedging
    elif details in ['risk','risk details']:
        
        return {'ES':ES_portfolio,'Cost of hedging':cost_of_hedging,
                'Total Exposures':total_exposures}
    elif str.lower(details) in ['all','all details']:
        cost_of_hedging = np.dot(price_BS,no_options)
        VaR = VaR_stocks(S,total_exposures,sigs,corr,alpha,h)
        VaR_wo = VaR_stocks(S,no_stocks,sigs,corr,alpha,h)
        ES_wo = ES_stocks(S,no_stocks,sigs,corr,alpha,h)
        RC = RC_VaR_ES(S, total_exposures, sigs, corr,alpha,h,measure = 'ES')['Risk Contribs']
        portf_value = np.dot(price_BS,no_options)+np.dot(no_stocks,S)
        return {'ES {0} % + options'.format(alpha*100):ES_portfolio,
                'ES {0} % - options'.format(alpha*100):ES_wo,
                'Cost of hedging':cost_of_hedging,
                'Value at Risk {0} % + options'.format(alpha*100):VaR,
                'Value at Risk {0} % wo options'.format(alpha*100):VaR_wo,
                'Prices':np.array(price_BS),
                'Deltas':np.array(delta_BS),
                'Portfolio Value': portf_value,
                'Risk Contributions':RC}

#%%
def std_dev_portf(S,sigs,qs,corr,Ks,expiries,underlyings,option_types,no_stocks,no_options,
                  rf,rets = None,details = 'no'):
    def std_dev_port(exposures,cov_mat):
        """
        standard deviation of exposures given covariance matrix of returns.
        """
        return np.sqrt(np.dot(np.dot(exposures,cov_mat),exposures.T))
    
    cov_mat = np.array(sigs).T*np.array(sigs)*corr
    
    price_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, 
            sigs[underlyings[i]-1], expiries[i],q = qs[underlyings[i]-1],option = option_types[i])
                for i in range(len(expiries))]
    
    delta_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, 
            sigs[underlyings[i]-1], expiries[i],q = qs[underlyings[i]-1],option = option_types[i],
            greeks = 'yes')[1] for i in range(len(expiries))]
    
    indexes = [[i for (i,x) in enumerate(underlyings) if x==item] for item in set(underlyings)]
    
    exposures_options = [sum(np.array(delta_BS)[indexes[i]]*\
                        np.array(no_options)[indexes[i]]) for i in range(len(indexes))]
    
    total_exposures = (np.array(no_stocks) + exposures_options)*S
    
    init_portf_value = np.dot(price_BS,no_options)+np.dot(no_stocks,S)
    
    std_dev = std_dev_port(total_exposures,cov_mat)/init_portf_value
    if details=='no':
        return std_dev
    else:
        portf_opt_value = np.dot(no_options,price_BS)
        if rets==None:
            rets = [0]*len(S)
        return {'Std dev of PnL': std_dev,'Option prices': price_BS,'Option deltas':delta_BS,
                'Factor Exposures':total_exposures,
                'Option port value':portf_opt_value,
                'Return':np.dot(total_exposures,rets)/init_portf_value,
                'Init portf value':init_portf_value}


    
def VaR_stocks_vanilla_options(S,sigs,qs,corr,Ks,expiries,underlyings,option_types,
                no_stocks,no_options,rf,exp_rets = None,h = 10/252,alpha = 0.95,details = 'no'):
    """
    VaR of a portfolio of equities and of vanilla options. 
    
    
    """
    price_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, 
            sigs[underlyings[i]-1], expiries[i],q = qs[underlyings[i]-1],option = option_types[i])
                for i in range(len(expiries))]
    
    delta_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, 
            sigs[underlyings[i]-1], expiries[i],q = qs[underlyings[i]-1],option = option_types[i],
            greeks = 'yes')[1] for i in range(len(expiries))]
    
    indexes = [[i for (i,x) in enumerate(underlyings) if x==item] for item in set(underlyings)]
    
    exposures_options = [sum(np.array(delta_BS)[indexes[i]]*\
                        np.array(no_options)[indexes[i]]) for i in range(len(indexes))]
    
    total_exposures = np.array(no_stocks) + exposures_options
    
    if exp_rets == None:
        exp_rets = [0]*len(S)
        
    VaR_portfolio  = VaR_stocks(S,total_exposures,sigs,corr,exp_rets,alpha,h)
    VaR_shares = VaR_stocks(S,no_stocks,sigs,corr,exp_rets,alpha,h)
    cost_of_hedging = np.dot(price_BS,no_options)

    if details in ['No','no']:
        return VaR_portfolio,cost_of_hedging
    else:
        portf_value = np.dot(price_BS,no_options)+np.dot(no_stocks,S)
        RC = RC_VaR_ES(S, total_exposures, sigs, corr,measure = 'VaR',alpha = alpha,h = h)['Risk Contribs']
        return {'VaR + options':VaR_portfolio,
                'Cost of hedging':cost_of_hedging,
                'VaR - options':VaR_shares,
                'Portfolio Value':portf_value,
                'Risk Contributions':RC,
                'prices':price_BS,
                'deltas':delta_BS}


def VaR_stocks_baskets(S,sigs,dvd,rf,corr,Ks,expiries,u_list,no_options,\
                      no_stocks,types,payoffs,coeffs = None,details = 'no'):
    """
    Parameters:
        S: share prices list
        
        sigs: vols list
        
        dvd: dividend list 
        
        u_list: underlying asset list 
        
        no_options: number of options. 
        
        payoffs: payoff_types.
        
    """
    S,sigs,dvd= np.array(S),np.array(sigs),np.array(dvd)
    basket_opts = [BOSpec.basketOptionSpec(S[u_list[i]],sigs[u_list[i]],\
                        corr[np.ix_(u_list[i],u_list[i])],Ks[i],expiries[i],\
                        qs = list(dvd[u_list[i]]),option_type = types[i],\
                        underlyings = u_list[i],payoff_type = payoffs[i]) \
                        for i in range(len(types))]
    
    delta_basket_options = np.zeros((len(types),len(S)))
    if coeffs == None:
        for i in range(len(types)):
            delta_basket_options[i,u_list[i]] = basket_opts[i].delta_basket(rf,
                                ns=10000)
    else:
        for i in range(len(types)):
            delta_basket_options[i,u_list[i]] = basket_opts[i].delta_basket(rf,\
                                ns = 10000,coeffs = coeffs) 
    exposures = [np.dot(no_options,delta_basket_options[:,i]) for i in range(len(S))]
    total_exposures = np.array(no_stocks) + np.array(exposures)
    Value_at_Risk= VaR_stocks(S,total_exposures,sigs,corr,alpha = 0.95,h = 10/252)
    if details in ['No','no']:
        return Value_at_Risk
    else:
        prices = [0]*len(types)
        if coeffs == None:
            for i in range(len(types)):
               prices[i] = basket_opts[i].value_basket(rf,ns = 10000)
        else:
            for i in range(len(types)):
               prices[i] = basket_opts[i].value_basket(rf,ns = 10000,coeffs = coeffs)
               
        cost_of_hedging = np.dot(prices,no_options)
        return {'VaR':Value_at_Risk,'Cost of hedging':cost_of_hedging,
                'Deltas baskets':delta_basket_options}

def VaR_basket_vanilla2(S, sigs, dvd, corr, rf, Ks_baskets, Ks_vanilla, expiries_vanilla,
                        expiries_baskets, no_basket_options,
                        no_vanilla_options,  no_stocks,
                        types_baskets,  types_vanilla,  u_list_basket,  u_list_vanilla,
                        payoffs = 'Average',  coeffs = None,alpha = 0.95,  h = 10/252,
                        details = 'No'):
    """
    
    """
    S,sigs,dvd = np.array(S),np.array(sigs),np.array(dvd)
    value_BS = np.array([opt_BS.option_price_BS(S[u_list_vanilla[i]-1],Ks_vanilla[i],
                    rf,sigs[u_list_vanilla[i]-1],expiries_vanilla[i],
            q = dvd[u_list_vanilla[i]-1],option = types_vanilla[i],greeks = 'yes')[0:2] 
            for i in range(len(expiries_vanilla))],ndmin = 2)
    delta_BS = value_BS[:,1]
    indexes = [[i for (i,x) in enumerate(u_list_vanilla) if x==item] 
                                    for item in set(u_list_vanilla)]
    
    exposures_vanillas = [sum(np.array(delta_BS)[indexes[i]]*\
                np.array(no_vanilla_options)[indexes[i]]) 
                          for i in range(len(indexes))]
        
    partial_exposures = np.array(no_stocks) + exposures_vanillas
    if payoffs == 'Average':
        payoffs = ['Average']*len(types_baskets)
    
    

def VaR_basket_vanilla(S, sigs, dvd, corr,  rf,  Ks_baskets,  Ks_vanilla,
            expiries_vanilla, expiries_baskets,  no_basket_options,  
            no_vanilla_options,  no_stocks,
            types_baskets,  types_vanilla,  u_list_basket,  u_list_vanilla,
            payoffs = 'Average',  coeffs = None,alpha = 0.95,  h = 10/252,
            details = 'No'):
    
    """
    Parameters:
        S: stock prices (it is setup to be a list)
        
        sigs: vector of sigs.
        
        dvd: vector of dividends
        
        rf: risk-free rate
        
        Ks: strike price of basket options
        
        Ks_vanilla: strike prices of vanilla options. 
        
        expiries_vanilla: vector (list) of expiries for vanilla options.
        
        Payoffs: payoffs for basket options. 
        
    Functionality: 
        We have a portfolio that can consist of: stocks, vanilla options,
            basket options
    """
    S,sigs,dvd= np.array(S),np.array(sigs),np.array(dvd)
    options_vanilla = [EOP.option(S[u_list_vanilla[i]-1],Ks_vanilla[i],\
                    sigs[u_list_vanilla[i]-1],expiries_vanilla[i],\
            dvd[u_list_vanilla[i]-1],types_vanilla[i]) \
            for i in range(len(expiries_vanilla))]
    
    delta_BS = [options_vanilla[i].delta_BS(rf) for i in range(len(options_vanilla))]
    indexes = [[i for (i,x) in enumerate(u_list_vanilla) if x==item] \
                                    for item in set(u_list_vanilla)]
    
    exposures_vanillas = [sum(np.array(delta_BS)[indexes[i]]*\
                np.array(no_vanilla_options)[indexes[i]]) for i in range(len(indexes))]
   
    partial_exposures = np.array(no_stocks) + exposures_vanillas
    
    if payoffs == 'Average':
        payoffs = ['Average']*len(types_baskets)
    
    basket_opts2 = [BOSpec.basketOptionSpec(S[u_list_basket[i]],sigs[u_list_basket[i]],
                    corr[np.ix_(u_list_basket[i],u_list_basket[i])],Ks_baskets[i],
                    expiries_baskets[i],qs = list(dvd[u_list_basket[i]]), 
                    option_type = types_baskets[i],
                        underlyings = u_list_basket[i],payoff_type = payoffs[i]) 
                        for i in range(len(types_baskets))]
    
    delta_basket_options2 = np.zeros((len(types_baskets),len(S)))
    
    if coeffs!=None:
        for i in range(len(types_baskets)):
            delta_basket_options2[i,u_list_basket[i]] = basket_opts2[i].delta_basket(
                rf,ns = 10000,coeffs = coeffs[i]) 
    else:
        for i in range(len(types_baskets)):
            delta_basket_options2[i,u_list_basket[i]] = basket_opts2[i].delta_basket(rf,ns = 10000)
    
    
    exposures_baskets = [np.dot(no_basket_options,delta_basket_options2[:,i]) 
                  for i in range(len(S))]
    
    total_exposures = exposures_baskets+partial_exposures
    
    Value_at_Risk = VaR_stocks(S,total_exposures,sigs,corr,alpha = 0.95,h = 10/252)
    
    if details in [ 'No' ,'no']:
        return Value_at_Risk
    else:
        """
        Return also the cost of hedging.
        """
        prices_opt = [options_vanilla[i].price_BS(rf) for i in range(len(options_vanilla))]
        if coeffs == None:
            price_baskets = [basket_opts2[i].value_basket(rf,
                        ns = 10000) for i in range(len(basket_opts2))]
        else:
            price_baskets = [basket_opts2[i].value_basket(rf,
                        ns = 10000,coeffs = coeffs[i]) 
                             for i in range(len(basket_opts2))]
        
        cost_of_hedging = np.dot(prices_opt,no_vanilla_options)+np.dot(price_baskets,no_basket_options)
        return {'VaR':Value_at_Risk, 'cost of Hedging':cost_of_hedging}
    
    
def PnLs_digital_options(S,underlyings,Ks,expiries,options,payoffs,no_options,trajs,stds,rebates = None,rf = 0.0):
    n = len(underlyings)
    ind1 = [i for i in range(n) if payoffs[i]=='CoN']
    ind2 = [i for i in range(n) if payoffs[i]=='AoN']
    
    if rebates!=None and len(rebates)!=len(ind1) and len(ind1)>0:
        raise TypeError('Number of rebates must be equal to number of CoN options')
    else:
        pass
    
    if isinstance(Ks,list)==True:
        Ks = np.array(Ks)
    if isinstance(expiries,list)== True:
        expiries = np.array(expiries)
    if isinstance(options,list) == True:
        options = np.array(options)
    if isinstance(payoffs,list)==True:
        payoffs = np.array(payoffs)
    if isinstance(no_options,list)==True:
        no_options =  np.array(no_options)
    
    if len(ind1)>0:
        digital_portf1 = opt_BS.CoN_portfolio(underlyings[ind1], Ks[ind1], expiries[ind1], options[ind1], 
                                                   payoffs[ind1], no_options[ind1],rebates = rebates)
        deltas_portf1 = digital_portf1.delta_portfolio(S,stds,rf = rf)
        vals01 = trajs[underlyings[ind1]-1,:,1].T
        PnLs1 = np.dot(vals01-S[underlyings[ind1]-1],deltas_portf1)
    else:
        pass
    
    if len(ind2)>0:
        digital_portf2 = opt_BS.AoN_portfolio(underlyings[ind2], Ks[ind2], expiries[ind2], options[ind2], 
                                                   payoffs[ind2], no_options[ind2])
        
        deltas_portf2 = digital_portf2.delta_portfolio(S,stds,rf = rf)
        vals02 = trajs[underlyings[ind2]-1,:,1].T
        PnLs2 = np.dot(vals02-S[underlyings[ind2]-1],deltas_portf2)
    else:
        pass
    
    if len(ind1)>0 and len(ind2)>0:
        return PnLs1+PnLs2
    elif len(ind1)==0 and len(ind2)>0:
        return PnLs2
    else:
        return PnLs1


def PnLs_vanilla_options1(S, underlyings, Ks, expiries, options, trajs, stds, no_options,rf = 0.0):
    """
    PnLs of vanilla options with already prepared trajectories. 
    
    The delta-approach is being used. 
    
    Parameters:
        S: underlyings share prices 
        
        trajs: 3d trajectories. 
        
    The first value of the trajectory is being used. 
    """
    portf_vanilla = opt_BS.option_portfolio(underlyings, Ks, expiries, options, no_options)
    vanilla_deltas = portf_vanilla.deltas_portfolio(S, stds, rf)
    vals0 = trajs[:,:,1].T
    PnLs = np.dot(vals0-S,vanilla_deltas)
    return PnLs 


#%%
def PnLs_vanilla_options2(S,sigs,qs,corr,Ks,expiries,underlyings,option_types,
                          no_stocks,no_options,rf,h = 10/252,ns = 10000):
    """
    Use analytic approach for vanilla options. 
    
    Expensive method: trajectories must be generated. 
    
    price_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, sigs[underlyings[i]-1], 
                expiries[i],q = qs[underlyings[i]-1],option = option_types[i]) 
                for i in range(len(expiries))]
    """
    n = len(S)
    #value_portfolio = np.dot(price_BS,no_options)
    deltas_BS = [opt_BS.option_price_BS(S[underlyings[i]-1], Ks[i], rf, 
            sigs[underlyings[i]-1], expiries[i],q = qs[underlyings[i]-1],option = option_types[i],
            greeks = 'yes')[1] for i in range(len(expiries))]
    indexes = [[i for (i,x) in enumerate(underlyings) if x==item] for item in set(underlyings)]
    exposures_options = [sum(np.array(deltas_BS)[indexes[i]] * np.array(no_options)[indexes[i]]) 
                        for i in range(len(indexes))]
    total_exposures = np.array(no_stocks) + exposures_options
    cov_mat = np.array(sigs,ndmin = 2).T*np.array(sigs,ndmin = 2)*corr
    ret_sims = stats.multivariate_normal.rvs(mean = [0]*n,cov = cov_mat,size = ns)
    PnLs = np.dot(ret_sims*np.sqrt(h),total_exposures*S)
    return PnLs


def PnLs_gap_portf(S,underlyings,stds,Ks,Ks2,expiries,options,no_options,trajs,rf =0.0,qs = None):
     """
     PnL of gap options with delta-approach and using as inputs:
        
        trajs: 3d numpy array with trajectories. 
        stds: standard deviations. Necessary for the computations of delta.
     
        Ks: the first level of strike price
     
        Ks2: the second level of strike price.
     """
     gap_portf = opt_BS.gap_portfolio(underlyings,Ks,Ks2,expiries,options,no_options)
     deltas_total = gap_portf.delta_portfolio(S,stds,rf = rf)
     vals0 = trajs[:,:,1].T
     pnls = np.dot(vals0-S,deltas_total)
     return pnls
 
def PnLs_MB_portf(S,underlyings,Ks,expiries,options,trajs,stds,rho,coeffs,
                  no_options = None,rf = 0.0, ret_type = 'total'):

    """ 
    USES A DELTA-GREEK APPROACH
    
    Parameters:
        underlyings: numpy 2D array. A multitude of underlyings are given
        
        trajs: trajectories (multivariate_geometric brownian motions) (3d numpy array)
        
        coeffs: list of lists. 
    
    RETURNS:
         PnLs of a portfolio of multivariate basket options + value of the portfolio
         If ret_type==detailed, gives P&Ls across each entity.         
    
    Explanation:
        vals_term: terminal values extracted from trajs. Needed for the pricing.
        
        vals0: price after the horizon of investment has passed. 
        

    """
    if no_options==None:
        no_options = [1]*len(Ks)
    MB_portf = MB.multivariate_basket_port(underlyings,Ks,expiries,options,no_options = no_options)
    vals0 = trajs[:,:,1].T
    vals_term = trajs[:,:,-1].T
    price = MB_portf.value_portf(S, stds, rho, vals_term, coeffs, rf = rf)
    deltas = MB_portf.delta_portf(S, stds, rho, vals_term, coeffs, rf = rf)
    if ret_type=='total':
        pnls_MB = np.dot(vals0-S,deltas)
        return pnls_MB,price,deltas
    else:
        pnls_MB = (vals0-S)*deltas
        return pnls_MB,price,deltas

def PnLs_MB_portf2(S,underlyings,Ks,expiries,options,trajs,coeffs,
                   no_options = None,rf = 0.0,ret_type = 'total'):
    """
    USES a DELTA-GREEK.
    """
    MB_portf = MB.multivariate_basket_port(underlyings,Ks,expiries,options,no_options = no_options)
    vals0 = trajs[:,:,1].T
    vals_term = trajs[:,:,-1].T
    #price = MB_portf.value_portf(S, stds, rho, vals_term, coeffs, rf = rf)
    deltas = MB_portf.delta_portf2(S, vals_term, coeffs, rf = rf)
    #print(deltas)
    if ret_type == 'total':
        pnls_MB = np.dot(vals0-S,deltas)
        return pnls_MB
    else:
        pnls_MB = (vals0-S)*deltas
        return pnls_MB

def PnLs_shares(S,no_shares,mus,sigs,rho,hor = 1/12,size = 10000,details = 'no'):
    if isinstance(sigs,list)==True:
       sigs = np.array(sigs)
    arr = [0]*len(S);cov_mat = np.eye(len(S))
    mvnorm_smpls = stats.multivariate_normal.rvs(mean = arr,cov = rho,size = size)
    share_vals = S*np.exp((mus-sigs**2/2)*hor+sigs*np.sqrt(hor)*mvnorm_smpls)
    if details == 'yes':
        pnls = (share_vals-S)*no_shares
    else:
        pnls = np.dot(share_vals-S,no_shares)
    return pnls

def risk_contribs(losses,alpha):
    """
    Losses: np.ndarray with columns for each entity. 
    
    Returns:
        risk contributions for Value at Risk
    """
    total_losses = np.sum(losses,axis = 1)
    value_at_risk = np.quantile(total_losses,alpha)
    ind = np.where(total_losses == max(total_losses[total_losses<value_at_risk]))
    risk_contribs = losses[ind[0][0],:]
    return risk_contribs

def risk_contribs_ES(losses,alpha):
    total_losses = np.sum(losses,axis = 1)
    value_at_risk = np.quantile(total_losses,alpha)
    pos = np.where(total_losses>value_at_risk)[0]
    RC_ES = np.mean(losses[pos,:],axis = 0)
    return RC_ES

"""
def ES_stocks_gaps_vanillas(S,sigs,qs,corr,Ks1,Ks2,expiries_gaps,underlyings,
        option_types,no_stocks,no_gaps,Ks_vanilla,expiries_vanillas,
        underlyings_vanillas,no_vanillas,rf,h = 10/252,alpha = 0.95):

    Functionality:
        Stocks + Gap options + vanillas. 
    
    Parameters:
        
        S: vector of underlying prices
    
    
    values_gaps = np.array([opt_BS.gap_option(S[underlyings[i]-1], Ks1[i], Ks2[i], 
                        rf, sigs[underlyings[i]-1], expiries_gaps[i],
                        qs[underlyings[i]-1],
            option_types[i]) for i in range(len(expiries_gaps))],ndmin = 2)
    delta_BS_gaps = values_gaps[:,1]
    indexes = [[i for (i,x) in enumerate(underlyings) if x==item] 
                                    for item in set(underlyings)]
    pass
"""