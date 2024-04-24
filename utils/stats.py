import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma
from scipy import ndimage as ndi
from lmfit import minimize, Parameters, Parameter, report_fit


def bootstrap(l,n_times,n_samples=None,confidence_interval=95,median=False,maximum=False,log=False, std=False):
    if n_samples == None:
        n_samples = len(l)
    if median:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,n_samples),n_samples)
            new_list=[l[idx] for idx in indices]
            new_means.append(ma.median(new_list,axis=0))
        new_means=ma.vstack(new_means)
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return ma.median(l,axis=0),cil,ciu
    elif maximum:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,n_samples),n_samples)
            new_list=[l[idx] for idx in indices]
            new_means.append(ma.max(new_list,axis=0))
        new_means=ma.vstack(new_means)
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return ma.max(l,axis=0),cil,ciu
    elif log:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,n_samples),n_samples)
            new_list=[np.log10(l[idx]) for idx in indices]
            new_means.append((ma.mean(new_list,axis=0)))
        new_means=10**(ma.vstack(new_means))
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return 10**(ma.mean(np.log10(np.array(l)),axis=0)),cil,ciu
    elif std:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,n_samples),n_samples)
            new_list=[l[idx] for idx in indices]
            new_means.append((ma.std(new_list,axis=0)))
        new_means=ma.vstack(new_means)
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return ma.std(l,axis=0),cil,ciu
    else:
        per=(100-confidence_interval)/2
        new_means=[]
        for i in range(n_times):
            indices=np.random.choice(range(0,n_samples),n_samples)
            new_list=[l[idx] for idx in indices]
            new_means.append(ma.mean(new_list,axis=0))
        new_means=ma.vstack(new_means)
        cil=np.zeros(new_means.shape[1])
        ciu=np.zeros(new_means.shape[1])
        for i in range(new_means.shape[1]):
            cil[i]=np.nanpercentile(new_means[:,i].filled(np.nan),per)
            ciu[i]=np.nanpercentile(new_means[:,i].filled(np.nan),100-per)
        cil = ma.masked_array(cil, np.isnan(cil))
        ciu = ma.masked_array(ciu, np.isnan(ciu))
        return ma.mean(l,axis=0),cil,ciu


def dist_bootstrap(statscollection_fish,t0,tf,num_states, n_times = 1000):

    """
    Do a bootstrapping over cumulative distributions and output complementary distribution
    with 95% confidence intervals
    """
    y_errorbar_metastate = []
    x_all_metastate = []

    for ms in range(num_states):
        allfish_stats = np.hstack(statscollection_fish[ms])
        x,y = cumulative_dist(allfish_stats,(t0,tf))
        y_all = 1-np.array([np.mean(y[x==x_unique]) for x_unique in np.unique(x)])
        x_all = np.sort(np.unique(x))
        x_all_metastate.append(x_all)

        dict_y = {}
        for x_ in x_all:
            dict_y[x_] = []

        for _ in range(n_times):
            x,y = cumulative_dist(np.hstack(np.random.choice(statscollection_fish[ms],len(statscollection_fish[ms]))),(t0,tf))
            y = 1-np.array([np.mean(y[x==x_unique]) for x_unique in np.unique(x)])
            x = np.sort(np.unique(x))
            for kx in range(len(y)):
                dict_y[x[kx]].append(y[kx])

        y_errorbars = np.zeros((len(dict_y.keys()),3))
        for kx,x_ in enumerate(x_all):
            values = np.array(dict_y[x_])
            values = values[values>0]
            cil = np.percentile(values,2.5)
            ciu = np.percentile(values,97.5)
            y_errorbars[kx] = [y_all[kx],cil,ciu]
        y_errorbar_metastate.append(y_errorbars)

    return y_errorbar_metastate, x_all_metastate


def histcount(array,binlims):
    binmin,binmax=binlims
    c=0
    for x in array:
        if binmin<x<binmax:
            c+=1
    return c

def gaussian(x,a,b,c):
    return a*np.exp(-((x-b)**2)/(2*c**2))


def histogram(array,n_bins,xmin,xmax,norm_amp=False,norm_var=False,plot=False,fit_gaussian=False):
    '''
    Returns the bincenters, the frequencies and the binsizes of an histogram.
    In case fit_gaussian=True, returns also the gaussian fitted to the histogram (r,g)
    xmin and xmax are the points where we want to start or end an histogram. In case norm_var=True,
    this is just the distance from the mean, in either side
    norm_var normalizes by the variance
    norm_amp normalizes by the amplitude
    plot plots the histogram
    fit_gaussian fits a gaussian distribution to the histogram
    '''
    binsize=(xmax-xmin)/float(n_bins)
    bincenters=[]
    x=xmin
    mean=array.mean()
    while x<xmax:
        bincenters.append(x+binsize/2.)
        x+=binsize
    if norm_var:
        array=array-array.mean()
        array=array/array.std()
        histogram=[]
        for b in bincenters:
            binlims=(b-binsize/2.,b+binsize/2.)
            histogram.append(histcount(array,binlims))
    else:
        histogram=[]
        for b in bincenters:
            binlims=(b-binsize/2.,b+binsize/2.)
            histogram.append(histcount(array,binlims))
    hist=np.array(histogram)
    if norm_amp:
        hist=hist/np.float(np.sum(hist))
    if fit_gaussian:
        sigma=array.std()
        a=hist.max()
        if norm_var:
            sigma=1
        r=np.linspace(xmin-binsize/2.,xmax+binsize/2.,n_bins*10)
        g=gaussian(r,a,array.mean(),sigma)
        print(sigma,a,array.mean())
        if norm_var:
            plt.plot(r+mean,g,c='k')
        else:
            plt.plot(r,g,c='k')
    if plot:
        if norm_var:
            plt.bar(np.array(bincenters)+mean,hist,binsize)
        else:
            plt.bar(np.array(bincenters),hist,binsize)
        plt.show()
    if norm_var and fit_gaussian:
        return bincenters,hist,binsize,r+mean,g
    if norm_var==False and fit_gaussian:
        return bincenters,hist,binsize,r,g
    else:
        return bincenters,hist,binsize


def acf(x, lags=500, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape)
    exclude = np.cumsum(exclude.astype(int))

    # from stackexchange
    x = x - x.mean()  # remove mean
    if type(lags) is int:
        lags = range(lags)

    C = ma.zeros((len(lags),))
    sigma2 = x.var()
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = 1
        elif l >= x.shape[0]:
            C[i] = ma.masked
        else:
            x0 = x[:-l].copy()
            x1 = x[l:].copy()
            reject = (exclude[l:]-exclude[:-l])>0
            x0[reject] = ma.masked
            x1[reject] = ma.masked
            C[i] = (x0*x1).mean()/sigma2
    return C


def acf_across(Xs, lags=500):
    x_mean = ma.hstack(Xs).mean()
    if type(lags) is int:
        lags = range(lags)
    Cs=[]
    sigma2 = ma.hstack(Xs).var()
    for i, l in enumerate(lags):
        if l == 0:
            Cs.append(sigma2)
        else:
            cs=[]
            for x in Xs:
                if x.shape[0] > 1:
                    x_=x-x_mean
                    x0 = x_[:-l].copy()
                    x1 = x_[l:].copy()
                    cs.append((x0*x1).compressed())
            Cs.append(np.hstack(cs))
    return sigma2,Cs


def ccf(x, y, lags, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape)
    exclude = np.cumsum(exclude.astype(int))

    x = x - x.mean()  # remove mean
    y = y - y.mean()
    if type(lags) is int:
        lags = np.arange(-lags,lags)
    C = ma.zeros((len(lags),))
    sigma2 = x.std()*y.std()
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = (x*y).mean()/sigma2
        else:
            if l > 0:
                x0 = x[:-l].copy()
                y1 = y[l:].copy()
            else:
                x0 = y[:l].copy()
                y1 = x[-l:].copy()
            reject = (exclude[l:]-exclude[:-l])>0
            x0[reject] = ma.masked
            y1[reject] = ma.masked

            C[i] = (x0*y1).mean()/sigma2
    return C

def dotacf(x, lags=500, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape)
    exclude = np.cumsum(exclude.astype(int))

    if type(lags) is int:
        lags = range(lags)
    C = ma.zeros((len(lags),))
    size=len(x)
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = (x*x).sum(axis=1).mean()
        else:
            x0 = x[:-l, :].copy()
            x1 = x[l:, :].copy()
            reject = (exclude[l:]-exclude[:-l])>0
            x0[reject, :] = ma.masked
            x1[reject, :] = ma.masked
            C[i] = (x0*x1).sum(axis=1).mean()
    return C


def density_plot(X,Y,xrange,yrange,n_grid_x,n_grid_y,border=5,smooth=False,log=False):
    # view area range
    view_xmin,view_xmax=xrange
    view_ymin,view_ymax=yrange

    # get data
    xl = X
    yl = Y

    # get visible data points
    xlvis = []
    ylvis = []
    for i in range(0,len(xl)):
        if view_xmin < xl[i] < view_xmax and view_ymin < yl[i] < view_ymax:
            xlvis.append(xl[i])
            ylvis.append(yl[i])

    kx = (n_grid_x - 1) / (view_xmax - view_xmin)
    ky = (n_grid_y - 1) / (view_ymax - view_ymin)
    imgw = (n_grid_x + 2 * border)
    imgh = (n_grid_y + 2 * border)
    img = np.zeros((imgh,imgw))
    for x, y in zip(xl,yl):
        ix = int((x - view_xmin) * kx) + border
        iy = int((y - view_ymin) * ky) + border
        if 0 <= ix < imgw and 0 <= iy < imgh:
            img[iy][ix] += 1
    if smooth:
        if log:
            log_img=np.log10(img)
            sel=~np.isfinite(log_img)
            log_img[sel]=0.
            filtered_img=ndi.gaussian_filter(log_img, (border,border))
            return filtered_img
        else:
            return ndi.gaussian_filter(img, (border,border))  ## gaussian convolution
    else:
        if log:
            return np.log10(img)
        else:
            return img




def density_plot_z(x_data,y_data,z_data,size_x,size_y,xrange,yrange,sum_=False,log=False,smooth=True,conv_window=5):
    
    grid_x = np.arange(xrange[0],xrange[1]+size_x,size_x)
    grid_y = np.arange(yrange[0],yrange[1]+size_x,size_y)
    X,Y = np.meshgrid(grid_x,grid_y)
    img = np.zeros((grid_x.shape[0],grid_y.shape[0]))
    for kx,x in enumerate(grid_x):
        for ky,y in enumerate(grid_y):
            sel_x = np.logical_and(x_data<x+size_x,x_data>=x)
            sel_y = np.logical_and(y_data<y+size_y,y_data>=y)
            sel = np.logical_and(sel_x,sel_y)
            if sel.sum()>0:
                if sum_:
                    img[kx,ky] = z_data[sel].sum()
                else:
                    img[kx,ky] = z_data[sel].mean()
    img[np.isnan(img)]=0
    if smooth:
        return ndi.gaussian_filter(img, (conv_window,conv_window))  ## gaussian convolution
    else:
        return img
        

def KLDiv(P, Q):
    if P.shape[0] != Q.shape[0]:
        raise Exception()
    if np.any(~np.isfinite(P)) or np.any(~np.isfinite(Q)):
        raise Exception()
    Q = Q / Q.sum()
    P = P / P.sum(axis=0)
    dist = np.sum(P*np.log2(P/Q), axis=0)
    if np.isnan(dist):
        dist = 0
    return dist


def JSDiv(P, Q):
    if P.shape[0] != Q.shape[0]:
        raise Exception()
    Q = Q / Q.sum(axis=0)
    P = P / P.sum(axis=0)
    M = 0.5*(P+Q)
    dist = 0.5*KLDiv(P, M) + 0.5*KLDiv(Q, M)
    return dist


def state_lifetime(states,tau):
    durations=[]
    state_labels = []
    for state in np.sort(np.unique(states.compressed())):
        gaps = states==state
        gaps_boundaries = np.where(np.abs(np.diff(np.concatenate([[False], gaps, [False]]))))[0].reshape(-1, 2)
        durations.append(np.hstack(np.diff(gaps_boundaries[:]))*tau)
        state_labels.append(state)
    return durations, state_labels


def cumulative_dist(data,lims,label='label'):
    lim0,lim1=lims
    data = np.array(data)
    data = data[data>=lim0]
    data = data[data<=lim1]
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted,p


def complementary_cumulative_dist(data,lims,label='label'):
    lim0,lim1=lims
    data = np.array(data)
    data = data[data>=lim0]
    data = data[data<=lim1]
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted,1-p


def fit_tscales(x,y):

    def fcn2(params,x,data):
        m=params['m'].value
        b=params['b'].value
        #Fits the drift

        return ((m*x + b) - data)

    params = Parameters()
    params.add('m',   value= 1)
    params.add('b', value=1)

    # do fit, here with leastsq model
    result = minimize(fcn2, params, args=(x,y))

    p1 = result.params['m'].value
    p2 = result.params['b'].value

    return p1,p2


def state_lifetimes_stats(lifetimes_fish,n_states=2, t0=1, tf=600,n_times=1000):
    y_errorbar_metastate = []
    x_all_metastate = []
    m_all_metastate = []
    b_all_metastate = []
    tscales_metastate = []
    params_metastate = []
    mean_lifetimes = []
    for ks in range(n_states):
        print(ks)
        all_lt = np.hstack(lifetimes_fish[ks])
        x,y = cumulative_dist(all_lt,(t0,tf))
        y_all = 1-np.array([np.mean(y[x==x_unique]) for x_unique in np.unique(x)])
        x_all = np.sort(np.unique(x))
        x_all_metastate.append(x_all)
        mean_lifetimes.append(ma.mean(ma.hstack(lifetimes_fish[ks])))

        sel = y_all>0
        m,b = fit_tscales(x_all[sel],np.log(y_all[sel]))
        m_all_metastate.append(m)
        b_all_metastate.append(b)

        dict_y = {}
        for x_ in x_all:
            dict_y[x_] = []

        tscales_bootstrap = []
        params_bootstrap = []
        rates_bootstrap = []
        for ii in range(n_times):
            x,y = cumulative_dist(np.hstack(np.random.choice(lifetimes_fish[ks],len(lifetimes_fish[ks]))),(t0,tf))
            y = 1-np.array([np.mean(y[x==x_unique]) for x_unique in np.unique(x)])
            x = np.sort(np.unique(x))
            sel = y>0
            m,b = fit_tscales(x[sel],np.log(y[sel]))
            tscale = round(-1/m,2)
            tscales_bootstrap.append(tscale)
            rates_bootstrap.append((1/tscale))
            params_bootstrap.append(np.array([m,b]))
            for kx in range(len(y)):
                dict_y[x[kx]].append(y[kx])


        params_cil = np.percentile(params_bootstrap,2.5,axis=0)
        params_ciu = np.percentile(params_bootstrap,97.5,axis=0)

        tscales_cil = np.percentile(np.vstack(tscales_bootstrap),2.5,axis=0)
        tscales_ciu = np.percentile(np.vstack(tscales_bootstrap),97.5,axis=0)
        tscales_metastate.append([tscales_cil[0], tscales_ciu[0]])

        params_metastate.append([params_cil,params_ciu])

        y_errorbars = np.zeros((len(dict_y.keys()),3))
        for kx,x_ in enumerate(x_all):
            values = np.array(dict_y[x_])
            values = values[values>0]
            cil = np.percentile(values,2.5)
            ciu = np.percentile(values,97.5)
            y_errorbars[kx] = [y_all[kx],cil,ciu]
        y_errorbar_metastate.append(y_errorbars)

    return y_errorbar_metastate, x_all_metastate, m_all_metastate, b_all_metastate, tscales_metastate, params_metastate, mean_lifetimes


import math
def polygon(sides, radius=1, rotation=0, translation=None):
    one_segment = math.pi * 2 / sides

    points = [
        (math.sin(one_segment * i + rotation) * radius,
         math.cos(one_segment * i + rotation) * radius)
        for i in range(sides)]

    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]

    return points

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
def draw_self_loop(ax,center, radius, fc='#2693de', ec='#000000', rwidth=0.001,theta1=-90, theta2=270, total_points=1, point_no=0):

    # Add the ring
    x = center[0] + radius*math.sin(point_no*((math.pi * 2)/total_points))
    y = center[1] + radius*math.cos(point_no*((math.pi * 2)/total_points))
    ring = mpatches.Wedge((x,y), radius, theta1, theta2, width=rwidth)
    # Triangle edges
#     offset = 0.02
#     xcent  = center[0] - radius + (rwidth/2)
#     left   = [xcent - offset, center[1]]
#     right  = [xcent + offset, center[1]]
#     bottom = [(left[0]+right[0])/2., center[1]-0.05]
#     arrow  = plt.Polygon([left, right, bottom, left])
    p = PatchCollection(
        [ring],
        edgecolor = ec,
        facecolor = fc
    )
    ax.add_collection(p)


def km_histogram(dtrajs, vmax, vmin,hist_size = 101, tau = 1):
    """Computes the drift and diffusion using Kramers-Moyal histogram binning method over multiple realizations of the data.
    Parameters:
        dtrajs: List of multiple realizations of a 1D time series
        vmax: Maximum value for calculation
        vmin: Minimum value for calculation
        hist_size: Number of bins to be
        tau: Delay parameter of the jump process
    Returns:
        drift: Drift component of the Fokker-Planck equation
        diff: Diffusion component of the Fokker-Planck equation
        bins: Bins over which the histogram was performed
    """

    bins = np.linspace(vmin - 1e-6,vmax+1e-6,hist_size)

    del_bin = []

    for b in range(bins.shape[0] - 1):
        del_cat = []
        for i in range(len(dtrajs)):
            dis = dtrajs[i][:]
            if len(dis) > 2:
                vs = np.where(np.logical_and(dis[:-tau] > bins[b], dis[:-tau] < bins[b+1]))[0]
                delval = dis[vs+tau] - dis[vs]
                if delval.size !=0:
                    del_cat.append(list(delval))
        del_cat = [item for sublist in del_cat for item in sublist]
        del_bin.append(del_cat)

    max_l = 0
    for delb in del_bin:
        if len(delb) > max_l:
            max_l = len(delb)

    maskd_delbin = ma.zeros((max_l,len(del_bin)))
    maskd_delbin[:] = np.nan

    for i,delb in enumerate(del_bin):
        maskd_delbin[:len(delb),i] = delb

    maskd_delbin = ma.masked_invalid(maskd_delbin)
    drift = (1/tau)*ma.mean(maskd_delbin,axis=0)
    diff = (0.5/tau)*ma.var(maskd_delbin,axis=0)

    drift[drift == ma.masked] = 0.
    diff[diff == ma.masked] = 0.

    return drift, diff, bins
