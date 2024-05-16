#data format library
#numpy
import numpy as np
import numpy.ma as ma
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
#load libraries
import json
import scipy.io as sio
import numpy as np
from scipy.interpolate import splev, splprep,interp1d, CubicSpline
import time as T
import matplotlib.pyplot as plt


class boutDef:
    def __init__(self, vec):
        self.numPoiss =  vec['FishNumber']
        self.begMove =  vec['BoutStart']
        self.endMove =  vec['BoutEnd']
        self.boutnum = 0
        self.tailAngle =  vec['TailAngle_Raw']
        self.posHeadX =  vec['HeadX']
        self.posHeadY =  vec['HeadY']
        self.rawheading =  vec['Heading_raw']
        self.correctedheading =  vec['Heading']
        self.posTailXVideoReferential =  vec['TailX_VideoReferential']
        self.posTailYVideoReferential =  vec['TailY_VideoReferential']
        self.posTailXHeadingReferential =  vec['TailX_HeadingReferential']
        self.posTailYHeadingReferential=  vec['TailY_HeadingReferential']
        self.tailAngleSmoothed=  vec['TailAngle_smoothed']
        self.freq =  vec['Bend_TimingAbsolute']
        self.freqX=  vec['Bend_Timing']
        self.freqY =  vec['Bend_Amplitude']
        self.param =  vec['param']
        self.posHeadX_int = self.posHeadX
        self.posHeadY_int = self.posHeadY
        self.speed = 0
        self.frequency = 0
        self.amp = 0
        self.nosc = 0
        self.angspeed = 0
        self.deltahead = 0
        self.time = 0 
        self.dist = 0
        self.disp = 0
        self.avgspeed = 0
        self.ispeeds = np.zeros(25)
        self.welltype = 0
        self.filename = 0
        self.wellnum = 0
        self.likelihood = 0
        self.taillength = 0
        self.tailarea = 0
        self.tailpc1 = 0
        self.tailpc2 = 0
        self.tailpc3 = 0
        self.tailangles = np.zeros((40,8))
        self.ibi_prev = 0
        self.ibi_next = 0
        self.warning = []


    def calc_bout_posHead_interp(self,seed=42):
        np.random.seed(seed)
        f,u = splprep([self.posHeadX + .1*np.random.randn(len(self.posHeadX)), self.posHeadY + .1*np.random.randn(len(self.posHeadX))],s = 10)
        new_points = splev(u, f)
        return new_points[0], new_points[1]

    #Speed in mm/sec
    def calc_speed(self, fps, px_to_mm):
        totaldist = self.calc_dist(px_to_mm)
        totaltime = self.calc_time(fps)
        return totaldist/totaltime

    #Instantaneous speed in mm/sec
    def calc_ispeed(self, fps, px_to_mm):
        numps = 6
        ispeeds = np.zeros(25)
        for j in range(min(len(self.posHeadX)-1,25)):
            if j >= len(self.posHeadX):
                ispeeds[j] = 0
            else:
                bXs = np.concatenate((self.posTailXVideoReferential[j][-numps:],[self.posHeadX[j]]))
                bYs = np.concatenate((self.posTailYVideoReferential[j][-numps:],[self.posHeadY[j]]))
                theta = np.arctan2((bYs[-1]-bYs[0]),(bXs[-1]-bXs[0]))
                delx = (self.posHeadY_int[j+1] - self.posHeadY_int[j])*px_to_mm
                dely = (self.posHeadX_int[j+1] - self.posHeadX_int[j])*px_to_mm
                del_ = np.sqrt(delx**2 + dely**2)
                phi = np.arctan2(dely,delx)
                ispeeds[j] = del_*np.cos(theta - phi)*fps
        return ispeeds

    #Frequency of oscillations in Hz
    def calc_frequency(self,fps):
        if type(self.freqX) is list:
            if len(self.freqX) > 1:
                return 0.5/(np.mean(np.asarray(self.freqX[1:]) - np.asarray(self.freqX[:-1]))/fps)
            else:
                return 0
    
    #Amplitude of oscillations
    def calc_amp(self):
        return np.max(np.abs(self.freqY))*180/np.pi

    #Number of oscillations
    def calc_nosc(self):
        if type(self.freq) is list:
            return len(self.freq)
        else:
            return 1.0

    #Mean angular speed in deg/sec
    def calc_angspeed(self,fps):
        totaltime = self.calc_time(fps)
        return self.calc_deltahead()/totaltime

    #Heading change in degrees
    def calc_deltahead(self):
        numps = 6
        bXs = np.concatenate((self.posTailXVideoReferential[0][-numps:],[self.posHeadX[0]]))
        bYs = np.concatenate((self.posTailYVideoReferential[0][-numps:],[self.posHeadY[0]]))
        slope0 = np.arctan2((bYs[-1]-bYs[0]),(bXs[-1]-bXs[0]))*180/np.pi
        
        bXs = np.concatenate((self.posTailXVideoReferential[-1][-numps:],[self.posHeadX[-1]]))
        bYs = np.concatenate((self.posTailYVideoReferential[-1][-numps:],[self.posHeadY[-1]]))
        slope1 = np.arctan2((bYs[-1]-bYs[0]),(bXs[-1]-bXs[0]))*180/np.pi
        delt = -(slope1 - slope0)
        if delt > 180:
            return 360 - delt
        elif delt < -180:
            return -(360 + delt)
        else:
            return delt

    #Bout time in seconds
    def calc_time(self,fps):
        return len(self.posHeadX)/fps

    #total distance travelled in mm
    def calc_dist(self,px_to_mm):
        dist1 = 0
        for j in range(len(self.posHeadX)-1):
            dist1 += np.sqrt((self.posHeadX_int[j+1] - self.posHeadX_int[j])**2 + (self.posHeadY_int[j+1] - self.posHeadY_int[j])**2)
        return dist1*px_to_mm

    #magnitude of displacement in mm
    def calc_disp(self,px_to_mm):
        disp1 = np.sqrt((self.posHeadX_int[-1] - self.posHeadX_int[0])**2 + (self.posHeadY_int[-1] - self.posHeadY_int[0])**2)
        return disp1*px_to_mm

    #Average speed in mm/s
    def calc_avgspeed(self,fps,px_to_mm):
        disp1 = self.calc_disp(px_to_mm)
        return disp1/self.calc_time(fps)

    #avg tail length in. mm
    def calc_taillength(self):
        return np.sum(np.abs(np.diff(self.tailAngleSmoothed)))

    #tail integral
    def calc_tailarea(self):
        return np.abs(np.sum(self.tailAngleSmoothed))

    #tailangles for all points
    def calc_tailangles_(self,maxT=50):
        numps = 8

        headx = self.posHeadX
        heady = self.posHeadY
        tailx = self.posTailXVideoReferential
        taily = self.posTailYVideoReferential
        tailangles_arr = ma.zeros((maxT,numps))

        for i in range(min(len(headx),tailangles_arr.shape[0])):
            
            # Take points from the tip to the swim bladder
            tailx_act = tailx[i][:-3]
            taily_act = taily[i][:-3]
            # Remove duplicates i.e 2 tail points assigned the same coordindates
            tailxy = np.vstack([tailx_act[::-1], taily_act[::-1]])
            _,idx = np.unique(tailxy,axis=1, return_index=True)
            new_tailxy = tailxy[:,np.sort(idx)]
            # Fit a spline to points and sample numps+1 equidistant points on the tail
            tck, u = splprep([new_tailxy[0],new_tailxy[1]],s=0)
            interp_i = np.linspace(0,1,numps+1)
            XY = splev(interp_i,tck)
            xs = XY[0]
            ys = XY[1]
            
            # Calculate tail angles by fixing axis as angle made by vector head -> swim bladder
            ang = np.arctan2(headx[i] - xs[0],heady[i] - ys[0])
            for j,jid in enumerate(np.arange(1,numps+1)):
                ang2 = np.arctan2(headx[i] - xs[jid],heady[i] - ys[jid])
                delang = ang2 - ang
                if np.abs(delang) < np.pi:
                    tailangles_arr[i,j] = delang
                elif delang > np.pi:
                    tailangles_arr[i,j] = delang - 2*np.pi
                elif delang < -np.pi:
                    tailangles_arr[i,j] = 2*np.pi + delang

        # Take cumulative sum to match JM and also removes noise and puts more emphasis on the last tail point
        # If interested, smooth tail angle using savgol_filter
        # return savgol_filter(np.cumsum(tailangles_arr, axis=1), 7,3, axis=0)
        return ma.cumsum(tailangles_arr,axis=1)

#    def calc_tailangles(self,maxT):
#        numps = 3
#        
#        headx = self.posHeadX
#        heady = self.posHeadY
#        tailx = self.posTailXVideoReferential
#        taily = self.posTailYVideoReferential
#    
#        tailangles_arr = np.zeros((maxT,7))
#        for i in range(min(len(self.posHeadX),tailangles_arr.shape[0])):
#            ang = np.arctan2(heady[i] - taily[i][-3],headx[i] - tailx[i][-3])
#            for j in range(tailangles_arr.shape[1]):
#                ang2 = np.arctan2(heady[i] - taily[i][j],headx[i] - tailx[i][j])
#                delang = ang2 - ang
#                if np.abs(delang) < np.pi:
#                    tailangles_arr[i,j] = delang
#                elif delang > np.pi:
#                    tailangles_arr[i,j] = delang - 2*np.pi
#                elif delang < -np.pi:
#                    tailangles_arr[i,j] = 2*np.pi + delang
#                #print(i,j,ang,ang2,tailangles_arr[i,j])
#        return tailangles_arr

    #calculate heading
    def calc_heading(self):
        return np.arctan2(self.posHeadY_int[-1] - self.posHeadY_int[-2],self.posHeadX_int[-1] - self.posHeadX_int[-2])*180.0/np.pi


def angle_between(vector1, vector2):
    """ Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(
        np.stack((v1_u[-2:], v2_u[-2:]))
    )
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    angle = sign * np.arccos(dot_p)
    return angle


# @njit(cache=True, nogil=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def pca_transform_shuffle(theta_Bouts,minT,dT,npoints,div=100,seed=42):
    """
    Returns eigenvalues, modes of the bout matrix and projects the bouts to the eigenspace

    Parameters:
        theta_Bouts: Matrix consisting of a collection of bouts
        minT:
        dT:
        npoints:
        div: Number of recordings to take. default = 100
        seed: Random seed to shuffle the dataset. default = 42

    Returns:
        eigvals: eigenvalues of the bout matrix
        modes: eigenvectors of the bout matrix
        phspace_all: projection of the bouts to the space spanned by the modes

    """
    theta_traj_Bouts = theta_Bouts[:,:,dT:minT+dT,:].reshape((theta_Bouts.shape[0],theta_Bouts.shape[1],minT*npoints))
    all_Bouts = ma.concatenate(theta_traj_Bouts,axis=0)
    
    X = all_Bouts-all_Bouts.mean(axis=0)
    cov = ma.cov(X.T)
    
    eigvals,modes = np.linalg.eig(cov)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    modes = modes[:,sorted_indices]

    phspace_all = X.dot(modes)
    
    return eigvals,modes,phspace_all

def pca_transform(theta_Bouts,minT,dT,npoints):
    """
    Returns eigenvalues, modes of the bout matrix and projects the bouts to the eigenspace

    Parameters:
        theta_Bouts: Matrix consisting of a collection of bouts
        minT: Time point for bout start
        dT: Maximum bout duration
        npoints: Total number of tail points

    Returns:
        eigvals: eigenvalues of the bout matrix
        modes: eigenvectors of the bout matrix
        phspace_all: projection of the bouts to the space spanned by the modes

    """
    theta_traj_Bouts = theta_Bouts[:,:,dT:minT+dT,:].reshape((theta_Bouts.shape[0],theta_Bouts.shape[1],minT*npoints))
    all_Bouts = ma.concatenate(theta_traj_Bouts,axis=0)
    
    X = all_Bouts-all_Bouts.mean(axis=0)
    cov = ((X.T).dot(X))/X.shape[0]
    
    eigvals,modes = np.linalg.eig(cov)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    modes = modes[:,sorted_indices]

    phspace_all = X.dot(modes)
    
    return eigvals,modes,phspace_all


def smooth_traj(X_,window_length=9,poly_order=3):
    """
    Returns the smooth heading using Savoy-Golay filter

    Parameters:
        X_: heading vector
        window_length: window length of SavGol filter. Default = 9
        ploy_order: the order of the polynomial to fit using the SavGol filter. Default = 3

    Returns:
        X_: Masked heading vector array
        X_smooth: Masked and smoothed heading vector
        S_smooth: Masked and smoothed derivative of the heading vector
    """
    sel = np.sum(np.abs(X_),axis=1)>0
    X_smooth = ma.zeros(X_.shape)
    S_smooth = ma.zeros(X_.shape)
    X_smooth[sel] = savgol_filter(X_[sel], window_length, poly_order, deriv=0, mode='nearest',axis=0)
    S_smooth[sel] = savgol_filter(X_[sel], window_length, poly_order, deriv=1, mode='nearest',axis=0)
    X_smooth[~sel] = ma.masked
    S_smooth[~sel] = ma.masked
    X_ = ma.array(X_)
    X_[X_==0] = ma.masked
    return X_,X_smooth,S_smooth


def smooth_angle_traj(X_,window_length=9,poly_order=3):
    """
    Returns the smooth tail angle using Savoy-Golay filter

    Parameters:
        X_: tail angle
        window_length: window length of SavGol filter. Default = 9
        ploy_order: the order of the polynomial to fit using the SavGol filter. Default = 3

    Returns:
        X_: Masked tail angle 
        X_smooth: Masked and smoothed heading vector
        S_smooth: Masked and smoothed derivative of the heading vector
    """
    sel = np.abs(X_)>0
    X_smooth = ma.zeros(X_.shape)
    S_smooth = ma.zeros(X_.shape)
    X_smooth[sel] = savgol_filter(X_[sel], window_length, poly_order, deriv=0, mode='nearest')
    S_smooth[sel] = savgol_filter(X_[sel], window_length, poly_order, deriv=1, mode='nearest')
    X_smooth[~sel] = ma.masked
    S_smooth[~sel] = ma.masked
    X_ = ma.array(X_)
    X_[X_==0] = ma.masked
    return X_,X_smooth,S_smooth



def reject(b):
    if b.time < 0.04 or b.time > 1.2:
        return True
    if b.dist > 25 or b.dist < 0.0 :
        return True
    if b.speed > 50 or b.speed < 1:
        return True
    if np.abs(b.deltahead) > 180:
        return True
    return False


def get_bouts_textdataset(foldername, filenames, welltypes, px_to_mm, fps, maxT,seed,exp_type):
    bouts_all = []

    for k,f in enumerate(filenames):
        with open(foldername + f + '/results_' +f+'.txt') as name:
            data = json.load(name)
            welltype = welltypes[k]
            numrejects = 0
            numaccepts = 0

            for j in range(len(welltype)):
                if exp_type != welltype[j]:
                    continue

                if len(data['wellPoissMouv'][j][0]) == 0:
                    continue
                numbouts = len(data['wellPoissMouv'][j][0])
                for i in range(numbouts):
                    bouts_temp = data['wellPoissMouv'][j][0][i]
                    b = boutDef(bouts_temp)
                    b.boutnum = i
                    b.posHeadX_int,b.posHeadY_int = b.calc_bout_posHead_interp(seed)
                    b.speed = b.calc_speed(fps,px_to_mm)
                    b.frequency = b.calc_frequency(fps)
                    b.amp = b.calc_amp()
                    b.nosc = b.calc_nosc()
                    b.angspeed = b.calc_angspeed(fps)
                    b.deltahead = b.calc_deltahead()
                    b.time = b.calc_time(fps)
                    b.dist = b.calc_dist(px_to_mm)
                    b.disp = b.calc_disp(px_to_mm)
                    b.avgspeed = b.calc_avgspeed(fps,px_to_mm)
                    b.welltype = welltype[j]
                    b.filename = f
                    b.taillength = b.calc_taillength()
                    b.tailarea = b.calc_tailarea()
                    b.tailangles = b.calc_tailangles_(maxT)
                    b.ispeeds = b.calc_ispeed(fps,px_to_mm)
                    b.wellnum = j
                    if i < numbouts-1:
                        bouts_temp_next = data['wellPoissMouv'][j][0][i+1]
                        b_next = boutDef(bouts_temp_next)
                        b.ibi_next = (b_next.begMove - b.endMove)/fps
                    if i > 0:
                        bouts_temp_prev = data['wellPoissMouv'][j][0][i-1]
                        b_prev = boutDef(bouts_temp_prev)
                        b.ibi_prev = (b.begMove - b_prev.endMove)/fps

                    if reject(b):
                        numrejects += 1
                        continue
                    else:
                        numaccepts += 1
                    bouts_all += [b]
            print(foldername + f + '/results_' +f+'.txt',numrejects, numaccepts, 1.0*numrejects/(numaccepts + numrejects + 1))
    return bouts_all

#Pool together bout information from a dataset and store in a dictionary
def pool_data(bout_dataset):
    numbouts = len(bout_dataset)
    data_collected = {'speeds':np.zeros(numbouts), 'frequencys':np.zeros(numbouts),'amps':np.zeros(numbouts),'noscs':np.zeros(numbouts),'angspeeds':np.zeros(numbouts),'deltaheads':np.zeros(numbouts),'dists':np.zeros(numbouts)\
                      ,'times':np.zeros(numbouts),'avgspeeds':np.zeros(numbouts),'disps':np.zeros(numbouts),'tailareas':np.zeros(numbouts)}

    for i,b in enumerate(bout_dataset):
        data_collected['speeds'][i] = b.speed
        data_collected['frequencys'][i] = b.frequency
        data_collected['amps'][i] = b.amp
        data_collected['noscs'][i] = b.nosc
        data_collected['angspeeds'][i] = b.angspeed
        data_collected['deltaheads'][i] = b.deltahead
        data_collected['times'][i] = b.time
        data_collected['dists'][i] = b.dist
        data_collected['disps'][i] = b.disp
        data_collected['avgspeeds'][i] = b.avgspeed
        data_collected['tailareas'][i] = b.tailarea
    return data_collected

#Collect two bout info
def collect_two_consecutive_bouts(bout_dataset, fps, px_to_mm):
    collection = []
    currfilename = bout_dataset[0].filename
    currwellnum = bout_dataset[0].wellnum
    for i,b in enumerate(bout_dataset[:-1]):
        b_next = bout_dataset[i+1]
        if b_next.filename == currfilename and b_next.wellnum == currwellnum:
            ibi = (b_next.begMove - b.endMove)/fps
            dist_bout = px_to_mm*np.sqrt((b_next.posHeadX[0] - b.posHeadX[-1])**2 + (b_next.posHeadY[0] - b.posHeadY[-1])**2)
            if ibi > 10 or dist_bout > 4:
                continue
            else:
                collection += [[b,b_next,ibi]]
        else:
            currfilename = b_next.filename
            currwellnum = b_next.wellnum
    return collection

#Collect three bout info
def collect_three_consecutive_bouts(bout_dataset, fps, px_to_mm):
    collection = []
    currwellnum = bout_dataset[0].wellnum
    for i,b in enumerate(bout_dataset[:-2]):
        b_next = bout_dataset[i+1]
        b_nextnext = bout_dataset[i+2]
        if (b_next.filename == currfilename and b_next.wellnum == currwellnum) and (b_nextnext.filename == currfilename and b_nextnext.wellnum == currwellnum):
            ibi = (b_next.begMove - b.endMove)/fps
            ibi2 = (b_nextnext.begMove - b_next.endMove)/fps
            dist_bout = px_to_mm*np.sqrt((b_next.posHeadX[0] - b.posHeadX[-1])**2 + (b_next.posHeadY[0] - b.posHeadY[-1])**2)
            dist_bout2 = px_to_mm*np.sqrt((b_nextnext.posHeadX[0] - b_next.posHeadX[-1])**2 + (b_nextnext.posHeadY[0] - b_next.posHeadY[-1])**2)

            if (ibi > 10 or dist_bout > 4) or (ibi2 > 10 or dist_bout2 > 4):
                continue
            else:
                collection += [[b,b_next,b_nextnext]]
        else:
            currfilename = b_next.filename
            currwellnum = b_next.wellnum
    return collection

#Collect four bout info
def collect_four_consecutive_bouts(bout_dataset, fps, px_to_mm):
    collection = []
    currfilename = bout_dataset[0].filename
    currwellnum = bout_dataset[0].wellnum
    for i,b in enumerate(bout_dataset[:-3]):
        b_next = bout_dataset[i+1]
        b_nextnext = bout_dataset[i+2]
        b_nextnextnext = bout_dataset[i+3]
        if (b_next.filename == currfilename and b_next.wellnum == currwellnum) and (b_nextnext.filename == currfilename and b_nextnext.wellnum == currwellnum) and (b_nextnextnext.filename == currfilename and b_nextnextnext.wellnum == currwellnum):
            ibi = (b_next.begMove - b.endMove)/fps
            ibi2 = (b_nextnext.begMove - b_next.endMove)/fps
            ibi3 = (b_nextnextnext.begMove - b_nextnext.endMove)/fps
            dist_bout = px_to_mm*np.sqrt((b_next.posHeadX[0] - b.posHeadX[-1])**2 + (b_next.posHeadY[0] - b.posHeadY[-1])**2)
            dist_bout2 = px_to_mm*np.sqrt((b_nextnext.posHeadX[0] - b_next.posHeadX[-1])**2 + (b_nextnext.posHeadY[0] - b_next.posHeadY[-1])**2)
            dist_bout3 = px_to_mm*np.sqrt((b_nextnextnext.posHeadX[0] - b_nextnext.posHeadX[-1])**2 + (b_nextnextnext.posHeadY[0] - b_nextnext.posHeadY[-1])**2)


            if (ibi > 10 or dist_bout > 4) or (ibi2 > 10 or dist_bout2 > 4) or (ibi3 > 10 or dist_bout3 > 4):
                continue
            else:
                collection += [[b,b_next,b_nextnext, b_nextnextnext]]
        else:
            currfilename = b_next.filename
            currwellnum = b_next.wellnum
    return collection

#Collect continuous set of bouts
def collect_trajectories(bout_dataset, fps, px_to_mm):
    collection = []
    currfilename = bout_dataset[0].filename
    currwellnum = bout_dataset[0].wellnum
    print(currfilename, currwellnum)
    currtraj = [bout_dataset[0]]
    for i,b in enumerate(bout_dataset[:-1]):
        b_next = bout_dataset[i+1]
        if b_next.filename == currfilename and b_next.wellnum == currwellnum:
            currtraj += [b_next]
        else:
            if len(currtraj) > 30:
                collection += [currtraj]
            currtraj = [b_next]
            currfilename = b_next.filename
            currwellnum = b_next.wellnum
    return collection

#Collect continuous set of bouts with no spacings between bouts
def collect_trajectories_nospacings(bout_dataset, fps, px_to_mm):
    collection = []
    currfilename = bout_dataset[0].filename
    currwellnum = bout_dataset[0].wellnum
    currtraj = [bout_dataset[0]]
    for i,b in enumerate(bout_dataset[:-1]):
        b_next = bout_dataset[i+1]
        ibi = (b_next.begMove - b.endMove)/fps
        dist_bout = px_to_mm*np.sqrt((b_next.posHeadX[0] - b.posHeadX[-1])**2 + (b_next.posHeadY[0] - b.posHeadY[-1])**2)
        if b_next.filename == currfilename and b_next.wellnum == currwellnum and ibi < 5 and dist_bout < 4:
            currtraj += [b_next]
        else:
            if len(currtraj) > 30:
                collection += [currtraj]
            currtraj = [b_next]
            currfilename = b_next.filename
            currwellnum = b_next.wellnum
    return collection

def collect_data_hmm(trajs_nospacings):
    nsamples = 0
    for t in trajs_nospacings:
        nsamples += len(t)

    data_hmm = np.zeros((nsamples,6))
    lengths = np.zeros(len(trajs_nospacings), dtype = int)
    for i,t in enumerate(trajs_nospacings):
        lengths[i] = len(t)
        for j in range(len(t)):
            data_hmm[np.sum(lengths[:i])+j][0] = np.abs(t[j].deltahead)
            data_hmm[np.sum(lengths[:i])+j][1] = t[j].speed
            data_hmm[np.sum(lengths[:i])+j][2] = t[j].taillength
            data_hmm[np.sum(lengths[:i])+j][3] = t[j].tailpc1
            data_hmm[np.sum(lengths[:i])+j][4] = t[j].tailpc2
            data_hmm[np.sum(lengths[:i])+j][5] = t[j].tailpc3

    return data_hmm, lengths

def collect_data_hmm_other(trajs_nospacings):
    nsamples = 0
    for t in trajs_nospacings:
        nsamples += len(t)

    data_hmm = np.zeros((nsamples,4))
    lengths = np.zeros(len(trajs_nospacings), dtype = int)
    for i,t in enumerate(trajs_nospacings):
        lengths[i] = len(t)
        for j in range(len(t)):
            data_hmm[np.sum(lengths[:i])+j][0] = np.mean(t[j].posHeadX)
            data_hmm[np.sum(lengths[:i])+j][1] = np.mean(t[j].posHeadY)
            data_hmm[np.sum(lengths[:i])+j][2] = t[j].dist
            data_hmm[np.sum(lengths[:i])+j][3] = t[j].angspeed

    return data_hmm, lengths

def collect_trajectory_hmm(traj_nospacings):
    data_hmm = np.zeros((len(traj_nospacings),4))
    for j in range(len(traj_nospacings)):
        data_hmm[j][0] = np.abs(traj_nospacings[j].angspeed)*1e-3
        data_hmm[j][1] = traj_nospacings[j].speed
        data_hmm[j][2] = traj_nospacings[j].time
        data_hmm[j][3] = traj_nospacings[j].amp

    return data_hmm

def get_tailangles(dataset):
    tailangles_all = np.zeros((len(dataset),112))
    for i,b in enumerate(dataset):
        tailangles_all[i] = np.abs(b.tailangles[:16,:].flatten())
    return tailangles_all

def update_tail_pcas(bouts,pcs):
    for i,b in enumerate(bouts):
        b.tailpc1 = pcs[i,0]
        b.tailpc2 = pcs[i,1]
        b.tailpc3 = pcs[i,2]
        b.tailpc4 = pcs[i,3]
    return bouts


def calc_tailangles(bout,maxT=30):
    numps = 8

    headx = bout['HeadX']
    heady = bout['HeadY']
    tailx = bout['TailX_VideoReferential']
    taily = bout['TailY_VideoReferential']

    tailangles_arr = np.zeros((maxT,numps))
    for i in range(min(len(headx),tailangles_arr.shape[0])):
        ang = np.arctan2(heady[i] - taily[i][-2],headx[i] - tailx[i][-2])
        for j,jid in enumerate(np.arange(0,numps)):
#             print(jid)
            ang2 = np.arctan2(heady[i] - taily[i][jid],headx[i] - tailx[i][jid])
            delang = ang2 - ang
            if np.abs(delang) < np.pi:
                tailangles_arr[i,j] = delang
            elif delang > np.pi:
                tailangles_arr[i,j] = delang - 2*np.pi
            elif delang < -np.pi:
                tailangles_arr[i,j] = 2*np.pi + delang
            #print(i,j,ang,ang2,tailangles_arr[i,j])
    return tailangles_arr[:,::-1]#savgol_filter(np.cumsum(tailangles_arr[:,::-1],axis=1),7,3,axis=0)


def calc_tailangles_new(bout,maxT=30):
    numps = 8

    headx = bout['HeadX']
    heady = bout['HeadY']
    tailx = bout['TailX_VideoReferential']
    taily = bout['TailY_VideoReferential']


    tailangles_arr = ma.zeros((maxT,numps))
    for i in range(min(len(headx),tailangles_arr.shape[0])):
        
        # Take points from the tip to the swim bladder
        tailx_act = tailx[i][:-3]
        taily_act = taily[i][:-3]
        # Remove duplicates i.e 2 tail points assigned the same coordindates
        tailxy = np.vstack([tailx_act[::-1], taily_act[::-1]])
        _,idx = np.unique(tailxy,axis=1, return_index=True)
        new_tailxy = tailxy[:,np.sort(idx)]
        # Fit a spline to points and sample numps+1 equidistant points on the tail
        tck, u = splprep([new_tailxy[0],new_tailxy[1]],s=0)
        interp_i = np.linspace(0,1,numps+1)
        XY = splev(interp_i,tck)
        xs = XY[0]
        ys = XY[1]
        
        # Calculate tail angles by fixing axis as angle made by vector head -> swim bladder
        ang = np.arctan2(headx[i] - xs[0],heady[i] - ys[0])
        for j,jid in enumerate(np.arange(1,numps+1)):
            ang2 = np.arctan2(headx[i] - xs[jid],heady[i] - ys[jid])
            delang = ang2 - ang
            if np.abs(delang) < np.pi:
                tailangles_arr[i,j] = delang
            elif delang > np.pi:
                tailangles_arr[i,j] = delang - 2*np.pi
            elif delang < -np.pi:
                tailangles_arr[i,j] = 2*np.pi + delang

    # Take cumulative sum to match JM and also removes noise and puts more emphasis on the last tail point
    # If interested, smooth tail angle using savgol_filter
    # return savgol_filter(np.cumsum(tailangles_arr, axis=1), 7,3, axis=0)
    return tailangles_arr


def calc_tailangles_splinefit_newzz2(bout,maxT=50):
    """
    Calculates the tail angles by skeletonizing only the tail. Requires to fit a spline on the points output by zebrazoom
    
    Paramters:
        bout: bout information output of zebrazoom. bout = data['wellPoissMouv'][numwell][numanimal][numbout]
        maxT: Maximum number of frames to capture. Set according to frame rate of video. Ideally take 300ms
    
    Returns:
        Cumulative sum of the tail angles
    """
    numps = 8

    headx = bout['HeadX']
    heady = bout['HeadY']
    tailx = bout['TailX_VideoReferential']
    taily = bout['TailY_VideoReferential']

    print(tailx)
    print(taily.shape)

    tailangles_arr = ma.zeros((maxT,numps))
    #count = 0
    for i in range(min(len(headx),tailangles_arr.shape[0])):
        
        # Take points from the tip to the swim bladder
        tailx_act = tailx[i][3:]
        taily_act = taily[i][3:]
        # Remove duplicates i.e 2 tail points assigned the same coordindates
        tailxy = np.vstack([tailx_act, taily_act])
        _,idx = np.unique(tailxy,axis=1, return_index=True)
        #if len(idx)<=2:
        #    count += 1
        #    continue
        new_tailxy = tailxy[:,np.sort(idx)]
        # Fit a spline to points and sample numps+1 equidistant points on the tail
        tck, u = splprep([new_tailxy[0],new_tailxy[1]],s=0)
        interp_i = np.linspace(0,1,numps+1)
        XY = splev(interp_i,tck)
        xs = XY[0]
        ys = XY[1]
        
        # Calculate tail angles by fixing axis as angle made by vector head -> swim bladder
        ang = np.arctan2(headx[i] - xs[0],heady[i] - ys[0])
        for j,jid in enumerate(np.arange(1,numps+1)):
            ang2 = np.arctan2(headx[i] - xs[jid],heady[i] - ys[jid])
            delang = ang2 - ang
            if np.abs(delang) < np.pi:
                tailangles_arr[i,j] = delang
            elif delang > np.pi:
                tailangles_arr[i,j] = delang - 2*np.pi
            elif delang < -np.pi:
                tailangles_arr[i,j] = 2*np.pi + delang

    # Take cumulative sum to match JM and also removes noise and puts more emphasis on the last tail point
    # If interested, smooth tail angle using savgol_filter
    # return savgol_filter(np.cumsum(tailangles_arr, axis=1), 7,3, axis=0)
    return ma.cumsum(tailangles_arr, axis=1)


def calc_tailangles_splinefit_newzz(bout,maxT=50):
    """
    Calculates the tail angles by skeletonizing only the tail. Requires to fit a spline on the points output by zebrazoom

    Paramters:
        bout: bout information output of zebrazoom. bout = data['wellPoissMouv'][numwell][numanimal][numbout]
        maxT: Maximum number of frames to capture. Set according to frame rate of video. Ideally take 300ms

    Returns:
        Cumulative sum of the tail angles
    """
    numps = 8

    headx = bout['HeadX']
    heady = bout['HeadY']
    tailx = bout['TailX_VideoReferential']
    taily = bout['TailY_VideoReferential']

    tailangles_arr = ma.zeros((maxT,numps))
    count = 0
    for i in range(min(len(headx),tailangles_arr.shape[0])):

        # Take points from the tip to the swim bladder
        tailx_act = tailx[i][3:]
        taily_act = taily[i][3:]
        # Remove duplicates i.e 2 tail points assigned the same coordindates
        tailxy = np.vstack([tailx_act[::-1], taily_act[::-1]])
        _,idx = np.unique(tailxy,axis=1, return_index=True) 
        if len(idx) < 2:
            count += 1
            continue
        new_tailxy = tailxy[:,np.sort(idx)]
        # Fit a spline to points and sample numps+1 equidistant points on the tail
        tck, u = splprep([new_tailxy[0],new_tailxy[1]],s=0)
        interp_i = np.linspace(0,1,numps+1)
        XY = splev(interp_i,tck)
        xs = XY[0]
        ys = XY[1]
        # Calculate tail angles by fixing axis as angle made by vector head -> swim bladder
        ang = np.arctan2(headx[i-count] - xs[0],heady[i-count] - ys[0])
        for j,jid in enumerate(np.arange(1,numps+1)):
            ang2 = np.arctan2(headx[i-count] - xs[jid],heady[i-count] - ys[jid])
            delang = ang2 - ang
            if np.abs(delang) < np.pi:
                tailangles_arr[i-count,j] = delang
            elif delang > np.pi:
                tailangles_arr[i-count,j] = delang - 2*np.pi
            elif delang < -np.pi:
                tailangles_arr[i-count,j] = 2*np.pi + delang

    # Take cumulative sum to match JM and also removes noise and puts more emphasis on the last tail point
    # If interested, smooth tail angle using savgol_filter
    # return savgol_filter(np.cumsum(tailangles_arr, axis=1), 7,3, axis=0)
    return ma.cumsum(tailangles_arr,axis=1)


def calc_tailangles_splinefit_newzz3(bout,maxT=50):
    """
    Calculates the tail angles by skeletonizing only the tail. Requires to fit a spline on the points output by zebrazoom
    
    Paramters:
        bout: bout information output of zebrazoom. bout = data['wellPoissMouv'][numwell][numanimal][numbout]
        maxT: Maximum number of frames to capture. Set according to frame rate of video. Ideally take 300ms
    
    Returns:
        Cumulative sum of the tail angles
    """
    numps = 8

    headx = bout['HeadX']
    heady = bout['HeadY']
    tailx = bout['TailX_VideoReferential']
    taily = bout['TailY_VideoReferential']

    tailangles_arr = ma.zeros((maxT,numps))
    count = 0
    for i in range(min(len(headx),tailangles_arr.shape[0])):
        
        # Take points from the tip to the swim bladder
        tailx_act = tailx[i][0:]
        taily_act = taily[i][0:]
        # Remove duplicates i.e 2 tail points assigned the same coordindates
        tailxy = np.vstack([tailx_act[::-1], taily_act[::-1]])
        _,idx = np.unique(tailxy,axis=1, return_index=True)
        if len(idx) <= 3:
            count += 1
            continue
        new_tailxy = tailxy[:,np.sort(idx)]
        # Fit a spline to points and sample numps+1 equidistant points on the tail
        tck, u = splprep([new_tailxy[0],new_tailxy[1]],s=0)
        interp_i = np.linspace(0,1,numps+1)
        XY = splev(interp_i,tck)
        xs = XY[0]
        ys = XY[1]
        
        # Calculate tail angles by fixing axis as angle made by vector head -> swim bladder
        ang = np.arctan2(headx[i-count] - xs[0],heady[i-count] - ys[0])
        for j,jid in enumerate(np.arange(1,numps+1)):
            ang2 = np.arctan2(headx[i-count] - xs[jid],heady[i-count] - ys[jid])
            delang = ang2 - ang
            if np.abs(delang) < np.pi:
                tailangles_arr[i-count,j] = delang
            elif delang > np.pi:
                tailangles_arr[i-count,j] = delang - 2*np.pi
            elif delang < -np.pi:
                tailangles_arr[i-count,j] = 2*np.pi + delang

    # Take cumulative sum to match JM and also removes noise and puts more emphasis on the last tail point
    # If interested, smooth tail angle using savgol_filter
    # return savgol_filter(np.cumsum(tailangles_arr, axis=1), 7,3, axis=0)
    return ma.cumsum(tailangles_arr,axis=1)
