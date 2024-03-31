'''
MIT License

Copyright (c) 2021 
Michal Adamik, Jozef Goga, Jarmila Pavlovicova, Andrej Babinec, Ivan Sekaj

Faculty of Electrical Engineering and Information Technology 
of the Slovak University of Technology in Bratislava
Ilkovicova 3, 812 19 Bratislava 1, Slovak Republic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
from PIL import Image
from PIL import ImageDraw
from image4layer import Image4Layer
from numba import njit
from numba import prange
import math
import random

# ----------------- DEFINITIONS ------------------
COLOUR_BLACK = (0, 0, 0, 255) # black background colour
COLOUR_WHITE = (255, 255, 255, 255) # white background colour

# --- Parallel optimalization parameter ---
PARALLEL_MODE = True

# ----------------- FUNCTIONS  ------------------
@njit
def numbaseed(seed):
    random.seed(seed)

def selbest(Oldpop,Fvpop,Nums):
    """
    The function copies best chromosomes from the old population into the new 
    population required number of strings according to their fitness. The 
    number of the selected chromosomes depends on the input vector Nums. 
    The best chromosome is the chromosome with the lowest value of its 
    objective function.
    
    Args:
    
        Oldpop: The primary population
    
        Fvpop: The fitness vector of primary population (Oldpop)
    
        Nums:  Vector in the form: Nums=[number of copies of the best chromosome, ... ,
                                         number of copies of the i-th best chromosome, ...]
    
    Returns:
        
        Newpop: The selected population based on fitness and specified input vector Nums
        
        Newfit: The fitness vector of newly created population (Newpop)
        
    """
    
    if(len(Fvpop.shape)<=1):
        Fvpop = np.reshape(Fvpop, (1,len(Fvpop)))
    N = len(Nums)
    fit = np.sort(Fvpop)
    nix = np.argsort(Fvpop)
    Newpop0 = np.zeros((N, Oldpop.shape[1]))
    Newpop = np.zeros((int(np.sum(Nums)), Oldpop.shape[1]))
    Newfit = np.zeros((int(np.sum(Nums),)))
    
    for i in range(N):
        Newpop0[i,:] = Oldpop[nix[0,i],:]
    
    r = 0
    for i in range(N):
        for j in range(Nums[i]):
            Newpop[r,:] = Newpop0[i,:]
            Newfit[r] = fit[0,i]
            r += 1

    return [Newpop, Newfit]


def selsusOld(Oldpop,Fvpop,n):    ##########  DEPRECATED / NA ZMAZANIE  ############
    """
	
	DEPRECATED
	
    The function selects from the old population a required number of 
    chromosomes using the "stochastic universal sampling" method. Under this 
    selection method the number of a parent copies in the selected new 
    population is proportional to its fitness.
    
    Args:
    
        Oldpop: The primary population
    
        Fvpop: The fitness vector of primary population (Oldpop)
    
        n:  Required number of selected chromosomes
    
    Returns:
        
        Newpop: The selected population based on stochastic universal sampling 
        
        Newfit: The fitness vector of newly created population (Newpop)
        
    """
    
    if(len(Fvpop.shape)<=1):
        Fvpop = np.reshape(Fvpop, (1,len(Fvpop)))
    Newpop = np.zeros((n, Oldpop.shape[1]))
    OldFvpop = np.copy(Fvpop)
    Newfit = np.zeros((n,))
    lpop, lstring = Oldpop.shape
    Fvpop = Fvpop - np.min(Fvpop) + 1
    sumfv = np.sum(Fvpop).astype(np.float32)
    w0 = np.zeros((lpop+1,))

    for i in range(lpop):
        men = Fvpop[0,i]*sumfv
        w0[i] = 1.0/men # creation of inverse weights 
    w0[i+1] = 0
    w = np.zeros((lpop+1,))
    for i in np.arange(lpop-1,-1,-1):
        w[i] = w[i+1] + w0[i]
    maxw = np.max(w)
    if (maxw==0):
        maxw = 0.00001
    w = (w/maxw)*100 # weigth vector
    pdel = 100.0/n 
    b0 = np.random.uniform()*pdel - 0.00001
    b = np.zeros((n,))
    for i in range(1,n+1):
        b[i-1] = (i-1)*pdel + b0
    for i in range(n):
        for j in range(lpop):
            if(b[i]<w[j] and b[i]>w[j+1]):
                break
        Newpop[i,:] = Oldpop[j,:]
        Newfit[i] = OldFvpop[0,j]
    
    return [Newpop,Newfit]


@njit(fastmath=True)
def selsus(Oldpop,Fvpop,n):
    """
    The function selects from the old population a required number of 
    chromosomes using the "stochastic universal sampling" method. Under this 
    selection method the number of a parent copies in the selected new 
    population is proportional to its fitness.
    
    Args:
    
        Oldpop: The primary population
    
        Fvpop: The fitness vector of primary population (Oldpop)
    
        n:  Required number of selected chromosomes
    
    Returns:
        
        Newpop: The selected population based on stochastic universal sampling 
        
        Newfit: The fitness vector of newly created population (Newpop)
        
    """
    
    lpop, lstring = Oldpop.shape
    newpop = np.zeros((n, lstring))
    newfit = np.zeros((n,))
    invFit = 1 - Fvpop
	
    fitsum = 0
    for i in range(lpop):
        fitsum = fitsum + invFit[i]
		
    dist = fitsum / n
    pickpoint = random.random() * dist
    pickpointlist = np.zeros(n)
    for i in range(n):
        pickpointlist[i] = pickpoint
        pickpoint = pickpoint + dist
    
    cand = 0
    accumulatedfit = invFit[0]
    for i in range(n):
        while accumulatedfit < pickpointlist[i]:
            cand = cand + 1
            accumulatedfit = accumulatedfit + invFit[cand]
        newpop[i,:] = Oldpop[cand,:]
        newfit[i] = invFit[cand]
        
		
    return (newpop,newfit)


def genLinespop(popsize, Amps, Space):
    """
    The function generates a population of random real-coded chromosomes which 
    genes are limited by a two-row matrix Space. The first row of the matrix 
    Space consists of the lower limits and the second row consists of the upper
    limits of the possible values of genes representing coordinates of line 
    segments. The endpoints are generated by addition or substraction of 
    random real-numbers to the mutated genes. The absolute values of the added 
    values are limited by the vector Amp. 
    
    Args:
    
        popsize: The size of the population (number of chromosomes to be created)
    
        Amps: Matrix of endpoints generation boundaries in the form:
    			[real-number vector of lower limits;
                 real-number vector of upper limits];
    
        Space:  Matrix of gene boundaries in the form: 
	                [real-number vector of lower limits of genes;
                     real-number vector of upper limits of genes];
    
    Returns:
        
        Newpop: The created population of line segments inside 
                specified boudaries 
        
    """
    
    lpop, lstring = Space.shape
    Newpop = np.zeros((int(popsize),int(lstring)))
    if(len(Amps.shape)<=1):
        Amps = np.reshape(Amps, (1,len(Amps)))

    for r in range(int(popsize)):
        dX = Space[1,0] - Space[0,0]
        dY = Space[1,2] - Space[0,2]
        Newpop[r,0] = np.random.uniform()*dX + Space[0,0]
        Newpop[r,2] = np.random.uniform()*dY + Space[0,2]
        for s in [0,2]:
            if(Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if(Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]
        
        Newpop[r,1] = Newpop[r,0] + np.random.randint(2*Amps[0,1]+1) - Amps[0,1]  
        Newpop[r,3] = Newpop[r,2] + np.random.randint(2*Amps[0,3]+1) - Amps[0,3] 
        
        for s in [1,3]:
            if(Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if(Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]

    return Newpop


def mutLineOld(Oldpop,factor,Amps,Space):    ##########  DEPRECATED / NA ZMAZANIE  ############
    """
	
	DEPRECATED
	
    The function mutates the population of chromosomes with the intensity
	proportional to the parameter rate from interval <0;1>. Only a few genes  
	from a few chromosomes are mutated in the population. The mutations are 
    realized by addition or substraction of random real-numbers to the mutated 
    genes. The absolute values of the added constants are limited by the vector 
    Amp. Next the mutated strings are limited using boundaries defined in 
	a two-row matrix Space. The first row of the matrix represents the lower 
	boundaries and the second row represents the upper boundaries of 
    corresponding genes.
    
    Args:
    
        Oldpop: The primary population
        
        factor: The mutation rate, 0 =< rate =< 1
    
        Amps: Matrix of gene mutation boundaries in the form:
    			[real-number vector of lower limits;
                 real-number vector of upper limits];
    
        Space:  Matrix of gene boundaries in the form: 
	                [real-number vector of lower limits of genes;
                     real-number vector of upper limits of genes];
    
    Returns:
        
        Newpop: The mutated population 
        
    """
    
    if(len(Amps.shape)<=1):
        Amps = np.reshape(Amps, (1,len(Amps)))
    lpop, lstring = Oldpop.shape
    
    if (factor>1):
        factor = 1
    if (factor<0):
        factor = 0
    
    n = int(np.ceil(lpop*lstring*factor*np.random.uniform()))
    Newpop = np.copy(Oldpop)
    
    for i in range(n):
        rN = np.random.randint(2)
        r = int(np.ceil(np.random.uniform()*lpop))-1
        if (rN==0): # x1, y1
            Newpop[r,0] = Oldpop[r,0] + (2.0*np.random.uniform()-1)*Amps[0,0]
            Newpop[r,2] = Oldpop[r,2] + (2.0*np.random.uniform()-1)*Amps[0,2]
        elif (rN==1): # x2, y2
            Newpop[r,1] = Oldpop[r,1] + (2.0*np.random.uniform()-1)*Amps[0,1]
            Newpop[r,3] = Oldpop[r,3] + (2.0*np.random.uniform()-1)*Amps[0,3]
        
        for s in range(4):
            if (Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if (Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]

    return Newpop



@njit(fastmath=True)
def mutLine(Oldpop,factor,Amps,Space):
    """
    The function mutates the population of chromosomes with the intensity
	proportional to the parameter rate from interval <0;1>. Only a few genes  
	from a few chromosomes are mutated in the population. The mutations are 
    realized by addition or substraction of random real-numbers to the mutated 
    genes. The absolute values of the added constants are limited by the vector 
    Amp. Next the mutated strings are limited using boundaries defined in 
	a two-row matrix Space. The first row of the matrix represents the lower 
	boundaries and the second row represents the upper boundaries of 
    corresponding genes.
    
    Args:
    
        Oldpop: The primary population
        
        factor: The mutation rate, 0 =< rate =< 1
    
        Amps: Matrix of gene mutation boundaries in the form:
    			[real-number vector of lower limits;
                 real-number vector of upper limits];
    
        Space:  Matrix of gene boundaries in the form: 
	                [real-number vector of lower limits of genes;
                     real-number vector of upper limits of genes];
    
    Returns:
        
        Newpop: The mutated population 
        
    """
    
    lpop, lstring = Oldpop.shape
    
    if (factor>1):
        factor = 1
    if (factor<0):
        factor = 0
    
    n = int(np.ceil(lpop*lstring*factor*random.random() ))
    Newpop = np.copy(Oldpop)
    
    for i in range(n):
        rN = int(2*random.random())
        r = int(np.ceil(random.random()*lpop))-1
        if (rN==0): # x1, y1
            Newpop[r,0] = Oldpop[r,0] + (2.0*random.random()-1)*Amps[0]
            Newpop[r,2] = Oldpop[r,2] + (2.0*random.random()-1)*Amps[2]
        elif (rN==1): # x2, y2
            Newpop[r,1] = Oldpop[r,1] + (2.0*random.random()-1)*Amps[1]
            Newpop[r,3] = Oldpop[r,3] + (2.0*random.random()-1)*Amps[3]
        
        for s in range(4):
            if (Newpop[r,s]<Space[0,s]):
                Newpop[r,s] = Space[0,s]
            if (Newpop[r,s]>Space[1,s]):
                Newpop[r,s] = Space[1,s]

    return Newpop


@njit(fastmath=True)
def cpuDrawLine(x0, y0, x1, y1, image, line_color, line_width):

    
    dx = x1 - x0
    dy = y1 - y0
    phi = math.atan2(dy, dx)
    line_len = math.sqrt(dx*dx + dy*dy)
    
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)


    for x in range(image.shape[0]):
        found_line = False
        for y in range(image.shape[1]):    
            x_diff = float (x) - x0
            y_diff = float (y) - y0           
             
            x_trans = cos_phi*x_diff + sin_phi*y_diff
            y_trans = -sin_phi*x_diff + cos_phi*y_diff

            if (x_trans >= 0) and (x_trans <= line_len) and (abs(y_trans) <= (line_width/2)):
                found_line = True

                image[x, y] = line_color
            else:
                if found_line:
                    break
                
    return image

@njit(parallel=PARALLEL_MODE, fastmath=True)
def ParllelCPUEvalFitness(Pop, orimg, geimg, fitOld, LINE_WIDTH, BLEND_MODE):
    if(len(Pop.shape)<=1):
        Pop = np.reshape(Pop, (1,len(Pop)))
    lpop, lstring = Pop.shape
    Fit = np.zeros((lpop,))
    
    for i in prange(lpop):
        G = Pop[i,:]
        Fit[i] = cpuEvalPartialSimilarity(orimg, geimg, G, fitOld, LINE_WIDTH, BLEND_MODE)     
    return Fit

@njit(fastmath=True)
def cpuEvalPartialSimilarity(orimg, geimg, gen, prevFit, LINE_WIDTH, BLEND_MODE):
       
    # evaluate only part of the image
    minX = int(np.min(gen[0:2]))
    maxX = int(np.max(gen[0:2]))
    
    minX = minX-LINE_WIDTH
    maxX = maxX+LINE_WIDTH
    
    deltaX = int(maxX - minX) + 1
    
    minY = int(np.min(gen[2:4]))
    maxY = int(np.max(gen[2:4]))
    
    minY = minY-LINE_WIDTH
    maxY = maxY+LINE_WIDTH
    
    deltaY = int(maxY - minY) + 1
    
    # reconstruct previous similarity 
    tSum = ((prevFit*255.0)**2.0) * geimg.size
    
    draw = np.ones(dtype=np.uint8, shape=(deltaX, deltaY)) * 255
    
    line = (int(gen[0]-minX), int(gen[2]-minY),int(gen[1]-minX), int(gen[3]-minY))
    mask_img = np.zeros(dtype=np.uint8, shape=draw.shape)
    mask_img = cpuDrawLine(line[0], line[1], line[2], line[3], mask_img, 1, LINE_WIDTH)
    
    num_rows_o, num_cols_o = orimg[minX:minX+deltaX, minY:minY+deltaY].shape
    num_rows_m, num_cols_m = mask_img.shape

    if (num_rows_m != num_rows_o or num_cols_m != num_cols_o):
        return prevFit 

    if (num_rows_o == 0 or num_cols_o == 0 or num_rows_m == 0 or num_cols_m == 0):
        return prevFit 

    tgrey = np.multiply(orimg[minX:minX+deltaX, minY:minY+deltaY], mask_img)
       
    # if mask is an empty array
    if (tgrey.size == 0):
        return prevFit
    
    # compute the lighest shade of the line segment
    c = int(np.max(tgrey))
    
    # create new line segment
    draw = cpuDrawLine(line[0], line[1], line[2], line[3], draw, c, LINE_WIDTH)
    partgenImg = geimg[minX:minX+deltaX, minY:minY+deltaY]
    
    draw = np.minimum(partgenImg, draw)
    
    # substract similarity between previously generated image and target image + add newly computed part
    newSum = tSum - np.sum((orimg[minX:minX+deltaX, minY:minY+deltaY] - partgenImg)**2.0) + np.sum((orimg[minX:minX+deltaX, minY:minY+deltaY] - draw)**2.0)
    if (newSum < 0):
        return prevFit
    else:
        return np.sqrt(newSum/(geimg.size)) / 255.0