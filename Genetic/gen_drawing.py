from flask import Flask, request, jsonify
# import libraries
import time
import datetime
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import numba
#import cv2
import math
import genetic
# add a folder with a library to the path
sys.path.append(".")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "./genetic")
# import functions from the genetic library
from genetic.utils import *
from PIL import Image, ImageDraw, ImageOps
import configparser
import shutil
import random

from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

@app.route('/start', methods=['POST'])
def start_algorithm():
    # Tu by boli prípadné kroky na spracovanie vstupných dát z požiadavky
    # Napríklad: data = request.json
    data = request.json

# Získanie reťazca z dát (ak je definovaný)
    my_string = data.get("my_string", "Default String")
    #print("Received my_string:", my_string)
    # Spustiť váš algoritmus v samostatnom vlákne, aby nedošlo k blokovaniu
    thread = threading.Thread(target=run_algorithm, args=(my_string,))
    thread.start()

    return jsonify({"message": "Algorithm started"}), 202

def run_algorithm(my_string):
    print("Received my_string:", my_string)

    
    
    if not os.path.isfile('settings.cfg'):
        shutil.copy2('defcfg.cfg','settings.cfg')
    
    config = configparser.RawConfigParser()
    config.read('settings.cfg')
    
    # ------------------------------ USER PARAMETERS --------------------------
    img_str = my_string  # image to load (from ./images folder)
    basewidth = config.getint('settings', 'basewidth') # height depends on input image
    deterministic_mode = config.getboolean('settings', 'deterministic_mode')  # reproducible results [True, False]
    deterministic_seed = config.getint('settings', 'deterministic_seed') # seed for pseudo-random generator
    N = config.getint('settings', 'N') # number of objects in created image
    show_fit_graph = config.getboolean('settings', 'show_fit_graph')  # show fitness graph [True, False]
    
    
    # --- Video creation ---
    video_create = config.getboolean('video', 'video_create') # generate animation [True, False]
    video_length = config.getint('video', 'video_length') # video length in sec
    video_fps = config.getint('video', 'video_fps') # video frames per sec
    
    # --- Genetic Optimization ---
    NEvo = config.getint('genetic', 'NEvo') # number of evolution steps per one optimized object 
    MAX_BUFF = config.getint('genetic', 'MAX_BUFF')  # stopping evolution if there are no changes (MAX_BUFF consecutive evolution steps)
    MAX_ADDMUT = config.getint('genetic', 'MAX_ADDMUT') # [%] - maximum aditive mutation range
    MUT_RATE = config.getint('genetic', 'MUT_RATE') # [%] - mutation rate (percentage of chromosomes to be mutated)
    LINE_WIDTH = config.getint('genetic', 'LINE_WIDTH') # [px] line width
    MLTPL_EVO_PARAMS = config.getint('genetic', 'MLTPL_EVO_PARAMS') # parameter multiplier
    BLEND_MODE = config.get('genetic', 'BLEND_MODE') # available options: ["normal", "multiply", "screen", "overlay", "darken", "lighten", "color_dodge", "color_burn", "hard_light", "soft_light", "difference", "exclusion", "hue", "saturation", "color", "luminosity", "vivid_light", "pin_light", "linear_dodge", "subtract"]
    
    # --- Parallel optimalization parameter ---
    NUM_THREADS = 4 # number of active threads
    numba.set_num_threads(NUM_THREADS)
    # -------------------------------------------------------------------------
    
    """
     Optimized parameters: x1,x2,y1,y2 (for one line)
    
       -------> y
       | °°°°°°°°°°°°°°°   
       | °             °
       | °             °
       v °             °
       x °             °
         °°°°°°°°°°°°°°°
    
     x1,x2 <0, image_height> - vector of x positions
     y1,y2 <0, image_width> - vector of y positions
    """
    '''
    NOTE:
        - numpy -> works with the image as a dimensional tensor (H, W, D)
          [ the higher axis designation and the tensor correspond to this (x,y,d) ]
        - Pillow -> works with the image as a dimensional tensor (W, H, D)
          [ this results in a discrepancy and a change of labeling, x and y in code according to object type ]
    '''
    
    # if deterministic mode, use specified seed for reproducible results
    if (deterministic_mode):
        random.seed(deterministic_seed)
        np.random.seed(deterministic_seed)
        numbaseed(deterministic_seed)
        
    # rendering settings (font and style)
    plt.style.use('fast')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 0.1 # frame boundaries in graphs
    
    # load image and convert it to greyscale
    orig_img = Image.open("./images/" + img_str).convert('L')
    # resize image to specified width with aspect ratio preserved
    wpercent = (basewidth/float(orig_img.size[0]))
    hsize = int((float(orig_img.size[1])*float(wpercent)))
    orig_img = orig_img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
    # padding to counteract pillow bug
    orig_img = np.pad(orig_img, pad_width= LINE_WIDTH , mode='constant', constant_values=255)
    # convert to numpy array
    orig_img = np.asarray(orig_img, dtype=int)
    # start the timer
    start_time = time.time()
    # --------------------------------------------------------
    
    # generate empty image with background colour
    gen_img = Image.new('RGBA', (orig_img.shape[1], orig_img.shape[0]), COLOUR_WHITE)
    gen_img = gen_img.convert('L') # canvas
    #counteract pillow bug
    lin_trans = Image.new('RGBA', (orig_img.shape[1], orig_img.shape[0]), COLOUR_WHITE) # canvas
    lin_trans = gen_img.convert('L') # canvas
    lin_trans_draw = ImageDraw.Draw(lin_trans)
    row, colm = gen_img.size 
    # Compute first fitness
    rfit = np.sqrt(np.sum((orig_img - np.asarray(gen_img, dtype=int))**2.0)/(orig_img.size)) / 255.0
    
    # definition of search space limitations (for one line segment only)
    OneSpace = np.concatenate((np.zeros((1,4)),           # mininum
                            np.array([[orig_img.shape[0]-1, orig_img.shape[0]-1, orig_img.shape[1]-1, orig_img.shape[1]-1]])), axis=0)  # maximum
    # range of changes for the additive mutation
    Amp = OneSpace[1,:]*(MAX_ADDMUT/100.0) 
    # results to be saved
    lpoly = np.zeros((N,6)) # (x1,x2,y1,y2,stroke,fitness)
    data = list() # list of fitness values
    # we start from the white canvas to which we add line segments
    buffer = 0 # auxiliary variable to stop evolution if no changes occur
    count = 1 # number of objects in final image
    count_frm = 1 # number of objects in video
    uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    uniq_filename = uniq_filename[:-7] # create unique file name
    if video_create: # create video file
        gif_img = gen_img.crop((LINE_WIDTH, LINE_WIDTH, row - LINE_WIDTH, colm - LINE_WIDTH)) 
        width, height = gif_img.size
        video_name = u"./results/{}.mp4".format(img_str.rsplit('.', 1)[0] + '_' + uniq_filename)
        video = cv2.VideoWriter(video_name, 0, video_fps, (width,height))
        numpydata = np.asarray(gif_img)
        video.write(numpydata)
     
    # repeat, until we reached specified number of line segments
    while(count<=N):
        # initial population generation
        NewPop = genLinespop(24*MLTPL_EVO_PARAMS, Amp, OneSpace)
        # first fitness evaluation
        fitness = ParllelCPUEvalFitness(NewPop, orig_img, np.asarray(gen_img, dtype=np.uint8), rfit, LINE_WIDTH, BLEND_MODE)  
        buffer = 0
        i = 0
        # start of genetic optimization process
        for i in range(NEvo):  # high enough value (we expect an early stop)
            OldPop = np.copy(NewPop)    # save population and fitness from previous generation
            fitnessOld = np.copy(fitness) 
            PartNewPop1, PartNewFit1 = selbest(OldPop, fitness, [3*MLTPL_EVO_PARAMS,2*MLTPL_EVO_PARAMS,1*MLTPL_EVO_PARAMS])    # select best lines
            PartNewPop2, PartNewFit2 = selsus(OldPop, fitness, 18*MLTPL_EVO_PARAMS)
            PartNewPop2 = mutLine(PartNewPop2, MUT_RATE/100.0, Amp, OneSpace)   # additive mutation
            NewPop = np.concatenate((PartNewPop1, PartNewPop2), axis=0) # create new population
            fitness = ParllelCPUEvalFitness(NewPop, orig_img, np.asarray(gen_img, dtype=np.uint8), rfit, LINE_WIDTH, BLEND_MODE)
            if (np.min(fitness) == np.min(fitnessOld)):
                buffer += 1 # if we stagnate start with counting
            else: 
                buffer = 0  # if the solution has improved, continue evolution
            # if we have exceeded the maximum limit, we will stop evolution
            if (buffer >= MAX_BUFF):
                break
        #print(f"GA stopped in generation {i}")    
        # add the best line segment in the image and continue evolution
        psol, rfitnew = selbest(NewPop, fitness, [1])
        
        if(rfit is None):
            rfit = 1e6 # safe big value
        # draw line segment only if it improves fitness
        if(rfitnew < rfit):
            rfit = rfitnew[0]
            data.append(rfit) # save line segment info
            
            # evaluate only part of the image
            minX = int(np.min([psol[0,0],psol[0,1]])) - LINE_WIDTH
            maxX = int(np.max([psol[0,0],psol[0,1]])) + LINE_WIDTH
            deltaX = int(maxX - minX) + 1 
            
            minY = int(np.min([psol[0,2],psol[0,3]])) - LINE_WIDTH
            maxY = int(np.max([psol[0,2],psol[0,3]])) + LINE_WIDTH
            deltaY = int(maxY - minY) + 1 
            
            draw = np.ones(dtype=np.uint8, shape=(deltaX, deltaY)) * 255
            
            line = (int(psol[0,0]-minX), int(psol[0,2]-minY),int(psol[0,1]-minX), int(psol[0,3]-minY))
            mask_img = np.zeros(dtype=np.uint8, shape=draw.shape)
            mask_img = cpuDrawLine(line[0], line[1], line[2], line[3], mask_img, 1, LINE_WIDTH)
    
            tgrey = np.multiply(orig_img[minX:minX+deltaX, minY:minY+deltaY], mask_img)
            
            # compute the lighest shade of the line segment  
            c = int(np.max(tgrey))
            
            # create new line segment
            draw = cpuDrawLine(line[0], line[1], line[2], line[3], draw, c, LINE_WIDTH)
            geimg = np.copy(np.asarray(gen_img, dtype=np.uint8))
            partgenImg = geimg[minX:minX+deltaX, minY:minY+deltaY]
            
            draw = np.minimum(partgenImg, draw)        
            geimg[minX:minX+deltaX, minY:minY+deltaY] = draw
            
            # Fittness for debugging purposess
            # if count % 100 == 0:
            #     # fffit = np.sqrt(np.sum((orig_img - geimg)**2.0)/(orig_img.size)) / 255.0
            #     print("# " + str(count) + " Fitness of entire image: " + str(fffit))
            #     print(f"# {count} Fitness error is {fffit - rfit}")
                  
            gen_img = Image.fromarray(geimg)
            
            print("# " + str(count) + " Fitness: " + str(rfit))
            lpoly[count-1,:] = np.concatenate(np.array((int(psol[0,2] - LINE_WIDTH), int(psol[0,3] - LINE_WIDTH), int(psol[0,0] - LINE_WIDTH), int(psol[0,1] - LINE_WIDTH), 255-c, rfit)).reshape(6,1))      
            count += 1 # increment counter of drawn line segments
            if video_create and math.sqrt(count/(N/(video_length*video_fps)/(video_length*video_fps)))>count_frm: # add frame to video at predetermined places
                gif_img = gen_img.crop((LINE_WIDTH, LINE_WIDTH, row - LINE_WIDTH, colm - LINE_WIDTH)) 
                numpydata = np.asarray(gif_img)
                video.write(numpydata)
                count_frm += 1
        else:
            print("Fit not improved, restarting generation")  
            
    # cropping final image
    gen_img = gen_img.crop((LINE_WIDTH, LINE_WIDTH, row - LINE_WIDTH, colm - LINE_WIDTH)) 
    
    
    # find out the final solution
    sol, rfit = selbest(NewPop, fitness, [1])
    print("Final fitness value: " + str(rfit[0]))
    print("--- Evolution lasted: %s seconds ---" % (time.time() - start_time))
    
    # save generated images
    out_path = "./results/" + my_string
    gen_img.save(out_path, dpi=(600,600))
    # save solution info to csv file
    np.savetxt("./results/" + img_str.rsplit('.', 1)[0] + '_' + uniq_filename + ".csv", lpoly, delimiter=";")
    # save the animation 
    i = 0
    if video_create:
        for i in range(int(video_fps/5)):
            video.write(numpydata)
        cv2.destroyAllWindows()
        video.release()
        
# Spustenie Flask aplikácie
if __name__ == '__main__':
    app.run(debug=True)