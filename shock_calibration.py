'''
Day1 ExPress
Dual CS+ conditioning with Suppression facilitiated extinction
'''
import argparse
import os
import sys
from glob import glob
import numpy as np
import pandas as pd
from collections import OrderedDict
from psychopy import iohub, visual, core, gui, data, monitors, tools, parallel
from psychopy import event as Event
import pickle
import time

#########################
##  utility functions  ##
#########################
def check4quit():
    for kp in keyboard.getPresses(keys=QUIT_KEY,mods=[QUIT_MODIFIER]):
        io.quit();win.close();sys.exit()

# for simulated keypresses/responses
class sim_kp(object):
    def __init__(self):
        self.char = np.random.choice(RESP_KEYS)
        self.time = np.random.normal(1.2,.1) + core.getTime()

def prep_for_response():
    Event.clearEvents()
    return core.getTime(), None

def collect_response():
    for kp in keyboard.getPresses(keys=RESP_KEYS):
        return kp
    if SUBJ == 'sim':
        return sim_kp()

def shock():
    port.setPin(2,1)
    time.sleep(.05)
    port.setPin(2,0)



def stamp_next_flip(on_or_offset,stamp_idx):
    stamp_df.loc[stamp_idx,'true_{:s}'.format(on_or_offset)] = clock.getTime()

def stamp_onoffset(on_or_offset,stamp_idx,BIO=False,SHOCK=False):
    win.callOnFlip(stamp_next_flip,on_or_offset,stamp_idx)
    
    if SHOCK: shock()

    if BIO and 'stim' in stamp_idx[1]:
        if 'on' in on_or_offset:
            port.setPin(4,1)
        elif 'off' in on_or_offset:
            port.setPin(4,0)


SCREENS = {
    'skyrawide':   dict(distance_cm=136,width_cm=85.7,pixel_dims=[1920,1080]),
    'skyrashit':   dict(distance_cm=136,width_cm=38.0,pixel_dims=[1024, 768]),
    'skyrasmall':  dict(distance_cm=136,width_cm=45.0,pixel_dims=[1024, 768]),
    'gus':         dict(distance_cm=67,width_cm=59.3,pixel_dims=[2560,1440]),
    'home':        dict(distance_cm=100,width_cm=68.47,pixel_dims=[3840, 2160]),
    'factory'   :  dict(distance_cm=136,width_cm=85.7,pixel_dims=[1920,1080]) 
}


parser = argparse.ArgumentParser(description='day1 shock calibration')
parser.add_argument('-c','--scrn',default='factory',type=str,choices=list(SCREENS.keys()),help='screen used for display')

args = parser.parse_args()

#=================================================
# Set some stimulus parameters and other constants
#=================================================
PROJ_NAME = 'express'
SCRN = args.scrn
SCAN = 'skyra' in SCRN
BIO = 'factory' in SCRN
DATE = data.getDateStr()

#i/o
home = os.path.expanduser('~')

#initialize the parallel port if in the testing room
if BIO:
    port = parallel.ParallelPort(address=0xEFF8)
    port.setData(0)

# screen
FRAME_RATE = 60. # flips per second
BKGRND_COLOR = 'black'

# keyboard
QUIT_KEY = '='
QUIT_MODIFIER = 'lshift'
SCAN_TRIGGER = '5'
RESP_KEYS = ['1', '2']

# stims
TARG_RAD     = 5     # memory image targets

###########################
##  open/setup psychopy  ##
###########################

# keyboard and clock
io = iohub.launchHubServer(); io.clearEvents('all')
keyboard = io.devices.keyboard
clock = core.Clock()

# screen
mon = monitors.Monitor('testMonitor')
mon.setDistance(SCREENS[SCRN]['distance_cm'])
mon.setWidth(SCREENS[SCRN]['width_cm'])
mon.setSizePix(SCREENS[SCRN]['pixel_dims'])
win = visual.Window(monitor=mon,units='deg',fullscr=True,color=BKGRND_COLOR)
mouse = Event.Mouse(win=win,visible=False)
# check to make sure frame rate is as expected, since it's used above for flip buffer
actual_frame_rate = win.getActualFrameRate() 
if not (FRAME_RATE-2 < actual_frame_rate < FRAME_RATE+2):
    raise Warning('Frame rate ({}) is not as expected ({}).'.format(actual_frame_rate,FRAME_RATE))
flip_buffer = 1./FRAME_RATE


FIX_COLOR    = 'grey'
SUPP_COLOR   = 'blue'
#initialize the stimuli
txtStim   = visual.TextStim(win,pos=[0,0],height=.5) #,wrapWidth=req_dva_size)


#######################
##  event functions  ##
#######################

def instructions():
    msg1 = 'We will now calibrate the intensity level of the electrical pulse.\n\nThe pulse is very short, only 50 milliseconds'
    msg2 = 'For the purpose of this experiment, the electrical pulse is meant to be annoying.\n\nIt is meant to be something that you do not like, and would anticipate receiving.\n\nBut, it is not meant to be something that would make you say "ouch!".'
    msg3 = 'We will start the calibration off at a low level that you might not even feel.\n\nWe will then move it up in stages until it is something that you would describe as highly annoying and unpleasant, but not painful. (5 or 6 out of 10)'
    for msg in [msg1,msg2,msg3]:
        txtStim.text = msg
        Event.clearEvents()
        txtStim.draw()
        win.flip()
        Event.waitKeys(keyList=['space'])

    cont = True 
    while cont is True:
        txtStim.text = 'Press space to deliver a shock. Press q to quit.\n\nPlease rate each shock 1-10 using the scale.'
        Event.clearEvents()
        txtStim.draw()
        win.flip()
        kps = Event.getKeys(keyList=['q','space'])
        if 'q' in kps:
            cont = False
        elif 'space' in kps:
            shock() 
######################
##  run experiment  ##
######################
# break message
# if not DEV:
instructions()

win.flip()
win.close()