#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.10),
    on Fri Mar  3 17:13:46 2023
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

#  WHATTUP BOI ## 
# specify which of the three inner loops will run on each of
# the Two iterations of the outer loop:

### BEGING EXP ###
## LAYER 1 ## 
# Randomize MS-1
orders = [[1, 0, 0, 0], [0, 1, 0, 0 ], [0, 0, 1, 0 ], [0, 0, 0, 1]]
orders1 = [[1, 0, 0, 0], [0, 1, 0, 0 ], [0, 0, 1, 0 ], [0, 0, 0, 1]]
orders2 = [[0,4],[4,0]]

shuffle(orders)
shuffle(orders1)
shuffle(orders2)
print(orders)
print(orders1)
print(orders2)



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.10'
expName = 'ATM_main_5-19-21'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/isaacchristian/Desktop/Princeton/RESEARCH/META/psypy/a_current_versions/ATM_main_5-19-21_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1440, 900], fullscr=False, screen=0, 
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "Begin"
BeginClock = core.Clock()
begin_text = visual.TextBox2(
     win, text='Welcome to the Attention Monitoring Task!\n\nPress Button 1 to begin', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='begin_text',
     autoLog=True,
)
begin_exp = keyboard.Keyboard()

# Initialize components for Routine "Rest_instr"
Rest_instrClock = core.Clock()
Rest_title = visual.TextStim(win=win, name='Rest_title',
    text='Resting State Scan',
    font='Arial',
    pos=(0, .35), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
rest_body = visual.TextBox2(
     win, text='1. In this block we will collect 5 minutes of your brain at rest\n\n2. Keep your eyes open and centered on the crosshair for the duration of the scan', font='Arial',
     pos=(0, 0),     letterHeight=0.03,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='rest_body',
     autoLog=True,
)
rest_nxt = visual.TextStim(win=win, name='rest_nxt',
    text='Press button 1 when you are ready to begin',
    font='Arial',
    pos=(0, -.3), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
advance_0 = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "Rest_scan"
Rest_scanClock = core.Clock()
trial_crosshair_rest = visual.ShapeStim(
    win=win, name='trial_crosshair_rest', vertices='cross',
    size=(0.1, 0.1),
    ori=0, pos=(0, 0),
    lineWidth=.1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=.5, depth=0.0, interpolate=True)
skip1 = keyboard.Keyboard()

# Initialize components for Routine "finish_trial"
finish_trialClock = core.Clock()
break_txt = visual.TextBox2(
     win, text='Task Complete! Yay! :)\n\nYou may now take a break for 30 seconds.\n\nPress button 1 if you’d like to move on\n', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='break_txt',
     autoLog=True,
)
break_btn = keyboard.Keyboard()

# Initialize components for Routine "MC_self_instr"
MC_self_instrClock = core.Clock()
MC_self_title = visual.TextStim(win=win, name='MC_self_title',
    text='Self Breath Counting ',
    font='Arial',
    pos=(0, .35), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
MC_self_body = visual.TextBox2(
     win, text='1. Count your breaths, increasing your count at the end of each breath cycle (breathe in, breathe out, +1)\n\n2. On every FIFTH breath cycle, press button 1\n\n3. Then, RESET your count, and begin counting from 1-5 again\n\n4. Keep your eyes open and centered on the crosshair for the duration of the task\n', font='Arial',
     pos=(0, 0),     letterHeight=0.03,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='MC_self_body',
     autoLog=True,
)
MC_self_nxt = visual.TextStim(win=win, name='MC_self_nxt',
    text='Press button 1 when you are ready to begin',
    font='Arial',
    pos=(0, -.3), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
advance_1 = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "self"
selfClock = core.Clock()
self_btn = keyboard.Keyboard()
trial_crosshair = visual.ShapeStim(
    win=win, name='trial_crosshair', vertices='cross',
    size=(0.1, 0.1),
    ori=0, pos=(0, 0),
    lineWidth=.1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=.5, depth=-1.0, interpolate=True)
skip2 = keyboard.Keyboard()

# Initialize components for Routine "finish_trial"
finish_trialClock = core.Clock()
break_txt = visual.TextBox2(
     win, text='Task Complete! Yay! :)\n\nYou may now take a break for 30 seconds.\n\nPress button 1 if you’d like to move on\n', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='break_txt',
     autoLog=True,
)
break_btn = keyboard.Keyboard()

# Initialize components for Routine "MM_self_instr"
MM_self_instrClock = core.Clock()
MM_self_title = visual.TextStim(win=win, name='MM_self_title',
    text='Self Mind Monitor',
    font='Arial',
    pos=(0, .35), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
MM_self_body = visual.TextBox2(
     win, text='1. Maintain focus on your breath for the entire 5 minutes of the task\n\n2. When you notice that your attention is no longer on the breath, press button 1\n\n3. Once noted, guide your attention back to your breath, and resume focus on breathing in and breathing out\n\n4. Keep your eyes open and centered on the crosshair for the duration of the task\n\n', font='Arial',
     pos=(0, 0),     letterHeight=0.03,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='MM_self_body',
     autoLog=True,
)
MM_self_nxt = visual.TextStim(win=win, name='MM_self_nxt',
    text='Press button 1 when you are ready to begin',
    font='Arial',
    pos=(0, -.3), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
advance_2 = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "self"
selfClock = core.Clock()
self_btn = keyboard.Keyboard()
trial_crosshair = visual.ShapeStim(
    win=win, name='trial_crosshair', vertices='cross',
    size=(0.1, 0.1),
    ori=0, pos=(0, 0),
    lineWidth=.1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=.5, depth=-1.0, interpolate=True)
skip2 = keyboard.Keyboard()

# Initialize components for Routine "finish_trial"
finish_trialClock = core.Clock()
break_txt = visual.TextBox2(
     win, text='Task Complete! Yay! :)\n\nYou may now take a break for 30 seconds.\n\nPress button 1 if you’d like to move on\n', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='break_txt',
     autoLog=True,
)
break_btn = keyboard.Keyboard()

# Initialize components for Routine "MM_other_instr"
MM_other_instrClock = core.Clock()
MM_other_title = visual.TextStim(win=win, name='MM_other_title',
    text='Other Mind Monitoring',
    font='Arial',
    pos=(0, .35), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
MM_other_body = visual.TextBox2(
     win, text='1. An actor will be presented onscreen for the next 5 minutes\n\n2. Press button 1 when you believe the actor’s mind wandered from the task (their attention is no longer on breath)\n', font='Arial',
     pos=(0, 0),     letterHeight=0.03,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='MM_other_body',
     autoLog=True,
)
MM_other_nxt = visual.TextStim(win=win, name='MM_other_nxt',
    text='Press button 1 when you are ready to begin',
    font='Arial',
    pos=(0, -.3), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
advance3 = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "movie_josh"
movie_joshClock = core.Clock()
other_j_btn1 = keyboard.Keyboard()
skip3 = keyboard.Keyboard()

# Initialize components for Routine "finish_trial"
finish_trialClock = core.Clock()
break_txt = visual.TextBox2(
     win, text='Task Complete! Yay! :)\n\nYou may now take a break for 30 seconds.\n\nPress button 1 if you’d like to move on\n', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='break_txt',
     autoLog=True,
)
break_btn = keyboard.Keyboard()

# Initialize components for Routine "MC_other_instr"
MC_other_instrClock = core.Clock()
MC_other_title = visual.TextStim(win=win, name='MC_other_title',
    text='Other Breath Counting',
    font='Arial',
    pos=(0, .35), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
MC_other_body = visual.TextBox2(
     win, text='1. An actor will be presented onscreen for the next 5 minutes\n\n2. Count the actor’s breaths, increasing your count at the end of each breath cycle (breathe in, breathe out, +1)\n\n3. On the actors FIFTH breath, press button 1\n\n4. RESET your count, and begin counting from 1-5 again', font='Arial',
     pos=(0, 0),     letterHeight=0.03,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='MC_other_body',
     autoLog=True,
)
MC_other_nxt = visual.TextStim(win=win, name='MC_other_nxt',
    text='Press button 1 when you are ready to begin',
    font='Arial',
    pos=(0, -.3), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
advance4 = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "movie_josh"
movie_joshClock = core.Clock()
other_j_btn1 = keyboard.Keyboard()
skip3 = keyboard.Keyboard()

# Initialize components for Routine "finish_trial"
finish_trialClock = core.Clock()
break_txt = visual.TextBox2(
     win, text='Task Complete! Yay! :)\n\nYou may now take a break for 30 seconds.\n\nPress button 1 if you’d like to move on\n', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='break_txt',
     autoLog=True,
)
break_btn = keyboard.Keyboard()

# Initialize components for Routine "MM_self_instr"
MM_self_instrClock = core.Clock()
MM_self_title = visual.TextStim(win=win, name='MM_self_title',
    text='Self Mind Monitor',
    font='Arial',
    pos=(0, .35), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
MM_self_body = visual.TextBox2(
     win, text='1. Maintain focus on your breath for the entire 5 minutes of the task\n\n2. When you notice that your attention is no longer on the breath, press button 1\n\n3. Once noted, guide your attention back to your breath, and resume focus on breathing in and breathing out\n\n4. Keep your eyes open and centered on the crosshair for the duration of the task\n\n', font='Arial',
     pos=(0, 0),     letterHeight=0.03,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='MM_self_body',
     autoLog=True,
)
MM_self_nxt = visual.TextStim(win=win, name='MM_self_nxt',
    text='Press button 1 when you are ready to begin',
    font='Arial',
    pos=(0, -.3), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
advance_2 = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "self"
selfClock = core.Clock()
self_btn = keyboard.Keyboard()
trial_crosshair = visual.ShapeStim(
    win=win, name='trial_crosshair', vertices='cross',
    size=(0.1, 0.1),
    ori=0, pos=(0, 0),
    lineWidth=.1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=.5, depth=-1.0, interpolate=True)
skip2 = keyboard.Keyboard()

# Initialize components for Routine "finish_trial"
finish_trialClock = core.Clock()
break_txt = visual.TextBox2(
     win, text='Task Complete! Yay! :)\n\nYou may now take a break for 30 seconds.\n\nPress button 1 if you’d like to move on\n', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='break_txt',
     autoLog=True,
)
break_btn = keyboard.Keyboard()

# Initialize components for Routine "MC_self_instr"
MC_self_instrClock = core.Clock()
MC_self_title = visual.TextStim(win=win, name='MC_self_title',
    text='Self Breath Counting ',
    font='Arial',
    pos=(0, .35), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
MC_self_body = visual.TextBox2(
     win, text='1. Count your breaths, increasing your count at the end of each breath cycle (breathe in, breathe out, +1)\n\n2. On every FIFTH breath cycle, press button 1\n\n3. Then, RESET your count, and begin counting from 1-5 again\n\n4. Keep your eyes open and centered on the crosshair for the duration of the task\n', font='Arial',
     pos=(0, 0),     letterHeight=0.03,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='MC_self_body',
     autoLog=True,
)
MC_self_nxt = visual.TextStim(win=win, name='MC_self_nxt',
    text='Press button 1 when you are ready to begin',
    font='Arial',
    pos=(0, -.3), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
advance_1 = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "self"
selfClock = core.Clock()
self_btn = keyboard.Keyboard()
trial_crosshair = visual.ShapeStim(
    win=win, name='trial_crosshair', vertices='cross',
    size=(0.1, 0.1),
    ori=0, pos=(0, 0),
    lineWidth=.1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=.5, depth=-1.0, interpolate=True)
skip2 = keyboard.Keyboard()

# Initialize components for Routine "finish_trial"
finish_trialClock = core.Clock()
break_txt = visual.TextBox2(
     win, text='Task Complete! Yay! :)\n\nYou may now take a break for 30 seconds.\n\nPress button 1 if you’d like to move on\n', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='break_txt',
     autoLog=True,
)
break_btn = keyboard.Keyboard()

# Initialize components for Routine "MC_other_instr"
MC_other_instrClock = core.Clock()
MC_other_title = visual.TextStim(win=win, name='MC_other_title',
    text='Other Breath Counting',
    font='Arial',
    pos=(0, .35), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
MC_other_body = visual.TextBox2(
     win, text='1. An actor will be presented onscreen for the next 5 minutes\n\n2. Count the actor’s breaths, increasing your count at the end of each breath cycle (breathe in, breathe out, +1)\n\n3. On the actors FIFTH breath, press button 1\n\n4. RESET your count, and begin counting from 1-5 again', font='Arial',
     pos=(0, 0),     letterHeight=0.03,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='MC_other_body',
     autoLog=True,
)
MC_other_nxt = visual.TextStim(win=win, name='MC_other_nxt',
    text='Press button 1 when you are ready to begin',
    font='Arial',
    pos=(0, -.3), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
advance4 = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "movie_nick"
movie_nickClock = core.Clock()
other_n_btn = keyboard.Keyboard()
skip4 = keyboard.Keyboard()

# Initialize components for Routine "finish_trial"
finish_trialClock = core.Clock()
break_txt = visual.TextBox2(
     win, text='Task Complete! Yay! :)\n\nYou may now take a break for 30 seconds.\n\nPress button 1 if you’d like to move on\n', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='break_txt',
     autoLog=True,
)
break_btn = keyboard.Keyboard()

# Initialize components for Routine "MM_other_instr"
MM_other_instrClock = core.Clock()
MM_other_title = visual.TextStim(win=win, name='MM_other_title',
    text='Other Mind Monitoring',
    font='Arial',
    pos=(0, .35), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
MM_other_body = visual.TextBox2(
     win, text='1. An actor will be presented onscreen for the next 5 minutes\n\n2. Press button 1 when you believe the actor’s mind wandered from the task (their attention is no longer on breath)\n', font='Arial',
     pos=(0, 0),     letterHeight=0.03,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='MM_other_body',
     autoLog=True,
)
MM_other_nxt = visual.TextStim(win=win, name='MM_other_nxt',
    text='Press button 1 when you are ready to begin',
    font='Arial',
    pos=(0, -.3), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
advance3 = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "movie_nick"
movie_nickClock = core.Clock()
other_n_btn = keyboard.Keyboard()
skip4 = keyboard.Keyboard()

# Initialize components for Routine "finish_trial"
finish_trialClock = core.Clock()
break_txt = visual.TextBox2(
     win, text='Task Complete! Yay! :)\n\nYou may now take a break for 30 seconds.\n\nPress button 1 if you’d like to move on\n', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='break_txt',
     autoLog=True,
)
break_btn = keyboard.Keyboard()

# Initialize components for Routine "Rest_instr"
Rest_instrClock = core.Clock()
Rest_title = visual.TextStim(win=win, name='Rest_title',
    text='Resting State Scan',
    font='Arial',
    pos=(0, .35), height=0.07, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
rest_body = visual.TextBox2(
     win, text='1. In this block we will collect 5 minutes of your brain at rest\n\n2. Keep your eyes open and centered on the crosshair for the duration of the scan', font='Arial',
     pos=(0, 0),     letterHeight=0.03,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='rest_body',
     autoLog=True,
)
rest_nxt = visual.TextStim(win=win, name='rest_nxt',
    text='Press button 1 when you are ready to begin',
    font='Arial',
    pos=(0, -.3), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
advance_0 = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "TR"
TRClock = core.Clock()
wait_scan_txt = visual.TextStim(win=win, name='wait_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt2 = visual.TextStim(win=win, name='still_txt2',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
scan_inter = keyboard.Keyboard()

# Initialize components for Routine "scan_wait"
scan_waitClock = core.Clock()
begin_scan_txt = visual.TextStim(win=win, name='begin_scan_txt',
    text='Waiting for scanner …',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
still_txt = visual.TextStim(win=win, name='still_txt',
    text='** Try to remain as still as possible for the entire scan! Thank u :)',
    font='Arial',
    pos=(-.075, -.15), height=0.03, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
begin_scan_btn = keyboard.Keyboard()

# Initialize components for Routine "Rest_scan"
Rest_scanClock = core.Clock()
trial_crosshair_rest = visual.ShapeStim(
    win=win, name='trial_crosshair_rest', vertices='cross',
    size=(0.1, 0.1),
    ori=0, pos=(0, 0),
    lineWidth=.1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=.5, depth=0.0, interpolate=True)
skip1 = keyboard.Keyboard()

# Initialize components for Routine "End_txt"
End_txtClock = core.Clock()
end = visual.TextBox2(
     win, text='Congratulations!\n\nYou completed the task. Wait for instructions from the task operator.\n\nThanks for your participation :)', font='Arial',
     pos=(0, 0),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='end',
     autoLog=True,
)
end_press = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "Begin"-------
continueRoutine = True
# update component parameters for each repeat
begin_exp.keys = []
begin_exp.rt = []
_begin_exp_allKeys = []
# keep track of which components have finished
BeginComponents = [begin_text, begin_exp]
for thisComponent in BeginComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
BeginClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Begin"-------
while continueRoutine:
    # get current time
    t = BeginClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=BeginClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *begin_text* updates
    if begin_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        begin_text.frameNStart = frameN  # exact frame index
        begin_text.tStart = t  # local t and not account for scr refresh
        begin_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(begin_text, 'tStartRefresh')  # time at next scr refresh
        begin_text.setAutoDraw(True)
    
    # *begin_exp* updates
    waitOnFlip = False
    if begin_exp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        begin_exp.frameNStart = frameN  # exact frame index
        begin_exp.tStart = t  # local t and not account for scr refresh
        begin_exp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(begin_exp, 'tStartRefresh')  # time at next scr refresh
        begin_exp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(begin_exp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(begin_exp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if begin_exp.status == STARTED and not waitOnFlip:
        theseKeys = begin_exp.getKeys(keyList=['space', '1'], waitRelease=False)
        _begin_exp_allKeys.extend(theseKeys)
        if len(_begin_exp_allKeys):
            begin_exp.keys = _begin_exp_allKeys[-1].name  # just the last key pressed
            begin_exp.rt = _begin_exp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in BeginComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Begin"-------
for thisComponent in BeginComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('begin_text.started', begin_text.tStartRefresh)
thisExp.addData('begin_text.stopped', begin_text.tStopRefresh)
# check responses
if begin_exp.keys in ['', [], None]:  # No response was made
    begin_exp.keys = None
thisExp.addData('begin_exp.keys',begin_exp.keys)
if begin_exp.keys != None:  # we had a response
    thisExp.addData('begin_exp.rt', begin_exp.rt)
thisExp.addData('begin_exp.started', begin_exp.tStartRefresh)
thisExp.addData('begin_exp.stopped', begin_exp.tStopRefresh)
thisExp.nextEntry()
# the Routine "Begin" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "Rest_instr"-------
continueRoutine = True
# update component parameters for each repeat
advance_0.keys = []
advance_0.rt = []
_advance_0_allKeys = []
# keep track of which components have finished
Rest_instrComponents = [Rest_title, rest_body, rest_nxt, advance_0]
for thisComponent in Rest_instrComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Rest_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Rest_instr"-------
while continueRoutine:
    # get current time
    t = Rest_instrClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Rest_instrClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Rest_title* updates
    if Rest_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Rest_title.frameNStart = frameN  # exact frame index
        Rest_title.tStart = t  # local t and not account for scr refresh
        Rest_title.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Rest_title, 'tStartRefresh')  # time at next scr refresh
        Rest_title.setAutoDraw(True)
    
    # *rest_body* updates
    if rest_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        rest_body.frameNStart = frameN  # exact frame index
        rest_body.tStart = t  # local t and not account for scr refresh
        rest_body.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest_body, 'tStartRefresh')  # time at next scr refresh
        rest_body.setAutoDraw(True)
    
    # *rest_nxt* updates
    if rest_nxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        rest_nxt.frameNStart = frameN  # exact frame index
        rest_nxt.tStart = t  # local t and not account for scr refresh
        rest_nxt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest_nxt, 'tStartRefresh')  # time at next scr refresh
        rest_nxt.setAutoDraw(True)
    
    # *advance_0* updates
    waitOnFlip = False
    if advance_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        advance_0.frameNStart = frameN  # exact frame index
        advance_0.tStart = t  # local t and not account for scr refresh
        advance_0.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(advance_0, 'tStartRefresh')  # time at next scr refresh
        advance_0.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(advance_0.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(advance_0.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if advance_0.status == STARTED and not waitOnFlip:
        theseKeys = advance_0.getKeys(keyList=['space', '1'], waitRelease=False)
        _advance_0_allKeys.extend(theseKeys)
        if len(_advance_0_allKeys):
            advance_0.keys = _advance_0_allKeys[-1].name  # just the last key pressed
            advance_0.rt = _advance_0_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Rest_instrComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Rest_instr"-------
for thisComponent in Rest_instrComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Rest_title.started', Rest_title.tStartRefresh)
thisExp.addData('Rest_title.stopped', Rest_title.tStopRefresh)
thisExp.addData('rest_body.started', rest_body.tStartRefresh)
thisExp.addData('rest_body.stopped', rest_body.tStopRefresh)
thisExp.addData('rest_nxt.started', rest_nxt.tStartRefresh)
thisExp.addData('rest_nxt.stopped', rest_nxt.tStopRefresh)
# check responses
if advance_0.keys in ['', [], None]:  # No response was made
    advance_0.keys = None
thisExp.addData('advance_0.keys',advance_0.keys)
if advance_0.keys != None:  # we had a response
    thisExp.addData('advance_0.rt', advance_0.rt)
thisExp.addData('advance_0.started', advance_0.tStartRefresh)
thisExp.addData('advance_0.stopped', advance_0.tStopRefresh)
thisExp.nextEntry()
# the Routine "Rest_instr" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "scan_wait"-------
continueRoutine = True
# update component parameters for each repeat
begin_scan_btn.keys = []
begin_scan_btn.rt = []
_begin_scan_btn_allKeys = []
# keep track of which components have finished
scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
for thisComponent in scan_waitComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "scan_wait"-------
while continueRoutine:
    # get current time
    t = scan_waitClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *begin_scan_txt* updates
    if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        begin_scan_txt.frameNStart = frameN  # exact frame index
        begin_scan_txt.tStart = t  # local t and not account for scr refresh
        begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
        begin_scan_txt.setAutoDraw(True)
    
    # *still_txt* updates
    if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        still_txt.frameNStart = frameN  # exact frame index
        still_txt.tStart = t  # local t and not account for scr refresh
        still_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
        still_txt.setAutoDraw(True)
    
    # *begin_scan_btn* updates
    waitOnFlip = False
    if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        begin_scan_btn.frameNStart = frameN  # exact frame index
        begin_scan_btn.tStart = t  # local t and not account for scr refresh
        begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
        begin_scan_btn.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
    if begin_scan_btn.status == STARTED and not waitOnFlip:
        theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
        _begin_scan_btn_allKeys.extend(theseKeys)
        if len(_begin_scan_btn_allKeys):
            begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
            begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in scan_waitComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "scan_wait"-------
for thisComponent in scan_waitComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
thisExp.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
thisExp.addData('still_txt.started', still_txt.tStartRefresh)
thisExp.addData('still_txt.stopped', still_txt.tStopRefresh)
# check responses
if begin_scan_btn.keys in ['', [], None]:  # No response was made
    begin_scan_btn.keys = None
thisExp.addData('begin_scan_btn.keys',begin_scan_btn.keys)
if begin_scan_btn.keys != None:  # we had a response
    thisExp.addData('begin_scan_btn.rt', begin_scan_btn.rt)
thisExp.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
thisExp.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
thisExp.nextEntry()
# the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "TR"-------
continueRoutine = True
# update component parameters for each repeat
scan_inter.keys = []
scan_inter.rt = []
_scan_inter_allKeys = []
# keep track of which components have finished
TRComponents = [wait_scan_txt, still_txt2, scan_inter]
for thisComponent in TRComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "TR"-------
while continueRoutine:
    # get current time
    t = TRClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=TRClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *wait_scan_txt* updates
    if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        wait_scan_txt.frameNStart = frameN  # exact frame index
        wait_scan_txt.tStart = t  # local t and not account for scr refresh
        wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
        wait_scan_txt.setAutoDraw(True)
    
    # *still_txt2* updates
    if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        still_txt2.frameNStart = frameN  # exact frame index
        still_txt2.tStart = t  # local t and not account for scr refresh
        still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
        still_txt2.setAutoDraw(True)
    
    # *scan_inter* updates
    waitOnFlip = False
    if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        scan_inter.frameNStart = frameN  # exact frame index
        scan_inter.tStart = t  # local t and not account for scr refresh
        scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
        scan_inter.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if scan_inter.status == STARTED and not waitOnFlip:
        theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
        _scan_inter_allKeys.extend(theseKeys)
        if len(_scan_inter_allKeys):
            scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
            scan_inter.rt = _scan_inter_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in TRComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "TR"-------
for thisComponent in TRComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
thisExp.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
thisExp.addData('still_txt2.started', still_txt2.tStartRefresh)
thisExp.addData('still_txt2.stopped', still_txt2.tStopRefresh)
# the Routine "TR" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "TR"-------
continueRoutine = True
# update component parameters for each repeat
scan_inter.keys = []
scan_inter.rt = []
_scan_inter_allKeys = []
# keep track of which components have finished
TRComponents = [wait_scan_txt, still_txt2, scan_inter]
for thisComponent in TRComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "TR"-------
while continueRoutine:
    # get current time
    t = TRClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=TRClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *wait_scan_txt* updates
    if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        wait_scan_txt.frameNStart = frameN  # exact frame index
        wait_scan_txt.tStart = t  # local t and not account for scr refresh
        wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
        wait_scan_txt.setAutoDraw(True)
    
    # *still_txt2* updates
    if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        still_txt2.frameNStart = frameN  # exact frame index
        still_txt2.tStart = t  # local t and not account for scr refresh
        still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
        still_txt2.setAutoDraw(True)
    
    # *scan_inter* updates
    waitOnFlip = False
    if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        scan_inter.frameNStart = frameN  # exact frame index
        scan_inter.tStart = t  # local t and not account for scr refresh
        scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
        scan_inter.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if scan_inter.status == STARTED and not waitOnFlip:
        theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
        _scan_inter_allKeys.extend(theseKeys)
        if len(_scan_inter_allKeys):
            scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
            scan_inter.rt = _scan_inter_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in TRComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "TR"-------
for thisComponent in TRComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
thisExp.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
thisExp.addData('still_txt2.started', still_txt2.tStartRefresh)
thisExp.addData('still_txt2.stopped', still_txt2.tStopRefresh)
# the Routine "TR" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "TR"-------
continueRoutine = True
# update component parameters for each repeat
scan_inter.keys = []
scan_inter.rt = []
_scan_inter_allKeys = []
# keep track of which components have finished
TRComponents = [wait_scan_txt, still_txt2, scan_inter]
for thisComponent in TRComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "TR"-------
while continueRoutine:
    # get current time
    t = TRClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=TRClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *wait_scan_txt* updates
    if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        wait_scan_txt.frameNStart = frameN  # exact frame index
        wait_scan_txt.tStart = t  # local t and not account for scr refresh
        wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
        wait_scan_txt.setAutoDraw(True)
    
    # *still_txt2* updates
    if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        still_txt2.frameNStart = frameN  # exact frame index
        still_txt2.tStart = t  # local t and not account for scr refresh
        still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
        still_txt2.setAutoDraw(True)
    
    # *scan_inter* updates
    waitOnFlip = False
    if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        scan_inter.frameNStart = frameN  # exact frame index
        scan_inter.tStart = t  # local t and not account for scr refresh
        scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
        scan_inter.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if scan_inter.status == STARTED and not waitOnFlip:
        theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
        _scan_inter_allKeys.extend(theseKeys)
        if len(_scan_inter_allKeys):
            scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
            scan_inter.rt = _scan_inter_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in TRComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "TR"-------
for thisComponent in TRComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
thisExp.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
thisExp.addData('still_txt2.started', still_txt2.tStartRefresh)
thisExp.addData('still_txt2.stopped', still_txt2.tStopRefresh)
# the Routine "TR" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "scan_wait"-------
continueRoutine = True
# update component parameters for each repeat
begin_scan_btn.keys = []
begin_scan_btn.rt = []
_begin_scan_btn_allKeys = []
# keep track of which components have finished
scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
for thisComponent in scan_waitComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "scan_wait"-------
while continueRoutine:
    # get current time
    t = scan_waitClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *begin_scan_txt* updates
    if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        begin_scan_txt.frameNStart = frameN  # exact frame index
        begin_scan_txt.tStart = t  # local t and not account for scr refresh
        begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
        begin_scan_txt.setAutoDraw(True)
    
    # *still_txt* updates
    if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        still_txt.frameNStart = frameN  # exact frame index
        still_txt.tStart = t  # local t and not account for scr refresh
        still_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
        still_txt.setAutoDraw(True)
    
    # *begin_scan_btn* updates
    waitOnFlip = False
    if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        begin_scan_btn.frameNStart = frameN  # exact frame index
        begin_scan_btn.tStart = t  # local t and not account for scr refresh
        begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
        begin_scan_btn.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
    if begin_scan_btn.status == STARTED and not waitOnFlip:
        theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
        _begin_scan_btn_allKeys.extend(theseKeys)
        if len(_begin_scan_btn_allKeys):
            begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
            begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in scan_waitComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "scan_wait"-------
for thisComponent in scan_waitComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
thisExp.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
thisExp.addData('still_txt.started', still_txt.tStartRefresh)
thisExp.addData('still_txt.stopped', still_txt.tStopRefresh)
# check responses
if begin_scan_btn.keys in ['', [], None]:  # No response was made
    begin_scan_btn.keys = None
thisExp.addData('begin_scan_btn.keys',begin_scan_btn.keys)
if begin_scan_btn.keys != None:  # we had a response
    thisExp.addData('begin_scan_btn.rt', begin_scan_btn.rt)
thisExp.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
thisExp.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
thisExp.nextEntry()
# the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "Rest_scan"-------
continueRoutine = True
routineTimer.add(307.500000)
# update component parameters for each repeat
skip1.keys = []
skip1.rt = []
_skip1_allKeys = []
# keep track of which components have finished
Rest_scanComponents = [trial_crosshair_rest, skip1]
for thisComponent in Rest_scanComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Rest_scanClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Rest_scan"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = Rest_scanClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Rest_scanClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *trial_crosshair_rest* updates
    if trial_crosshair_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        trial_crosshair_rest.frameNStart = frameN  # exact frame index
        trial_crosshair_rest.tStart = t  # local t and not account for scr refresh
        trial_crosshair_rest.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(trial_crosshair_rest, 'tStartRefresh')  # time at next scr refresh
        trial_crosshair_rest.setAutoDraw(True)
    if trial_crosshair_rest.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > trial_crosshair_rest.tStartRefresh + 307.5-frameTolerance:
            # keep track of stop time/frame for later
            trial_crosshair_rest.tStop = t  # not accounting for scr refresh
            trial_crosshair_rest.frameNStop = frameN  # exact frame index
            win.timeOnFlip(trial_crosshair_rest, 'tStopRefresh')  # time at next scr refresh
            trial_crosshair_rest.setAutoDraw(False)
    
    # *skip1* updates
    waitOnFlip = False
    if skip1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        skip1.frameNStart = frameN  # exact frame index
        skip1.tStart = t  # local t and not account for scr refresh
        skip1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(skip1, 'tStartRefresh')  # time at next scr refresh
        skip1.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(skip1.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(skip1.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if skip1.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > skip1.tStartRefresh + 307.5-frameTolerance:
            # keep track of stop time/frame for later
            skip1.tStop = t  # not accounting for scr refresh
            skip1.frameNStop = frameN  # exact frame index
            win.timeOnFlip(skip1, 'tStopRefresh')  # time at next scr refresh
            skip1.status = FINISHED
    if skip1.status == STARTED and not waitOnFlip:
        theseKeys = skip1.getKeys(keyList=['s'], waitRelease=False)
        _skip1_allKeys.extend(theseKeys)
        if len(_skip1_allKeys):
            skip1.keys = _skip1_allKeys[-1].name  # just the last key pressed
            skip1.rt = _skip1_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Rest_scanComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Rest_scan"-------
for thisComponent in Rest_scanComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('trial_crosshair_rest.started', trial_crosshair_rest.tStartRefresh)
thisExp.addData('trial_crosshair_rest.stopped', trial_crosshair_rest.tStopRefresh)

# ------Prepare to start Routine "finish_trial"-------
continueRoutine = True
routineTimer.add(30.000000)
# update component parameters for each repeat
break_btn.keys = []
break_btn.rt = []
_break_btn_allKeys = []
# keep track of which components have finished
finish_trialComponents = [break_txt, break_btn]
for thisComponent in finish_trialComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
finish_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "finish_trial"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = finish_trialClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=finish_trialClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *break_txt* updates
    if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        break_txt.frameNStart = frameN  # exact frame index
        break_txt.tStart = t  # local t and not account for scr refresh
        break_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
        break_txt.setAutoDraw(True)
    if break_txt.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > break_txt.tStartRefresh + 30-frameTolerance:
            # keep track of stop time/frame for later
            break_txt.tStop = t  # not accounting for scr refresh
            break_txt.frameNStop = frameN  # exact frame index
            win.timeOnFlip(break_txt, 'tStopRefresh')  # time at next scr refresh
            break_txt.setAutoDraw(False)
    
    # *break_btn* updates
    waitOnFlip = False
    if break_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        break_btn.frameNStart = frameN  # exact frame index
        break_btn.tStart = t  # local t and not account for scr refresh
        break_btn.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(break_btn, 'tStartRefresh')  # time at next scr refresh
        break_btn.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(break_btn.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(break_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if break_btn.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > break_btn.tStartRefresh + 30-frameTolerance:
            # keep track of stop time/frame for later
            break_btn.tStop = t  # not accounting for scr refresh
            break_btn.frameNStop = frameN  # exact frame index
            win.timeOnFlip(break_btn, 'tStopRefresh')  # time at next scr refresh
            break_btn.status = FINISHED
    if break_btn.status == STARTED and not waitOnFlip:
        theseKeys = break_btn.getKeys(keyList=['space', '1'], waitRelease=False)
        _break_btn_allKeys.extend(theseKeys)
        if len(_break_btn_allKeys):
            break_btn.keys = _break_btn_allKeys[-1].name  # just the last key pressed
            break_btn.rt = _break_btn_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in finish_trialComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "finish_trial"-------
for thisComponent in finish_trialComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('break_txt.started', break_txt.tStartRefresh)
thisExp.addData('break_txt.stopped', break_txt.tStopRefresh)
# check responses
if break_btn.keys in ['', [], None]:  # No response was made
    break_btn.keys = None
thisExp.addData('break_btn.keys',break_btn.keys)
if break_btn.keys != None:  # we had a response
    thisExp.addData('break_btn.rt', break_btn.rt)
thisExp.addData('break_btn.started', break_btn.tStartRefresh)
thisExp.addData('break_btn.stopped', break_btn.tStopRefresh)
thisExp.nextEntry()

# set up handler to look after randomisation of conditions etc
big_guy = data.TrialHandler(nReps=2, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='big_guy')
thisExp.addLoop(big_guy)  # add the loop to the experiment
thisBig_guy = big_guy.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBig_guy.rgb)
if thisBig_guy != None:
    for paramName in thisBig_guy:
        exec('{} = thisBig_guy[paramName]'.format(paramName))

for thisBig_guy in big_guy:
    currentLoop = big_guy
    # abbreviate parameter names if possible (e.g. rgb = thisBig_guy.rgb)
    if thisBig_guy != None:
        for paramName in thisBig_guy:
            exec('{} = thisBig_guy[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    outer_loop1 = data.TrialHandler(nReps=orders2[0][big_guy.thisN], method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='outer_loop1')
    thisExp.addLoop(outer_loop1)  # add the loop to the experiment
    thisOuter_loop1 = outer_loop1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisOuter_loop1.rgb)
    if thisOuter_loop1 != None:
        for paramName in thisOuter_loop1:
            exec('{} = thisOuter_loop1[paramName]'.format(paramName))
    
    for thisOuter_loop1 in outer_loop1:
        currentLoop = outer_loop1
        # abbreviate parameter names if possible (e.g. rgb = thisOuter_loop1.rgb)
        if thisOuter_loop1 != None:
            for paramName in thisOuter_loop1:
                exec('{} = thisOuter_loop1[paramName]'.format(paramName))
        
        # set up handler to look after randomisation of conditions etc
        MC_self_loop1 = data.TrialHandler(nReps=orders[0][outer_loop1.thisN], method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='MC_self_loop1')
        thisExp.addLoop(MC_self_loop1)  # add the loop to the experiment
        thisMC_self_loop1 = MC_self_loop1.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMC_self_loop1.rgb)
        if thisMC_self_loop1 != None:
            for paramName in thisMC_self_loop1:
                exec('{} = thisMC_self_loop1[paramName]'.format(paramName))
        
        for thisMC_self_loop1 in MC_self_loop1:
            currentLoop = MC_self_loop1
            # abbreviate parameter names if possible (e.g. rgb = thisMC_self_loop1.rgb)
            if thisMC_self_loop1 != None:
                for paramName in thisMC_self_loop1:
                    exec('{} = thisMC_self_loop1[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "MC_self_instr"-------
            continueRoutine = True
            # update component parameters for each repeat
            advance_1.keys = []
            advance_1.rt = []
            _advance_1_allKeys = []
            # keep track of which components have finished
            MC_self_instrComponents = [MC_self_title, MC_self_body, MC_self_nxt, advance_1]
            for thisComponent in MC_self_instrComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            MC_self_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "MC_self_instr"-------
            while continueRoutine:
                # get current time
                t = MC_self_instrClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=MC_self_instrClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *MC_self_title* updates
                if MC_self_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_self_title.frameNStart = frameN  # exact frame index
                    MC_self_title.tStart = t  # local t and not account for scr refresh
                    MC_self_title.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_self_title, 'tStartRefresh')  # time at next scr refresh
                    MC_self_title.setAutoDraw(True)
                
                # *MC_self_body* updates
                if MC_self_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_self_body.frameNStart = frameN  # exact frame index
                    MC_self_body.tStart = t  # local t and not account for scr refresh
                    MC_self_body.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_self_body, 'tStartRefresh')  # time at next scr refresh
                    MC_self_body.setAutoDraw(True)
                
                # *MC_self_nxt* updates
                if MC_self_nxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_self_nxt.frameNStart = frameN  # exact frame index
                    MC_self_nxt.tStart = t  # local t and not account for scr refresh
                    MC_self_nxt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_self_nxt, 'tStartRefresh')  # time at next scr refresh
                    MC_self_nxt.setAutoDraw(True)
                
                # *advance_1* updates
                waitOnFlip = False
                if advance_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    advance_1.frameNStart = frameN  # exact frame index
                    advance_1.tStart = t  # local t and not account for scr refresh
                    advance_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(advance_1, 'tStartRefresh')  # time at next scr refresh
                    advance_1.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(advance_1.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(advance_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if advance_1.status == STARTED and not waitOnFlip:
                    theseKeys = advance_1.getKeys(keyList=['space', '1'], waitRelease=False)
                    _advance_1_allKeys.extend(theseKeys)
                    if len(_advance_1_allKeys):
                        advance_1.keys = _advance_1_allKeys[-1].name  # just the last key pressed
                        advance_1.rt = _advance_1_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in MC_self_instrComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "MC_self_instr"-------
            for thisComponent in MC_self_instrComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop1.addData('MC_self_title.started', MC_self_title.tStartRefresh)
            MC_self_loop1.addData('MC_self_title.stopped', MC_self_title.tStopRefresh)
            MC_self_loop1.addData('MC_self_body.started', MC_self_body.tStartRefresh)
            MC_self_loop1.addData('MC_self_body.stopped', MC_self_body.tStopRefresh)
            MC_self_loop1.addData('MC_self_nxt.started', MC_self_nxt.tStartRefresh)
            MC_self_loop1.addData('MC_self_nxt.stopped', MC_self_nxt.tStopRefresh)
            # check responses
            if advance_1.keys in ['', [], None]:  # No response was made
                advance_1.keys = None
            MC_self_loop1.addData('advance_1.keys',advance_1.keys)
            if advance_1.keys != None:  # we had a response
                MC_self_loop1.addData('advance_1.rt', advance_1.rt)
            MC_self_loop1.addData('advance_1.started', advance_1.tStartRefresh)
            MC_self_loop1.addData('advance_1.stopped', advance_1.tStopRefresh)
            # the Routine "MC_self_instr" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_self_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_self_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_self_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_self_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_self_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_self_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_self_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_self_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_self_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_self_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_self_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_self_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "scan_wait"-------
            continueRoutine = True
            # update component parameters for each repeat
            begin_scan_btn.keys = []
            begin_scan_btn.rt = []
            _begin_scan_btn_allKeys = []
            # keep track of which components have finished
            scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
            for thisComponent in scan_waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "scan_wait"-------
            while continueRoutine:
                # get current time
                t = scan_waitClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *begin_scan_txt* updates
                if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_txt.frameNStart = frameN  # exact frame index
                    begin_scan_txt.tStart = t  # local t and not account for scr refresh
                    begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_txt.setAutoDraw(True)
                
                # *still_txt* updates
                if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt.frameNStart = frameN  # exact frame index
                    still_txt.tStart = t  # local t and not account for scr refresh
                    still_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
                    still_txt.setAutoDraw(True)
                
                # *begin_scan_btn* updates
                waitOnFlip = False
                if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_btn.frameNStart = frameN  # exact frame index
                    begin_scan_btn.tStart = t  # local t and not account for scr refresh
                    begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
                if begin_scan_btn.status == STARTED and not waitOnFlip:
                    theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
                    _begin_scan_btn_allKeys.extend(theseKeys)
                    if len(_begin_scan_btn_allKeys):
                        begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
                        begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in scan_waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "scan_wait"-------
            for thisComponent in scan_waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop1.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
            MC_self_loop1.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
            MC_self_loop1.addData('still_txt.started', still_txt.tStartRefresh)
            MC_self_loop1.addData('still_txt.stopped', still_txt.tStopRefresh)
            # check responses
            if begin_scan_btn.keys in ['', [], None]:  # No response was made
                begin_scan_btn.keys = None
            MC_self_loop1.addData('begin_scan_btn.keys',begin_scan_btn.keys)
            if begin_scan_btn.keys != None:  # we had a response
                MC_self_loop1.addData('begin_scan_btn.rt', begin_scan_btn.rt)
            MC_self_loop1.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
            MC_self_loop1.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
            # the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "self"-------
            continueRoutine = True
            routineTimer.add(307.500000)
            # update component parameters for each repeat
            self_btn.keys = []
            self_btn.rt = []
            _self_btn_allKeys = []
            skip2.keys = []
            skip2.rt = []
            _skip2_allKeys = []
            # keep track of which components have finished
            selfComponents = [self_btn, trial_crosshair, skip2]
            for thisComponent in selfComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            selfClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "self"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = selfClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=selfClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *self_btn* updates
                waitOnFlip = False
                if self_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    self_btn.frameNStart = frameN  # exact frame index
                    self_btn.tStart = t  # local t and not account for scr refresh
                    self_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(self_btn, 'tStartRefresh')  # time at next scr refresh
                    self_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(self_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(self_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if self_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > self_btn.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        self_btn.tStop = t  # not accounting for scr refresh
                        self_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(self_btn, 'tStopRefresh')  # time at next scr refresh
                        self_btn.status = FINISHED
                if self_btn.status == STARTED and not waitOnFlip:
                    theseKeys = self_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _self_btn_allKeys.extend(theseKeys)
                    if len(_self_btn_allKeys):
                        self_btn.keys = [key.name for key in _self_btn_allKeys]  # storing all keys
                        self_btn.rt = [key.rt for key in _self_btn_allKeys]
                
                # *trial_crosshair* updates
                if trial_crosshair.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_crosshair.frameNStart = frameN  # exact frame index
                    trial_crosshair.tStart = t  # local t and not account for scr refresh
                    trial_crosshair.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_crosshair, 'tStartRefresh')  # time at next scr refresh
                    trial_crosshair.setAutoDraw(True)
                if trial_crosshair.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_crosshair.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_crosshair.tStop = t  # not accounting for scr refresh
                        trial_crosshair.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(trial_crosshair, 'tStopRefresh')  # time at next scr refresh
                        trial_crosshair.setAutoDraw(False)
                
                # *skip2* updates
                waitOnFlip = False
                if skip2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    skip2.frameNStart = frameN  # exact frame index
                    skip2.tStart = t  # local t and not account for scr refresh
                    skip2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(skip2, 'tStartRefresh')  # time at next scr refresh
                    skip2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(skip2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(skip2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if skip2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > skip2.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        skip2.tStop = t  # not accounting for scr refresh
                        skip2.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(skip2, 'tStopRefresh')  # time at next scr refresh
                        skip2.status = FINISHED
                if skip2.status == STARTED and not waitOnFlip:
                    theseKeys = skip2.getKeys(keyList=['s'], waitRelease=False)
                    _skip2_allKeys.extend(theseKeys)
                    if len(_skip2_allKeys):
                        skip2.keys = _skip2_allKeys[-1].name  # just the last key pressed
                        skip2.rt = _skip2_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in selfComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "self"-------
            for thisComponent in selfComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # check responses
            if self_btn.keys in ['', [], None]:  # No response was made
                self_btn.keys = None
            MC_self_loop1.addData('self_btn.keys',self_btn.keys)
            if self_btn.keys != None:  # we had a response
                MC_self_loop1.addData('self_btn.rt', self_btn.rt)
            MC_self_loop1.addData('self_btn.started', self_btn.tStartRefresh)
            MC_self_loop1.addData('self_btn.stopped', self_btn.tStopRefresh)
            MC_self_loop1.addData('trial_crosshair.started', trial_crosshair.tStartRefresh)
            MC_self_loop1.addData('trial_crosshair.stopped', trial_crosshair.tStopRefresh)
            
            # ------Prepare to start Routine "finish_trial"-------
            continueRoutine = True
            routineTimer.add(30.000000)
            # update component parameters for each repeat
            break_btn.keys = []
            break_btn.rt = []
            _break_btn_allKeys = []
            # keep track of which components have finished
            finish_trialComponents = [break_txt, break_btn]
            for thisComponent in finish_trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            finish_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "finish_trial"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = finish_trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=finish_trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *break_txt* updates
                if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_txt.frameNStart = frameN  # exact frame index
                    break_txt.tStart = t  # local t and not account for scr refresh
                    break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                    break_txt.setAutoDraw(True)
                if break_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_txt.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_txt.tStop = t  # not accounting for scr refresh
                        break_txt.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_txt, 'tStopRefresh')  # time at next scr refresh
                        break_txt.setAutoDraw(False)
                
                # *break_btn* updates
                waitOnFlip = False
                if break_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_btn.frameNStart = frameN  # exact frame index
                    break_btn.tStart = t  # local t and not account for scr refresh
                    break_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_btn, 'tStartRefresh')  # time at next scr refresh
                    break_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(break_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(break_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if break_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_btn.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_btn.tStop = t  # not accounting for scr refresh
                        break_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_btn, 'tStopRefresh')  # time at next scr refresh
                        break_btn.status = FINISHED
                if break_btn.status == STARTED and not waitOnFlip:
                    theseKeys = break_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _break_btn_allKeys.extend(theseKeys)
                    if len(_break_btn_allKeys):
                        break_btn.keys = _break_btn_allKeys[-1].name  # just the last key pressed
                        break_btn.rt = _break_btn_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in finish_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "finish_trial"-------
            for thisComponent in finish_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop1.addData('break_txt.started', break_txt.tStartRefresh)
            MC_self_loop1.addData('break_txt.stopped', break_txt.tStopRefresh)
            # check responses
            if break_btn.keys in ['', [], None]:  # No response was made
                break_btn.keys = None
            MC_self_loop1.addData('break_btn.keys',break_btn.keys)
            if break_btn.keys != None:  # we had a response
                MC_self_loop1.addData('break_btn.rt', break_btn.rt)
            MC_self_loop1.addData('break_btn.started', break_btn.tStartRefresh)
            MC_self_loop1.addData('break_btn.stopped', break_btn.tStopRefresh)
            thisExp.nextEntry()
            
        # completed orders[0][outer_loop1.thisN] repeats of 'MC_self_loop1'
        
        
        # set up handler to look after randomisation of conditions etc
        MM_self_loop1 = data.TrialHandler(nReps=orders[1][outer_loop1.thisN], method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='MM_self_loop1')
        thisExp.addLoop(MM_self_loop1)  # add the loop to the experiment
        thisMM_self_loop1 = MM_self_loop1.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMM_self_loop1.rgb)
        if thisMM_self_loop1 != None:
            for paramName in thisMM_self_loop1:
                exec('{} = thisMM_self_loop1[paramName]'.format(paramName))
        
        for thisMM_self_loop1 in MM_self_loop1:
            currentLoop = MM_self_loop1
            # abbreviate parameter names if possible (e.g. rgb = thisMM_self_loop1.rgb)
            if thisMM_self_loop1 != None:
                for paramName in thisMM_self_loop1:
                    exec('{} = thisMM_self_loop1[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "MM_self_instr"-------
            continueRoutine = True
            # update component parameters for each repeat
            advance_2.keys = []
            advance_2.rt = []
            _advance_2_allKeys = []
            # keep track of which components have finished
            MM_self_instrComponents = [MM_self_title, MM_self_body, MM_self_nxt, advance_2]
            for thisComponent in MM_self_instrComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            MM_self_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "MM_self_instr"-------
            while continueRoutine:
                # get current time
                t = MM_self_instrClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=MM_self_instrClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *MM_self_title* updates
                if MM_self_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_self_title.frameNStart = frameN  # exact frame index
                    MM_self_title.tStart = t  # local t and not account for scr refresh
                    MM_self_title.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_self_title, 'tStartRefresh')  # time at next scr refresh
                    MM_self_title.setAutoDraw(True)
                
                # *MM_self_body* updates
                if MM_self_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_self_body.frameNStart = frameN  # exact frame index
                    MM_self_body.tStart = t  # local t and not account for scr refresh
                    MM_self_body.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_self_body, 'tStartRefresh')  # time at next scr refresh
                    MM_self_body.setAutoDraw(True)
                
                # *MM_self_nxt* updates
                if MM_self_nxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_self_nxt.frameNStart = frameN  # exact frame index
                    MM_self_nxt.tStart = t  # local t and not account for scr refresh
                    MM_self_nxt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_self_nxt, 'tStartRefresh')  # time at next scr refresh
                    MM_self_nxt.setAutoDraw(True)
                
                # *advance_2* updates
                waitOnFlip = False
                if advance_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    advance_2.frameNStart = frameN  # exact frame index
                    advance_2.tStart = t  # local t and not account for scr refresh
                    advance_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(advance_2, 'tStartRefresh')  # time at next scr refresh
                    advance_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(advance_2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(advance_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if advance_2.status == STARTED and not waitOnFlip:
                    theseKeys = advance_2.getKeys(keyList=['space', '1'], waitRelease=False)
                    _advance_2_allKeys.extend(theseKeys)
                    if len(_advance_2_allKeys):
                        advance_2.keys = _advance_2_allKeys[-1].name  # just the last key pressed
                        advance_2.rt = _advance_2_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in MM_self_instrComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "MM_self_instr"-------
            for thisComponent in MM_self_instrComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop1.addData('MM_self_title.started', MM_self_title.tStartRefresh)
            MM_self_loop1.addData('MM_self_title.stopped', MM_self_title.tStopRefresh)
            MM_self_loop1.addData('MM_self_body.started', MM_self_body.tStartRefresh)
            MM_self_loop1.addData('MM_self_body.stopped', MM_self_body.tStopRefresh)
            MM_self_loop1.addData('MM_self_nxt.started', MM_self_nxt.tStartRefresh)
            MM_self_loop1.addData('MM_self_nxt.stopped', MM_self_nxt.tStopRefresh)
            # check responses
            if advance_2.keys in ['', [], None]:  # No response was made
                advance_2.keys = None
            MM_self_loop1.addData('advance_2.keys',advance_2.keys)
            if advance_2.keys != None:  # we had a response
                MM_self_loop1.addData('advance_2.rt', advance_2.rt)
            MM_self_loop1.addData('advance_2.started', advance_2.tStartRefresh)
            MM_self_loop1.addData('advance_2.stopped', advance_2.tStopRefresh)
            # the Routine "MM_self_instr" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_self_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_self_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_self_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_self_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_self_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_self_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_self_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_self_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_self_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_self_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_self_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_self_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "scan_wait"-------
            continueRoutine = True
            # update component parameters for each repeat
            begin_scan_btn.keys = []
            begin_scan_btn.rt = []
            _begin_scan_btn_allKeys = []
            # keep track of which components have finished
            scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
            for thisComponent in scan_waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "scan_wait"-------
            while continueRoutine:
                # get current time
                t = scan_waitClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *begin_scan_txt* updates
                if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_txt.frameNStart = frameN  # exact frame index
                    begin_scan_txt.tStart = t  # local t and not account for scr refresh
                    begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_txt.setAutoDraw(True)
                
                # *still_txt* updates
                if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt.frameNStart = frameN  # exact frame index
                    still_txt.tStart = t  # local t and not account for scr refresh
                    still_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
                    still_txt.setAutoDraw(True)
                
                # *begin_scan_btn* updates
                waitOnFlip = False
                if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_btn.frameNStart = frameN  # exact frame index
                    begin_scan_btn.tStart = t  # local t and not account for scr refresh
                    begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
                if begin_scan_btn.status == STARTED and not waitOnFlip:
                    theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
                    _begin_scan_btn_allKeys.extend(theseKeys)
                    if len(_begin_scan_btn_allKeys):
                        begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
                        begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in scan_waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "scan_wait"-------
            for thisComponent in scan_waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop1.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
            MM_self_loop1.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
            MM_self_loop1.addData('still_txt.started', still_txt.tStartRefresh)
            MM_self_loop1.addData('still_txt.stopped', still_txt.tStopRefresh)
            # check responses
            if begin_scan_btn.keys in ['', [], None]:  # No response was made
                begin_scan_btn.keys = None
            MM_self_loop1.addData('begin_scan_btn.keys',begin_scan_btn.keys)
            if begin_scan_btn.keys != None:  # we had a response
                MM_self_loop1.addData('begin_scan_btn.rt', begin_scan_btn.rt)
            MM_self_loop1.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
            MM_self_loop1.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
            # the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "self"-------
            continueRoutine = True
            routineTimer.add(307.500000)
            # update component parameters for each repeat
            self_btn.keys = []
            self_btn.rt = []
            _self_btn_allKeys = []
            skip2.keys = []
            skip2.rt = []
            _skip2_allKeys = []
            # keep track of which components have finished
            selfComponents = [self_btn, trial_crosshair, skip2]
            for thisComponent in selfComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            selfClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "self"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = selfClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=selfClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *self_btn* updates
                waitOnFlip = False
                if self_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    self_btn.frameNStart = frameN  # exact frame index
                    self_btn.tStart = t  # local t and not account for scr refresh
                    self_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(self_btn, 'tStartRefresh')  # time at next scr refresh
                    self_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(self_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(self_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if self_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > self_btn.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        self_btn.tStop = t  # not accounting for scr refresh
                        self_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(self_btn, 'tStopRefresh')  # time at next scr refresh
                        self_btn.status = FINISHED
                if self_btn.status == STARTED and not waitOnFlip:
                    theseKeys = self_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _self_btn_allKeys.extend(theseKeys)
                    if len(_self_btn_allKeys):
                        self_btn.keys = [key.name for key in _self_btn_allKeys]  # storing all keys
                        self_btn.rt = [key.rt for key in _self_btn_allKeys]
                
                # *trial_crosshair* updates
                if trial_crosshair.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_crosshair.frameNStart = frameN  # exact frame index
                    trial_crosshair.tStart = t  # local t and not account for scr refresh
                    trial_crosshair.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_crosshair, 'tStartRefresh')  # time at next scr refresh
                    trial_crosshair.setAutoDraw(True)
                if trial_crosshair.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_crosshair.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_crosshair.tStop = t  # not accounting for scr refresh
                        trial_crosshair.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(trial_crosshair, 'tStopRefresh')  # time at next scr refresh
                        trial_crosshair.setAutoDraw(False)
                
                # *skip2* updates
                waitOnFlip = False
                if skip2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    skip2.frameNStart = frameN  # exact frame index
                    skip2.tStart = t  # local t and not account for scr refresh
                    skip2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(skip2, 'tStartRefresh')  # time at next scr refresh
                    skip2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(skip2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(skip2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if skip2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > skip2.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        skip2.tStop = t  # not accounting for scr refresh
                        skip2.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(skip2, 'tStopRefresh')  # time at next scr refresh
                        skip2.status = FINISHED
                if skip2.status == STARTED and not waitOnFlip:
                    theseKeys = skip2.getKeys(keyList=['s'], waitRelease=False)
                    _skip2_allKeys.extend(theseKeys)
                    if len(_skip2_allKeys):
                        skip2.keys = _skip2_allKeys[-1].name  # just the last key pressed
                        skip2.rt = _skip2_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in selfComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "self"-------
            for thisComponent in selfComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # check responses
            if self_btn.keys in ['', [], None]:  # No response was made
                self_btn.keys = None
            MM_self_loop1.addData('self_btn.keys',self_btn.keys)
            if self_btn.keys != None:  # we had a response
                MM_self_loop1.addData('self_btn.rt', self_btn.rt)
            MM_self_loop1.addData('self_btn.started', self_btn.tStartRefresh)
            MM_self_loop1.addData('self_btn.stopped', self_btn.tStopRefresh)
            MM_self_loop1.addData('trial_crosshair.started', trial_crosshair.tStartRefresh)
            MM_self_loop1.addData('trial_crosshair.stopped', trial_crosshair.tStopRefresh)
            
            # ------Prepare to start Routine "finish_trial"-------
            continueRoutine = True
            routineTimer.add(30.000000)
            # update component parameters for each repeat
            break_btn.keys = []
            break_btn.rt = []
            _break_btn_allKeys = []
            # keep track of which components have finished
            finish_trialComponents = [break_txt, break_btn]
            for thisComponent in finish_trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            finish_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "finish_trial"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = finish_trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=finish_trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *break_txt* updates
                if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_txt.frameNStart = frameN  # exact frame index
                    break_txt.tStart = t  # local t and not account for scr refresh
                    break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                    break_txt.setAutoDraw(True)
                if break_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_txt.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_txt.tStop = t  # not accounting for scr refresh
                        break_txt.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_txt, 'tStopRefresh')  # time at next scr refresh
                        break_txt.setAutoDraw(False)
                
                # *break_btn* updates
                waitOnFlip = False
                if break_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_btn.frameNStart = frameN  # exact frame index
                    break_btn.tStart = t  # local t and not account for scr refresh
                    break_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_btn, 'tStartRefresh')  # time at next scr refresh
                    break_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(break_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(break_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if break_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_btn.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_btn.tStop = t  # not accounting for scr refresh
                        break_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_btn, 'tStopRefresh')  # time at next scr refresh
                        break_btn.status = FINISHED
                if break_btn.status == STARTED and not waitOnFlip:
                    theseKeys = break_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _break_btn_allKeys.extend(theseKeys)
                    if len(_break_btn_allKeys):
                        break_btn.keys = _break_btn_allKeys[-1].name  # just the last key pressed
                        break_btn.rt = _break_btn_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in finish_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "finish_trial"-------
            for thisComponent in finish_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop1.addData('break_txt.started', break_txt.tStartRefresh)
            MM_self_loop1.addData('break_txt.stopped', break_txt.tStopRefresh)
            # check responses
            if break_btn.keys in ['', [], None]:  # No response was made
                break_btn.keys = None
            MM_self_loop1.addData('break_btn.keys',break_btn.keys)
            if break_btn.keys != None:  # we had a response
                MM_self_loop1.addData('break_btn.rt', break_btn.rt)
            MM_self_loop1.addData('break_btn.started', break_btn.tStartRefresh)
            MM_self_loop1.addData('break_btn.stopped', break_btn.tStopRefresh)
            thisExp.nextEntry()
            
        # completed orders[1][outer_loop1.thisN] repeats of 'MM_self_loop1'
        
        
        # set up handler to look after randomisation of conditions etc
        MM_other_loop1 = data.TrialHandler(nReps=orders[2][outer_loop1.thisN], method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='MM_other_loop1')
        thisExp.addLoop(MM_other_loop1)  # add the loop to the experiment
        thisMM_other_loop1 = MM_other_loop1.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMM_other_loop1.rgb)
        if thisMM_other_loop1 != None:
            for paramName in thisMM_other_loop1:
                exec('{} = thisMM_other_loop1[paramName]'.format(paramName))
        
        for thisMM_other_loop1 in MM_other_loop1:
            currentLoop = MM_other_loop1
            # abbreviate parameter names if possible (e.g. rgb = thisMM_other_loop1.rgb)
            if thisMM_other_loop1 != None:
                for paramName in thisMM_other_loop1:
                    exec('{} = thisMM_other_loop1[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "MM_other_instr"-------
            continueRoutine = True
            # update component parameters for each repeat
            advance3.keys = []
            advance3.rt = []
            _advance3_allKeys = []
            # keep track of which components have finished
            MM_other_instrComponents = [MM_other_title, MM_other_body, MM_other_nxt, advance3]
            for thisComponent in MM_other_instrComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            MM_other_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "MM_other_instr"-------
            while continueRoutine:
                # get current time
                t = MM_other_instrClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=MM_other_instrClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *MM_other_title* updates
                if MM_other_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_other_title.frameNStart = frameN  # exact frame index
                    MM_other_title.tStart = t  # local t and not account for scr refresh
                    MM_other_title.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_other_title, 'tStartRefresh')  # time at next scr refresh
                    MM_other_title.setAutoDraw(True)
                
                # *MM_other_body* updates
                if MM_other_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_other_body.frameNStart = frameN  # exact frame index
                    MM_other_body.tStart = t  # local t and not account for scr refresh
                    MM_other_body.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_other_body, 'tStartRefresh')  # time at next scr refresh
                    MM_other_body.setAutoDraw(True)
                
                # *MM_other_nxt* updates
                if MM_other_nxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_other_nxt.frameNStart = frameN  # exact frame index
                    MM_other_nxt.tStart = t  # local t and not account for scr refresh
                    MM_other_nxt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_other_nxt, 'tStartRefresh')  # time at next scr refresh
                    MM_other_nxt.setAutoDraw(True)
                
                # *advance3* updates
                waitOnFlip = False
                if advance3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    advance3.frameNStart = frameN  # exact frame index
                    advance3.tStart = t  # local t and not account for scr refresh
                    advance3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(advance3, 'tStartRefresh')  # time at next scr refresh
                    advance3.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(advance3.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(advance3.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if advance3.status == STARTED and not waitOnFlip:
                    theseKeys = advance3.getKeys(keyList=['space'], waitRelease=False)
                    _advance3_allKeys.extend(theseKeys)
                    if len(_advance3_allKeys):
                        advance3.keys = [key.name for key in _advance3_allKeys]  # storing all keys
                        advance3.rt = [key.rt for key in _advance3_allKeys]
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in MM_other_instrComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "MM_other_instr"-------
            for thisComponent in MM_other_instrComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop1.addData('MM_other_title.started', MM_other_title.tStartRefresh)
            MM_other_loop1.addData('MM_other_title.stopped', MM_other_title.tStopRefresh)
            MM_other_loop1.addData('MM_other_body.started', MM_other_body.tStartRefresh)
            MM_other_loop1.addData('MM_other_body.stopped', MM_other_body.tStopRefresh)
            MM_other_loop1.addData('MM_other_nxt.started', MM_other_nxt.tStartRefresh)
            MM_other_loop1.addData('MM_other_nxt.stopped', MM_other_nxt.tStopRefresh)
            # check responses
            if advance3.keys in ['', [], None]:  # No response was made
                advance3.keys = None
            MM_other_loop1.addData('advance3.keys',advance3.keys)
            if advance3.keys != None:  # we had a response
                MM_other_loop1.addData('advance3.rt', advance3.rt)
            MM_other_loop1.addData('advance3.started', advance3.tStartRefresh)
            MM_other_loop1.addData('advance3.stopped', advance3.tStopRefresh)
            # the Routine "MM_other_instr" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_other_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_other_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_other_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_other_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_other_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_other_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_other_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_other_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_other_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_other_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_other_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_other_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "scan_wait"-------
            continueRoutine = True
            # update component parameters for each repeat
            begin_scan_btn.keys = []
            begin_scan_btn.rt = []
            _begin_scan_btn_allKeys = []
            # keep track of which components have finished
            scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
            for thisComponent in scan_waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "scan_wait"-------
            while continueRoutine:
                # get current time
                t = scan_waitClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *begin_scan_txt* updates
                if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_txt.frameNStart = frameN  # exact frame index
                    begin_scan_txt.tStart = t  # local t and not account for scr refresh
                    begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_txt.setAutoDraw(True)
                
                # *still_txt* updates
                if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt.frameNStart = frameN  # exact frame index
                    still_txt.tStart = t  # local t and not account for scr refresh
                    still_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
                    still_txt.setAutoDraw(True)
                
                # *begin_scan_btn* updates
                waitOnFlip = False
                if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_btn.frameNStart = frameN  # exact frame index
                    begin_scan_btn.tStart = t  # local t and not account for scr refresh
                    begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
                if begin_scan_btn.status == STARTED and not waitOnFlip:
                    theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
                    _begin_scan_btn_allKeys.extend(theseKeys)
                    if len(_begin_scan_btn_allKeys):
                        begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
                        begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in scan_waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "scan_wait"-------
            for thisComponent in scan_waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop1.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
            MM_other_loop1.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
            MM_other_loop1.addData('still_txt.started', still_txt.tStartRefresh)
            MM_other_loop1.addData('still_txt.stopped', still_txt.tStopRefresh)
            # check responses
            if begin_scan_btn.keys in ['', [], None]:  # No response was made
                begin_scan_btn.keys = None
            MM_other_loop1.addData('begin_scan_btn.keys',begin_scan_btn.keys)
            if begin_scan_btn.keys != None:  # we had a response
                MM_other_loop1.addData('begin_scan_btn.rt', begin_scan_btn.rt)
            MM_other_loop1.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
            MM_other_loop1.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
            # the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "movie_josh"-------
            continueRoutine = True
            routineTimer.add(307.500000)
            # update component parameters for each repeat
            other_j_btn1.keys = []
            other_j_btn1.rt = []
            _other_j_btn1_allKeys = []
            josh_mov1 = visual.MovieStim3(
                win=win, name='josh_mov1',
                noAudio = True,
                filename='josh_meditation.mp4',
                ori=0, pos=(0, 0), opacity=1,
                loop=False,
                depth=-1.0,
                )
            skip3.keys = []
            skip3.rt = []
            _skip3_allKeys = []
            # keep track of which components have finished
            movie_joshComponents = [other_j_btn1, josh_mov1, skip3]
            for thisComponent in movie_joshComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            movie_joshClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "movie_josh"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = movie_joshClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=movie_joshClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *other_j_btn1* updates
                waitOnFlip = False
                if other_j_btn1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    other_j_btn1.frameNStart = frameN  # exact frame index
                    other_j_btn1.tStart = t  # local t and not account for scr refresh
                    other_j_btn1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(other_j_btn1, 'tStartRefresh')  # time at next scr refresh
                    other_j_btn1.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(other_j_btn1.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(other_j_btn1.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if other_j_btn1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > other_j_btn1.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        other_j_btn1.tStop = t  # not accounting for scr refresh
                        other_j_btn1.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(other_j_btn1, 'tStopRefresh')  # time at next scr refresh
                        other_j_btn1.status = FINISHED
                if other_j_btn1.status == STARTED and not waitOnFlip:
                    theseKeys = other_j_btn1.getKeys(keyList=['space', '1'], waitRelease=False)
                    _other_j_btn1_allKeys.extend(theseKeys)
                    if len(_other_j_btn1_allKeys):
                        other_j_btn1.keys = [key.name for key in _other_j_btn1_allKeys]  # storing all keys
                        other_j_btn1.rt = [key.rt for key in _other_j_btn1_allKeys]
                
                # *josh_mov1* updates
                if josh_mov1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    josh_mov1.frameNStart = frameN  # exact frame index
                    josh_mov1.tStart = t  # local t and not account for scr refresh
                    josh_mov1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(josh_mov1, 'tStartRefresh')  # time at next scr refresh
                    josh_mov1.setAutoDraw(True)
                if josh_mov1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > josh_mov1.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        josh_mov1.tStop = t  # not accounting for scr refresh
                        josh_mov1.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(josh_mov1, 'tStopRefresh')  # time at next scr refresh
                        josh_mov1.setAutoDraw(False)
                
                # *skip3* updates
                waitOnFlip = False
                if skip3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    skip3.frameNStart = frameN  # exact frame index
                    skip3.tStart = t  # local t and not account for scr refresh
                    skip3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(skip3, 'tStartRefresh')  # time at next scr refresh
                    skip3.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(skip3.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(skip3.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if skip3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > skip3.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        skip3.tStop = t  # not accounting for scr refresh
                        skip3.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(skip3, 'tStopRefresh')  # time at next scr refresh
                        skip3.status = FINISHED
                if skip3.status == STARTED and not waitOnFlip:
                    theseKeys = skip3.getKeys(keyList=['s'], waitRelease=False)
                    _skip3_allKeys.extend(theseKeys)
                    if len(_skip3_allKeys):
                        skip3.keys = _skip3_allKeys[-1].name  # just the last key pressed
                        skip3.rt = _skip3_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in movie_joshComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "movie_josh"-------
            for thisComponent in movie_joshComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # check responses
            if other_j_btn1.keys in ['', [], None]:  # No response was made
                other_j_btn1.keys = None
            MM_other_loop1.addData('other_j_btn1.keys',other_j_btn1.keys)
            if other_j_btn1.keys != None:  # we had a response
                MM_other_loop1.addData('other_j_btn1.rt', other_j_btn1.rt)
            MM_other_loop1.addData('other_j_btn1.started', other_j_btn1.tStartRefresh)
            MM_other_loop1.addData('other_j_btn1.stopped', other_j_btn1.tStopRefresh)
            josh_mov1.stop()
            
            # ------Prepare to start Routine "finish_trial"-------
            continueRoutine = True
            routineTimer.add(30.000000)
            # update component parameters for each repeat
            break_btn.keys = []
            break_btn.rt = []
            _break_btn_allKeys = []
            # keep track of which components have finished
            finish_trialComponents = [break_txt, break_btn]
            for thisComponent in finish_trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            finish_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "finish_trial"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = finish_trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=finish_trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *break_txt* updates
                if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_txt.frameNStart = frameN  # exact frame index
                    break_txt.tStart = t  # local t and not account for scr refresh
                    break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                    break_txt.setAutoDraw(True)
                if break_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_txt.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_txt.tStop = t  # not accounting for scr refresh
                        break_txt.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_txt, 'tStopRefresh')  # time at next scr refresh
                        break_txt.setAutoDraw(False)
                
                # *break_btn* updates
                waitOnFlip = False
                if break_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_btn.frameNStart = frameN  # exact frame index
                    break_btn.tStart = t  # local t and not account for scr refresh
                    break_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_btn, 'tStartRefresh')  # time at next scr refresh
                    break_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(break_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(break_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if break_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_btn.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_btn.tStop = t  # not accounting for scr refresh
                        break_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_btn, 'tStopRefresh')  # time at next scr refresh
                        break_btn.status = FINISHED
                if break_btn.status == STARTED and not waitOnFlip:
                    theseKeys = break_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _break_btn_allKeys.extend(theseKeys)
                    if len(_break_btn_allKeys):
                        break_btn.keys = _break_btn_allKeys[-1].name  # just the last key pressed
                        break_btn.rt = _break_btn_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in finish_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "finish_trial"-------
            for thisComponent in finish_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop1.addData('break_txt.started', break_txt.tStartRefresh)
            MM_other_loop1.addData('break_txt.stopped', break_txt.tStopRefresh)
            # check responses
            if break_btn.keys in ['', [], None]:  # No response was made
                break_btn.keys = None
            MM_other_loop1.addData('break_btn.keys',break_btn.keys)
            if break_btn.keys != None:  # we had a response
                MM_other_loop1.addData('break_btn.rt', break_btn.rt)
            MM_other_loop1.addData('break_btn.started', break_btn.tStartRefresh)
            MM_other_loop1.addData('break_btn.stopped', break_btn.tStopRefresh)
            thisExp.nextEntry()
            
        # completed orders[2][outer_loop1.thisN] repeats of 'MM_other_loop1'
        
        
        # set up handler to look after randomisation of conditions etc
        MC_other_loop1 = data.TrialHandler(nReps=orders[3][outer_loop1.thisN], method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='MC_other_loop1')
        thisExp.addLoop(MC_other_loop1)  # add the loop to the experiment
        thisMC_other_loop1 = MC_other_loop1.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMC_other_loop1.rgb)
        if thisMC_other_loop1 != None:
            for paramName in thisMC_other_loop1:
                exec('{} = thisMC_other_loop1[paramName]'.format(paramName))
        
        for thisMC_other_loop1 in MC_other_loop1:
            currentLoop = MC_other_loop1
            # abbreviate parameter names if possible (e.g. rgb = thisMC_other_loop1.rgb)
            if thisMC_other_loop1 != None:
                for paramName in thisMC_other_loop1:
                    exec('{} = thisMC_other_loop1[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "MC_other_instr"-------
            continueRoutine = True
            # update component parameters for each repeat
            advance4.keys = []
            advance4.rt = []
            _advance4_allKeys = []
            # keep track of which components have finished
            MC_other_instrComponents = [MC_other_title, MC_other_body, MC_other_nxt, advance4]
            for thisComponent in MC_other_instrComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            MC_other_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "MC_other_instr"-------
            while continueRoutine:
                # get current time
                t = MC_other_instrClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=MC_other_instrClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *MC_other_title* updates
                if MC_other_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_other_title.frameNStart = frameN  # exact frame index
                    MC_other_title.tStart = t  # local t and not account for scr refresh
                    MC_other_title.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_other_title, 'tStartRefresh')  # time at next scr refresh
                    MC_other_title.setAutoDraw(True)
                
                # *MC_other_body* updates
                if MC_other_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_other_body.frameNStart = frameN  # exact frame index
                    MC_other_body.tStart = t  # local t and not account for scr refresh
                    MC_other_body.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_other_body, 'tStartRefresh')  # time at next scr refresh
                    MC_other_body.setAutoDraw(True)
                
                # *MC_other_nxt* updates
                if MC_other_nxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_other_nxt.frameNStart = frameN  # exact frame index
                    MC_other_nxt.tStart = t  # local t and not account for scr refresh
                    MC_other_nxt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_other_nxt, 'tStartRefresh')  # time at next scr refresh
                    MC_other_nxt.setAutoDraw(True)
                
                # *advance4* updates
                waitOnFlip = False
                if advance4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    advance4.frameNStart = frameN  # exact frame index
                    advance4.tStart = t  # local t and not account for scr refresh
                    advance4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(advance4, 'tStartRefresh')  # time at next scr refresh
                    advance4.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(advance4.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(advance4.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if advance4.status == STARTED and not waitOnFlip:
                    theseKeys = advance4.getKeys(keyList=['space', '1'], waitRelease=False)
                    _advance4_allKeys.extend(theseKeys)
                    if len(_advance4_allKeys):
                        advance4.keys = _advance4_allKeys[-1].name  # just the last key pressed
                        advance4.rt = _advance4_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in MC_other_instrComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "MC_other_instr"-------
            for thisComponent in MC_other_instrComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop1.addData('MC_other_title.started', MC_other_title.tStartRefresh)
            MC_other_loop1.addData('MC_other_title.stopped', MC_other_title.tStopRefresh)
            MC_other_loop1.addData('MC_other_body.started', MC_other_body.tStartRefresh)
            MC_other_loop1.addData('MC_other_body.stopped', MC_other_body.tStopRefresh)
            MC_other_loop1.addData('MC_other_nxt.started', MC_other_nxt.tStartRefresh)
            MC_other_loop1.addData('MC_other_nxt.stopped', MC_other_nxt.tStopRefresh)
            # check responses
            if advance4.keys in ['', [], None]:  # No response was made
                advance4.keys = None
            MC_other_loop1.addData('advance4.keys',advance4.keys)
            if advance4.keys != None:  # we had a response
                MC_other_loop1.addData('advance4.rt', advance4.rt)
            MC_other_loop1.addData('advance4.started', advance4.tStartRefresh)
            MC_other_loop1.addData('advance4.stopped', advance4.tStopRefresh)
            # the Routine "MC_other_instr" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_other_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_other_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_other_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_other_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_other_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_other_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_other_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_other_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_other_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop1.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_other_loop1.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_other_loop1.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_other_loop1.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "scan_wait"-------
            continueRoutine = True
            # update component parameters for each repeat
            begin_scan_btn.keys = []
            begin_scan_btn.rt = []
            _begin_scan_btn_allKeys = []
            # keep track of which components have finished
            scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
            for thisComponent in scan_waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "scan_wait"-------
            while continueRoutine:
                # get current time
                t = scan_waitClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *begin_scan_txt* updates
                if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_txt.frameNStart = frameN  # exact frame index
                    begin_scan_txt.tStart = t  # local t and not account for scr refresh
                    begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_txt.setAutoDraw(True)
                
                # *still_txt* updates
                if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt.frameNStart = frameN  # exact frame index
                    still_txt.tStart = t  # local t and not account for scr refresh
                    still_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
                    still_txt.setAutoDraw(True)
                
                # *begin_scan_btn* updates
                waitOnFlip = False
                if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_btn.frameNStart = frameN  # exact frame index
                    begin_scan_btn.tStart = t  # local t and not account for scr refresh
                    begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
                if begin_scan_btn.status == STARTED and not waitOnFlip:
                    theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
                    _begin_scan_btn_allKeys.extend(theseKeys)
                    if len(_begin_scan_btn_allKeys):
                        begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
                        begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in scan_waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "scan_wait"-------
            for thisComponent in scan_waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop1.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
            MC_other_loop1.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
            MC_other_loop1.addData('still_txt.started', still_txt.tStartRefresh)
            MC_other_loop1.addData('still_txt.stopped', still_txt.tStopRefresh)
            # check responses
            if begin_scan_btn.keys in ['', [], None]:  # No response was made
                begin_scan_btn.keys = None
            MC_other_loop1.addData('begin_scan_btn.keys',begin_scan_btn.keys)
            if begin_scan_btn.keys != None:  # we had a response
                MC_other_loop1.addData('begin_scan_btn.rt', begin_scan_btn.rt)
            MC_other_loop1.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
            MC_other_loop1.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
            # the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "movie_josh"-------
            continueRoutine = True
            routineTimer.add(307.500000)
            # update component parameters for each repeat
            other_j_btn1.keys = []
            other_j_btn1.rt = []
            _other_j_btn1_allKeys = []
            josh_mov1 = visual.MovieStim3(
                win=win, name='josh_mov1',
                noAudio = True,
                filename='josh_meditation.mp4',
                ori=0, pos=(0, 0), opacity=1,
                loop=False,
                depth=-1.0,
                )
            skip3.keys = []
            skip3.rt = []
            _skip3_allKeys = []
            # keep track of which components have finished
            movie_joshComponents = [other_j_btn1, josh_mov1, skip3]
            for thisComponent in movie_joshComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            movie_joshClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "movie_josh"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = movie_joshClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=movie_joshClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *other_j_btn1* updates
                waitOnFlip = False
                if other_j_btn1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    other_j_btn1.frameNStart = frameN  # exact frame index
                    other_j_btn1.tStart = t  # local t and not account for scr refresh
                    other_j_btn1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(other_j_btn1, 'tStartRefresh')  # time at next scr refresh
                    other_j_btn1.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(other_j_btn1.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(other_j_btn1.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if other_j_btn1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > other_j_btn1.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        other_j_btn1.tStop = t  # not accounting for scr refresh
                        other_j_btn1.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(other_j_btn1, 'tStopRefresh')  # time at next scr refresh
                        other_j_btn1.status = FINISHED
                if other_j_btn1.status == STARTED and not waitOnFlip:
                    theseKeys = other_j_btn1.getKeys(keyList=['space', '1'], waitRelease=False)
                    _other_j_btn1_allKeys.extend(theseKeys)
                    if len(_other_j_btn1_allKeys):
                        other_j_btn1.keys = [key.name for key in _other_j_btn1_allKeys]  # storing all keys
                        other_j_btn1.rt = [key.rt for key in _other_j_btn1_allKeys]
                
                # *josh_mov1* updates
                if josh_mov1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    josh_mov1.frameNStart = frameN  # exact frame index
                    josh_mov1.tStart = t  # local t and not account for scr refresh
                    josh_mov1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(josh_mov1, 'tStartRefresh')  # time at next scr refresh
                    josh_mov1.setAutoDraw(True)
                if josh_mov1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > josh_mov1.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        josh_mov1.tStop = t  # not accounting for scr refresh
                        josh_mov1.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(josh_mov1, 'tStopRefresh')  # time at next scr refresh
                        josh_mov1.setAutoDraw(False)
                
                # *skip3* updates
                waitOnFlip = False
                if skip3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    skip3.frameNStart = frameN  # exact frame index
                    skip3.tStart = t  # local t and not account for scr refresh
                    skip3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(skip3, 'tStartRefresh')  # time at next scr refresh
                    skip3.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(skip3.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(skip3.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if skip3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > skip3.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        skip3.tStop = t  # not accounting for scr refresh
                        skip3.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(skip3, 'tStopRefresh')  # time at next scr refresh
                        skip3.status = FINISHED
                if skip3.status == STARTED and not waitOnFlip:
                    theseKeys = skip3.getKeys(keyList=['s'], waitRelease=False)
                    _skip3_allKeys.extend(theseKeys)
                    if len(_skip3_allKeys):
                        skip3.keys = _skip3_allKeys[-1].name  # just the last key pressed
                        skip3.rt = _skip3_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in movie_joshComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "movie_josh"-------
            for thisComponent in movie_joshComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # check responses
            if other_j_btn1.keys in ['', [], None]:  # No response was made
                other_j_btn1.keys = None
            MC_other_loop1.addData('other_j_btn1.keys',other_j_btn1.keys)
            if other_j_btn1.keys != None:  # we had a response
                MC_other_loop1.addData('other_j_btn1.rt', other_j_btn1.rt)
            MC_other_loop1.addData('other_j_btn1.started', other_j_btn1.tStartRefresh)
            MC_other_loop1.addData('other_j_btn1.stopped', other_j_btn1.tStopRefresh)
            josh_mov1.stop()
            
            # ------Prepare to start Routine "finish_trial"-------
            continueRoutine = True
            routineTimer.add(30.000000)
            # update component parameters for each repeat
            break_btn.keys = []
            break_btn.rt = []
            _break_btn_allKeys = []
            # keep track of which components have finished
            finish_trialComponents = [break_txt, break_btn]
            for thisComponent in finish_trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            finish_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "finish_trial"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = finish_trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=finish_trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *break_txt* updates
                if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_txt.frameNStart = frameN  # exact frame index
                    break_txt.tStart = t  # local t and not account for scr refresh
                    break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                    break_txt.setAutoDraw(True)
                if break_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_txt.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_txt.tStop = t  # not accounting for scr refresh
                        break_txt.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_txt, 'tStopRefresh')  # time at next scr refresh
                        break_txt.setAutoDraw(False)
                
                # *break_btn* updates
                waitOnFlip = False
                if break_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_btn.frameNStart = frameN  # exact frame index
                    break_btn.tStart = t  # local t and not account for scr refresh
                    break_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_btn, 'tStartRefresh')  # time at next scr refresh
                    break_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(break_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(break_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if break_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_btn.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_btn.tStop = t  # not accounting for scr refresh
                        break_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_btn, 'tStopRefresh')  # time at next scr refresh
                        break_btn.status = FINISHED
                if break_btn.status == STARTED and not waitOnFlip:
                    theseKeys = break_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _break_btn_allKeys.extend(theseKeys)
                    if len(_break_btn_allKeys):
                        break_btn.keys = _break_btn_allKeys[-1].name  # just the last key pressed
                        break_btn.rt = _break_btn_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in finish_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "finish_trial"-------
            for thisComponent in finish_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop1.addData('break_txt.started', break_txt.tStartRefresh)
            MC_other_loop1.addData('break_txt.stopped', break_txt.tStopRefresh)
            # check responses
            if break_btn.keys in ['', [], None]:  # No response was made
                break_btn.keys = None
            MC_other_loop1.addData('break_btn.keys',break_btn.keys)
            if break_btn.keys != None:  # we had a response
                MC_other_loop1.addData('break_btn.rt', break_btn.rt)
            MC_other_loop1.addData('break_btn.started', break_btn.tStartRefresh)
            MC_other_loop1.addData('break_btn.stopped', break_btn.tStopRefresh)
            thisExp.nextEntry()
            
        # completed orders[3][outer_loop1.thisN] repeats of 'MC_other_loop1'
        
        thisExp.nextEntry()
        
    # completed orders2[0][big_guy.thisN] repeats of 'outer_loop1'
    
    
    # set up handler to look after randomisation of conditions etc
    outer_loop2 = data.TrialHandler(nReps=orders2[1][big_guy.thisN], method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='outer_loop2')
    thisExp.addLoop(outer_loop2)  # add the loop to the experiment
    thisOuter_loop2 = outer_loop2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisOuter_loop2.rgb)
    if thisOuter_loop2 != None:
        for paramName in thisOuter_loop2:
            exec('{} = thisOuter_loop2[paramName]'.format(paramName))
    
    for thisOuter_loop2 in outer_loop2:
        currentLoop = outer_loop2
        # abbreviate parameter names if possible (e.g. rgb = thisOuter_loop2.rgb)
        if thisOuter_loop2 != None:
            for paramName in thisOuter_loop2:
                exec('{} = thisOuter_loop2[paramName]'.format(paramName))
        
        # set up handler to look after randomisation of conditions etc
        MM_self_loop2 = data.TrialHandler(nReps=orders1[0][outer_loop2.thisN], method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='MM_self_loop2')
        thisExp.addLoop(MM_self_loop2)  # add the loop to the experiment
        thisMM_self_loop2 = MM_self_loop2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMM_self_loop2.rgb)
        if thisMM_self_loop2 != None:
            for paramName in thisMM_self_loop2:
                exec('{} = thisMM_self_loop2[paramName]'.format(paramName))
        
        for thisMM_self_loop2 in MM_self_loop2:
            currentLoop = MM_self_loop2
            # abbreviate parameter names if possible (e.g. rgb = thisMM_self_loop2.rgb)
            if thisMM_self_loop2 != None:
                for paramName in thisMM_self_loop2:
                    exec('{} = thisMM_self_loop2[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "MM_self_instr"-------
            continueRoutine = True
            # update component parameters for each repeat
            advance_2.keys = []
            advance_2.rt = []
            _advance_2_allKeys = []
            # keep track of which components have finished
            MM_self_instrComponents = [MM_self_title, MM_self_body, MM_self_nxt, advance_2]
            for thisComponent in MM_self_instrComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            MM_self_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "MM_self_instr"-------
            while continueRoutine:
                # get current time
                t = MM_self_instrClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=MM_self_instrClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *MM_self_title* updates
                if MM_self_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_self_title.frameNStart = frameN  # exact frame index
                    MM_self_title.tStart = t  # local t and not account for scr refresh
                    MM_self_title.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_self_title, 'tStartRefresh')  # time at next scr refresh
                    MM_self_title.setAutoDraw(True)
                
                # *MM_self_body* updates
                if MM_self_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_self_body.frameNStart = frameN  # exact frame index
                    MM_self_body.tStart = t  # local t and not account for scr refresh
                    MM_self_body.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_self_body, 'tStartRefresh')  # time at next scr refresh
                    MM_self_body.setAutoDraw(True)
                
                # *MM_self_nxt* updates
                if MM_self_nxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_self_nxt.frameNStart = frameN  # exact frame index
                    MM_self_nxt.tStart = t  # local t and not account for scr refresh
                    MM_self_nxt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_self_nxt, 'tStartRefresh')  # time at next scr refresh
                    MM_self_nxt.setAutoDraw(True)
                
                # *advance_2* updates
                waitOnFlip = False
                if advance_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    advance_2.frameNStart = frameN  # exact frame index
                    advance_2.tStart = t  # local t and not account for scr refresh
                    advance_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(advance_2, 'tStartRefresh')  # time at next scr refresh
                    advance_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(advance_2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(advance_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if advance_2.status == STARTED and not waitOnFlip:
                    theseKeys = advance_2.getKeys(keyList=['space', '1'], waitRelease=False)
                    _advance_2_allKeys.extend(theseKeys)
                    if len(_advance_2_allKeys):
                        advance_2.keys = _advance_2_allKeys[-1].name  # just the last key pressed
                        advance_2.rt = _advance_2_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in MM_self_instrComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "MM_self_instr"-------
            for thisComponent in MM_self_instrComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop2.addData('MM_self_title.started', MM_self_title.tStartRefresh)
            MM_self_loop2.addData('MM_self_title.stopped', MM_self_title.tStopRefresh)
            MM_self_loop2.addData('MM_self_body.started', MM_self_body.tStartRefresh)
            MM_self_loop2.addData('MM_self_body.stopped', MM_self_body.tStopRefresh)
            MM_self_loop2.addData('MM_self_nxt.started', MM_self_nxt.tStartRefresh)
            MM_self_loop2.addData('MM_self_nxt.stopped', MM_self_nxt.tStopRefresh)
            # check responses
            if advance_2.keys in ['', [], None]:  # No response was made
                advance_2.keys = None
            MM_self_loop2.addData('advance_2.keys',advance_2.keys)
            if advance_2.keys != None:  # we had a response
                MM_self_loop2.addData('advance_2.rt', advance_2.rt)
            MM_self_loop2.addData('advance_2.started', advance_2.tStartRefresh)
            MM_self_loop2.addData('advance_2.stopped', advance_2.tStopRefresh)
            # the Routine "MM_self_instr" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_self_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_self_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_self_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_self_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_self_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_self_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_self_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_self_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_self_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_self_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_self_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_self_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "scan_wait"-------
            continueRoutine = True
            # update component parameters for each repeat
            begin_scan_btn.keys = []
            begin_scan_btn.rt = []
            _begin_scan_btn_allKeys = []
            # keep track of which components have finished
            scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
            for thisComponent in scan_waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "scan_wait"-------
            while continueRoutine:
                # get current time
                t = scan_waitClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *begin_scan_txt* updates
                if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_txt.frameNStart = frameN  # exact frame index
                    begin_scan_txt.tStart = t  # local t and not account for scr refresh
                    begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_txt.setAutoDraw(True)
                
                # *still_txt* updates
                if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt.frameNStart = frameN  # exact frame index
                    still_txt.tStart = t  # local t and not account for scr refresh
                    still_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
                    still_txt.setAutoDraw(True)
                
                # *begin_scan_btn* updates
                waitOnFlip = False
                if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_btn.frameNStart = frameN  # exact frame index
                    begin_scan_btn.tStart = t  # local t and not account for scr refresh
                    begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
                if begin_scan_btn.status == STARTED and not waitOnFlip:
                    theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
                    _begin_scan_btn_allKeys.extend(theseKeys)
                    if len(_begin_scan_btn_allKeys):
                        begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
                        begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in scan_waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "scan_wait"-------
            for thisComponent in scan_waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop2.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
            MM_self_loop2.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
            MM_self_loop2.addData('still_txt.started', still_txt.tStartRefresh)
            MM_self_loop2.addData('still_txt.stopped', still_txt.tStopRefresh)
            # check responses
            if begin_scan_btn.keys in ['', [], None]:  # No response was made
                begin_scan_btn.keys = None
            MM_self_loop2.addData('begin_scan_btn.keys',begin_scan_btn.keys)
            if begin_scan_btn.keys != None:  # we had a response
                MM_self_loop2.addData('begin_scan_btn.rt', begin_scan_btn.rt)
            MM_self_loop2.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
            MM_self_loop2.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
            # the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "self"-------
            continueRoutine = True
            routineTimer.add(307.500000)
            # update component parameters for each repeat
            self_btn.keys = []
            self_btn.rt = []
            _self_btn_allKeys = []
            skip2.keys = []
            skip2.rt = []
            _skip2_allKeys = []
            # keep track of which components have finished
            selfComponents = [self_btn, trial_crosshair, skip2]
            for thisComponent in selfComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            selfClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "self"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = selfClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=selfClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *self_btn* updates
                waitOnFlip = False
                if self_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    self_btn.frameNStart = frameN  # exact frame index
                    self_btn.tStart = t  # local t and not account for scr refresh
                    self_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(self_btn, 'tStartRefresh')  # time at next scr refresh
                    self_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(self_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(self_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if self_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > self_btn.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        self_btn.tStop = t  # not accounting for scr refresh
                        self_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(self_btn, 'tStopRefresh')  # time at next scr refresh
                        self_btn.status = FINISHED
                if self_btn.status == STARTED and not waitOnFlip:
                    theseKeys = self_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _self_btn_allKeys.extend(theseKeys)
                    if len(_self_btn_allKeys):
                        self_btn.keys = [key.name for key in _self_btn_allKeys]  # storing all keys
                        self_btn.rt = [key.rt for key in _self_btn_allKeys]
                
                # *trial_crosshair* updates
                if trial_crosshair.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_crosshair.frameNStart = frameN  # exact frame index
                    trial_crosshair.tStart = t  # local t and not account for scr refresh
                    trial_crosshair.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_crosshair, 'tStartRefresh')  # time at next scr refresh
                    trial_crosshair.setAutoDraw(True)
                if trial_crosshair.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_crosshair.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_crosshair.tStop = t  # not accounting for scr refresh
                        trial_crosshair.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(trial_crosshair, 'tStopRefresh')  # time at next scr refresh
                        trial_crosshair.setAutoDraw(False)
                
                # *skip2* updates
                waitOnFlip = False
                if skip2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    skip2.frameNStart = frameN  # exact frame index
                    skip2.tStart = t  # local t and not account for scr refresh
                    skip2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(skip2, 'tStartRefresh')  # time at next scr refresh
                    skip2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(skip2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(skip2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if skip2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > skip2.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        skip2.tStop = t  # not accounting for scr refresh
                        skip2.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(skip2, 'tStopRefresh')  # time at next scr refresh
                        skip2.status = FINISHED
                if skip2.status == STARTED and not waitOnFlip:
                    theseKeys = skip2.getKeys(keyList=['s'], waitRelease=False)
                    _skip2_allKeys.extend(theseKeys)
                    if len(_skip2_allKeys):
                        skip2.keys = _skip2_allKeys[-1].name  # just the last key pressed
                        skip2.rt = _skip2_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in selfComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "self"-------
            for thisComponent in selfComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # check responses
            if self_btn.keys in ['', [], None]:  # No response was made
                self_btn.keys = None
            MM_self_loop2.addData('self_btn.keys',self_btn.keys)
            if self_btn.keys != None:  # we had a response
                MM_self_loop2.addData('self_btn.rt', self_btn.rt)
            MM_self_loop2.addData('self_btn.started', self_btn.tStartRefresh)
            MM_self_loop2.addData('self_btn.stopped', self_btn.tStopRefresh)
            MM_self_loop2.addData('trial_crosshair.started', trial_crosshair.tStartRefresh)
            MM_self_loop2.addData('trial_crosshair.stopped', trial_crosshair.tStopRefresh)
            
            # ------Prepare to start Routine "finish_trial"-------
            continueRoutine = True
            routineTimer.add(30.000000)
            # update component parameters for each repeat
            break_btn.keys = []
            break_btn.rt = []
            _break_btn_allKeys = []
            # keep track of which components have finished
            finish_trialComponents = [break_txt, break_btn]
            for thisComponent in finish_trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            finish_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "finish_trial"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = finish_trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=finish_trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *break_txt* updates
                if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_txt.frameNStart = frameN  # exact frame index
                    break_txt.tStart = t  # local t and not account for scr refresh
                    break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                    break_txt.setAutoDraw(True)
                if break_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_txt.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_txt.tStop = t  # not accounting for scr refresh
                        break_txt.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_txt, 'tStopRefresh')  # time at next scr refresh
                        break_txt.setAutoDraw(False)
                
                # *break_btn* updates
                waitOnFlip = False
                if break_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_btn.frameNStart = frameN  # exact frame index
                    break_btn.tStart = t  # local t and not account for scr refresh
                    break_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_btn, 'tStartRefresh')  # time at next scr refresh
                    break_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(break_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(break_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if break_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_btn.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_btn.tStop = t  # not accounting for scr refresh
                        break_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_btn, 'tStopRefresh')  # time at next scr refresh
                        break_btn.status = FINISHED
                if break_btn.status == STARTED and not waitOnFlip:
                    theseKeys = break_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _break_btn_allKeys.extend(theseKeys)
                    if len(_break_btn_allKeys):
                        break_btn.keys = _break_btn_allKeys[-1].name  # just the last key pressed
                        break_btn.rt = _break_btn_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in finish_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "finish_trial"-------
            for thisComponent in finish_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_self_loop2.addData('break_txt.started', break_txt.tStartRefresh)
            MM_self_loop2.addData('break_txt.stopped', break_txt.tStopRefresh)
            # check responses
            if break_btn.keys in ['', [], None]:  # No response was made
                break_btn.keys = None
            MM_self_loop2.addData('break_btn.keys',break_btn.keys)
            if break_btn.keys != None:  # we had a response
                MM_self_loop2.addData('break_btn.rt', break_btn.rt)
            MM_self_loop2.addData('break_btn.started', break_btn.tStartRefresh)
            MM_self_loop2.addData('break_btn.stopped', break_btn.tStopRefresh)
            thisExp.nextEntry()
            
        # completed orders1[0][outer_loop2.thisN] repeats of 'MM_self_loop2'
        
        
        # set up handler to look after randomisation of conditions etc
        MC_self_loop2 = data.TrialHandler(nReps=orders1[1][outer_loop2.thisN], method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='MC_self_loop2')
        thisExp.addLoop(MC_self_loop2)  # add the loop to the experiment
        thisMC_self_loop2 = MC_self_loop2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMC_self_loop2.rgb)
        if thisMC_self_loop2 != None:
            for paramName in thisMC_self_loop2:
                exec('{} = thisMC_self_loop2[paramName]'.format(paramName))
        
        for thisMC_self_loop2 in MC_self_loop2:
            currentLoop = MC_self_loop2
            # abbreviate parameter names if possible (e.g. rgb = thisMC_self_loop2.rgb)
            if thisMC_self_loop2 != None:
                for paramName in thisMC_self_loop2:
                    exec('{} = thisMC_self_loop2[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "MC_self_instr"-------
            continueRoutine = True
            # update component parameters for each repeat
            advance_1.keys = []
            advance_1.rt = []
            _advance_1_allKeys = []
            # keep track of which components have finished
            MC_self_instrComponents = [MC_self_title, MC_self_body, MC_self_nxt, advance_1]
            for thisComponent in MC_self_instrComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            MC_self_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "MC_self_instr"-------
            while continueRoutine:
                # get current time
                t = MC_self_instrClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=MC_self_instrClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *MC_self_title* updates
                if MC_self_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_self_title.frameNStart = frameN  # exact frame index
                    MC_self_title.tStart = t  # local t and not account for scr refresh
                    MC_self_title.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_self_title, 'tStartRefresh')  # time at next scr refresh
                    MC_self_title.setAutoDraw(True)
                
                # *MC_self_body* updates
                if MC_self_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_self_body.frameNStart = frameN  # exact frame index
                    MC_self_body.tStart = t  # local t and not account for scr refresh
                    MC_self_body.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_self_body, 'tStartRefresh')  # time at next scr refresh
                    MC_self_body.setAutoDraw(True)
                
                # *MC_self_nxt* updates
                if MC_self_nxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_self_nxt.frameNStart = frameN  # exact frame index
                    MC_self_nxt.tStart = t  # local t and not account for scr refresh
                    MC_self_nxt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_self_nxt, 'tStartRefresh')  # time at next scr refresh
                    MC_self_nxt.setAutoDraw(True)
                
                # *advance_1* updates
                waitOnFlip = False
                if advance_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    advance_1.frameNStart = frameN  # exact frame index
                    advance_1.tStart = t  # local t and not account for scr refresh
                    advance_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(advance_1, 'tStartRefresh')  # time at next scr refresh
                    advance_1.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(advance_1.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(advance_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if advance_1.status == STARTED and not waitOnFlip:
                    theseKeys = advance_1.getKeys(keyList=['space', '1'], waitRelease=False)
                    _advance_1_allKeys.extend(theseKeys)
                    if len(_advance_1_allKeys):
                        advance_1.keys = _advance_1_allKeys[-1].name  # just the last key pressed
                        advance_1.rt = _advance_1_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in MC_self_instrComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "MC_self_instr"-------
            for thisComponent in MC_self_instrComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop2.addData('MC_self_title.started', MC_self_title.tStartRefresh)
            MC_self_loop2.addData('MC_self_title.stopped', MC_self_title.tStopRefresh)
            MC_self_loop2.addData('MC_self_body.started', MC_self_body.tStartRefresh)
            MC_self_loop2.addData('MC_self_body.stopped', MC_self_body.tStopRefresh)
            MC_self_loop2.addData('MC_self_nxt.started', MC_self_nxt.tStartRefresh)
            MC_self_loop2.addData('MC_self_nxt.stopped', MC_self_nxt.tStopRefresh)
            # check responses
            if advance_1.keys in ['', [], None]:  # No response was made
                advance_1.keys = None
            MC_self_loop2.addData('advance_1.keys',advance_1.keys)
            if advance_1.keys != None:  # we had a response
                MC_self_loop2.addData('advance_1.rt', advance_1.rt)
            MC_self_loop2.addData('advance_1.started', advance_1.tStartRefresh)
            MC_self_loop2.addData('advance_1.stopped', advance_1.tStopRefresh)
            # the Routine "MC_self_instr" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_self_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_self_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_self_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_self_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_self_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_self_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_self_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_self_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_self_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_self_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_self_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_self_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "scan_wait"-------
            continueRoutine = True
            # update component parameters for each repeat
            begin_scan_btn.keys = []
            begin_scan_btn.rt = []
            _begin_scan_btn_allKeys = []
            # keep track of which components have finished
            scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
            for thisComponent in scan_waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "scan_wait"-------
            while continueRoutine:
                # get current time
                t = scan_waitClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *begin_scan_txt* updates
                if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_txt.frameNStart = frameN  # exact frame index
                    begin_scan_txt.tStart = t  # local t and not account for scr refresh
                    begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_txt.setAutoDraw(True)
                
                # *still_txt* updates
                if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt.frameNStart = frameN  # exact frame index
                    still_txt.tStart = t  # local t and not account for scr refresh
                    still_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
                    still_txt.setAutoDraw(True)
                
                # *begin_scan_btn* updates
                waitOnFlip = False
                if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_btn.frameNStart = frameN  # exact frame index
                    begin_scan_btn.tStart = t  # local t and not account for scr refresh
                    begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
                if begin_scan_btn.status == STARTED and not waitOnFlip:
                    theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
                    _begin_scan_btn_allKeys.extend(theseKeys)
                    if len(_begin_scan_btn_allKeys):
                        begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
                        begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in scan_waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "scan_wait"-------
            for thisComponent in scan_waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop2.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
            MC_self_loop2.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
            MC_self_loop2.addData('still_txt.started', still_txt.tStartRefresh)
            MC_self_loop2.addData('still_txt.stopped', still_txt.tStopRefresh)
            # check responses
            if begin_scan_btn.keys in ['', [], None]:  # No response was made
                begin_scan_btn.keys = None
            MC_self_loop2.addData('begin_scan_btn.keys',begin_scan_btn.keys)
            if begin_scan_btn.keys != None:  # we had a response
                MC_self_loop2.addData('begin_scan_btn.rt', begin_scan_btn.rt)
            MC_self_loop2.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
            MC_self_loop2.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
            # the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "self"-------
            continueRoutine = True
            routineTimer.add(307.500000)
            # update component parameters for each repeat
            self_btn.keys = []
            self_btn.rt = []
            _self_btn_allKeys = []
            skip2.keys = []
            skip2.rt = []
            _skip2_allKeys = []
            # keep track of which components have finished
            selfComponents = [self_btn, trial_crosshair, skip2]
            for thisComponent in selfComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            selfClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "self"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = selfClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=selfClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *self_btn* updates
                waitOnFlip = False
                if self_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    self_btn.frameNStart = frameN  # exact frame index
                    self_btn.tStart = t  # local t and not account for scr refresh
                    self_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(self_btn, 'tStartRefresh')  # time at next scr refresh
                    self_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(self_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(self_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if self_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > self_btn.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        self_btn.tStop = t  # not accounting for scr refresh
                        self_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(self_btn, 'tStopRefresh')  # time at next scr refresh
                        self_btn.status = FINISHED
                if self_btn.status == STARTED and not waitOnFlip:
                    theseKeys = self_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _self_btn_allKeys.extend(theseKeys)
                    if len(_self_btn_allKeys):
                        self_btn.keys = [key.name for key in _self_btn_allKeys]  # storing all keys
                        self_btn.rt = [key.rt for key in _self_btn_allKeys]
                
                # *trial_crosshair* updates
                if trial_crosshair.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_crosshair.frameNStart = frameN  # exact frame index
                    trial_crosshair.tStart = t  # local t and not account for scr refresh
                    trial_crosshair.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_crosshair, 'tStartRefresh')  # time at next scr refresh
                    trial_crosshair.setAutoDraw(True)
                if trial_crosshair.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_crosshair.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_crosshair.tStop = t  # not accounting for scr refresh
                        trial_crosshair.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(trial_crosshair, 'tStopRefresh')  # time at next scr refresh
                        trial_crosshair.setAutoDraw(False)
                
                # *skip2* updates
                waitOnFlip = False
                if skip2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    skip2.frameNStart = frameN  # exact frame index
                    skip2.tStart = t  # local t and not account for scr refresh
                    skip2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(skip2, 'tStartRefresh')  # time at next scr refresh
                    skip2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(skip2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(skip2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if skip2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > skip2.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        skip2.tStop = t  # not accounting for scr refresh
                        skip2.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(skip2, 'tStopRefresh')  # time at next scr refresh
                        skip2.status = FINISHED
                if skip2.status == STARTED and not waitOnFlip:
                    theseKeys = skip2.getKeys(keyList=['s'], waitRelease=False)
                    _skip2_allKeys.extend(theseKeys)
                    if len(_skip2_allKeys):
                        skip2.keys = _skip2_allKeys[-1].name  # just the last key pressed
                        skip2.rt = _skip2_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in selfComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "self"-------
            for thisComponent in selfComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # check responses
            if self_btn.keys in ['', [], None]:  # No response was made
                self_btn.keys = None
            MC_self_loop2.addData('self_btn.keys',self_btn.keys)
            if self_btn.keys != None:  # we had a response
                MC_self_loop2.addData('self_btn.rt', self_btn.rt)
            MC_self_loop2.addData('self_btn.started', self_btn.tStartRefresh)
            MC_self_loop2.addData('self_btn.stopped', self_btn.tStopRefresh)
            MC_self_loop2.addData('trial_crosshair.started', trial_crosshair.tStartRefresh)
            MC_self_loop2.addData('trial_crosshair.stopped', trial_crosshair.tStopRefresh)
            
            # ------Prepare to start Routine "finish_trial"-------
            continueRoutine = True
            routineTimer.add(30.000000)
            # update component parameters for each repeat
            break_btn.keys = []
            break_btn.rt = []
            _break_btn_allKeys = []
            # keep track of which components have finished
            finish_trialComponents = [break_txt, break_btn]
            for thisComponent in finish_trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            finish_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "finish_trial"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = finish_trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=finish_trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *break_txt* updates
                if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_txt.frameNStart = frameN  # exact frame index
                    break_txt.tStart = t  # local t and not account for scr refresh
                    break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                    break_txt.setAutoDraw(True)
                if break_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_txt.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_txt.tStop = t  # not accounting for scr refresh
                        break_txt.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_txt, 'tStopRefresh')  # time at next scr refresh
                        break_txt.setAutoDraw(False)
                
                # *break_btn* updates
                waitOnFlip = False
                if break_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_btn.frameNStart = frameN  # exact frame index
                    break_btn.tStart = t  # local t and not account for scr refresh
                    break_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_btn, 'tStartRefresh')  # time at next scr refresh
                    break_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(break_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(break_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if break_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_btn.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_btn.tStop = t  # not accounting for scr refresh
                        break_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_btn, 'tStopRefresh')  # time at next scr refresh
                        break_btn.status = FINISHED
                if break_btn.status == STARTED and not waitOnFlip:
                    theseKeys = break_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _break_btn_allKeys.extend(theseKeys)
                    if len(_break_btn_allKeys):
                        break_btn.keys = _break_btn_allKeys[-1].name  # just the last key pressed
                        break_btn.rt = _break_btn_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in finish_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "finish_trial"-------
            for thisComponent in finish_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_self_loop2.addData('break_txt.started', break_txt.tStartRefresh)
            MC_self_loop2.addData('break_txt.stopped', break_txt.tStopRefresh)
            # check responses
            if break_btn.keys in ['', [], None]:  # No response was made
                break_btn.keys = None
            MC_self_loop2.addData('break_btn.keys',break_btn.keys)
            if break_btn.keys != None:  # we had a response
                MC_self_loop2.addData('break_btn.rt', break_btn.rt)
            MC_self_loop2.addData('break_btn.started', break_btn.tStartRefresh)
            MC_self_loop2.addData('break_btn.stopped', break_btn.tStopRefresh)
            thisExp.nextEntry()
            
        # completed orders1[1][outer_loop2.thisN] repeats of 'MC_self_loop2'
        
        
        # set up handler to look after randomisation of conditions etc
        MC_other_loop2 = data.TrialHandler(nReps=orders1[2][outer_loop2.thisN], method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='MC_other_loop2')
        thisExp.addLoop(MC_other_loop2)  # add the loop to the experiment
        thisMC_other_loop2 = MC_other_loop2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMC_other_loop2.rgb)
        if thisMC_other_loop2 != None:
            for paramName in thisMC_other_loop2:
                exec('{} = thisMC_other_loop2[paramName]'.format(paramName))
        
        for thisMC_other_loop2 in MC_other_loop2:
            currentLoop = MC_other_loop2
            # abbreviate parameter names if possible (e.g. rgb = thisMC_other_loop2.rgb)
            if thisMC_other_loop2 != None:
                for paramName in thisMC_other_loop2:
                    exec('{} = thisMC_other_loop2[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "MC_other_instr"-------
            continueRoutine = True
            # update component parameters for each repeat
            advance4.keys = []
            advance4.rt = []
            _advance4_allKeys = []
            # keep track of which components have finished
            MC_other_instrComponents = [MC_other_title, MC_other_body, MC_other_nxt, advance4]
            for thisComponent in MC_other_instrComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            MC_other_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "MC_other_instr"-------
            while continueRoutine:
                # get current time
                t = MC_other_instrClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=MC_other_instrClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *MC_other_title* updates
                if MC_other_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_other_title.frameNStart = frameN  # exact frame index
                    MC_other_title.tStart = t  # local t and not account for scr refresh
                    MC_other_title.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_other_title, 'tStartRefresh')  # time at next scr refresh
                    MC_other_title.setAutoDraw(True)
                
                # *MC_other_body* updates
                if MC_other_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_other_body.frameNStart = frameN  # exact frame index
                    MC_other_body.tStart = t  # local t and not account for scr refresh
                    MC_other_body.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_other_body, 'tStartRefresh')  # time at next scr refresh
                    MC_other_body.setAutoDraw(True)
                
                # *MC_other_nxt* updates
                if MC_other_nxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MC_other_nxt.frameNStart = frameN  # exact frame index
                    MC_other_nxt.tStart = t  # local t and not account for scr refresh
                    MC_other_nxt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MC_other_nxt, 'tStartRefresh')  # time at next scr refresh
                    MC_other_nxt.setAutoDraw(True)
                
                # *advance4* updates
                waitOnFlip = False
                if advance4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    advance4.frameNStart = frameN  # exact frame index
                    advance4.tStart = t  # local t and not account for scr refresh
                    advance4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(advance4, 'tStartRefresh')  # time at next scr refresh
                    advance4.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(advance4.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(advance4.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if advance4.status == STARTED and not waitOnFlip:
                    theseKeys = advance4.getKeys(keyList=['space', '1'], waitRelease=False)
                    _advance4_allKeys.extend(theseKeys)
                    if len(_advance4_allKeys):
                        advance4.keys = _advance4_allKeys[-1].name  # just the last key pressed
                        advance4.rt = _advance4_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in MC_other_instrComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "MC_other_instr"-------
            for thisComponent in MC_other_instrComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop2.addData('MC_other_title.started', MC_other_title.tStartRefresh)
            MC_other_loop2.addData('MC_other_title.stopped', MC_other_title.tStopRefresh)
            MC_other_loop2.addData('MC_other_body.started', MC_other_body.tStartRefresh)
            MC_other_loop2.addData('MC_other_body.stopped', MC_other_body.tStopRefresh)
            MC_other_loop2.addData('MC_other_nxt.started', MC_other_nxt.tStartRefresh)
            MC_other_loop2.addData('MC_other_nxt.stopped', MC_other_nxt.tStopRefresh)
            # check responses
            if advance4.keys in ['', [], None]:  # No response was made
                advance4.keys = None
            MC_other_loop2.addData('advance4.keys',advance4.keys)
            if advance4.keys != None:  # we had a response
                MC_other_loop2.addData('advance4.rt', advance4.rt)
            MC_other_loop2.addData('advance4.started', advance4.tStartRefresh)
            MC_other_loop2.addData('advance4.stopped', advance4.tStopRefresh)
            # the Routine "MC_other_instr" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_other_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_other_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_other_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_other_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_other_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_other_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_other_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_other_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_other_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MC_other_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MC_other_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MC_other_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "scan_wait"-------
            continueRoutine = True
            # update component parameters for each repeat
            begin_scan_btn.keys = []
            begin_scan_btn.rt = []
            _begin_scan_btn_allKeys = []
            # keep track of which components have finished
            scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
            for thisComponent in scan_waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "scan_wait"-------
            while continueRoutine:
                # get current time
                t = scan_waitClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *begin_scan_txt* updates
                if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_txt.frameNStart = frameN  # exact frame index
                    begin_scan_txt.tStart = t  # local t and not account for scr refresh
                    begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_txt.setAutoDraw(True)
                
                # *still_txt* updates
                if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt.frameNStart = frameN  # exact frame index
                    still_txt.tStart = t  # local t and not account for scr refresh
                    still_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
                    still_txt.setAutoDraw(True)
                
                # *begin_scan_btn* updates
                waitOnFlip = False
                if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_btn.frameNStart = frameN  # exact frame index
                    begin_scan_btn.tStart = t  # local t and not account for scr refresh
                    begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
                if begin_scan_btn.status == STARTED and not waitOnFlip:
                    theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
                    _begin_scan_btn_allKeys.extend(theseKeys)
                    if len(_begin_scan_btn_allKeys):
                        begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
                        begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in scan_waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "scan_wait"-------
            for thisComponent in scan_waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop2.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
            MC_other_loop2.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
            MC_other_loop2.addData('still_txt.started', still_txt.tStartRefresh)
            MC_other_loop2.addData('still_txt.stopped', still_txt.tStopRefresh)
            # check responses
            if begin_scan_btn.keys in ['', [], None]:  # No response was made
                begin_scan_btn.keys = None
            MC_other_loop2.addData('begin_scan_btn.keys',begin_scan_btn.keys)
            if begin_scan_btn.keys != None:  # we had a response
                MC_other_loop2.addData('begin_scan_btn.rt', begin_scan_btn.rt)
            MC_other_loop2.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
            MC_other_loop2.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
            # the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "movie_nick"-------
            continueRoutine = True
            routineTimer.add(307.500000)
            # update component parameters for each repeat
            other_n_btn.keys = []
            other_n_btn.rt = []
            _other_n_btn_allKeys = []
            nick_mov2 = visual.MovieStim3(
                win=win, name='nick_mov2',
                noAudio = True,
                filename='nick_meditation.mp4',
                ori=0, pos=(0, 0), opacity=1,
                loop=False,
                depth=-1.0,
                )
            skip4.keys = []
            skip4.rt = []
            _skip4_allKeys = []
            # keep track of which components have finished
            movie_nickComponents = [other_n_btn, nick_mov2, skip4]
            for thisComponent in movie_nickComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            movie_nickClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "movie_nick"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = movie_nickClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=movie_nickClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *other_n_btn* updates
                waitOnFlip = False
                if other_n_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    other_n_btn.frameNStart = frameN  # exact frame index
                    other_n_btn.tStart = t  # local t and not account for scr refresh
                    other_n_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(other_n_btn, 'tStartRefresh')  # time at next scr refresh
                    other_n_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(other_n_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(other_n_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if other_n_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > other_n_btn.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        other_n_btn.tStop = t  # not accounting for scr refresh
                        other_n_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(other_n_btn, 'tStopRefresh')  # time at next scr refresh
                        other_n_btn.status = FINISHED
                if other_n_btn.status == STARTED and not waitOnFlip:
                    theseKeys = other_n_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _other_n_btn_allKeys.extend(theseKeys)
                    if len(_other_n_btn_allKeys):
                        other_n_btn.keys = [key.name for key in _other_n_btn_allKeys]  # storing all keys
                        other_n_btn.rt = [key.rt for key in _other_n_btn_allKeys]
                
                # *nick_mov2* updates
                if nick_mov2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    nick_mov2.frameNStart = frameN  # exact frame index
                    nick_mov2.tStart = t  # local t and not account for scr refresh
                    nick_mov2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(nick_mov2, 'tStartRefresh')  # time at next scr refresh
                    nick_mov2.setAutoDraw(True)
                if nick_mov2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > nick_mov2.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        nick_mov2.tStop = t  # not accounting for scr refresh
                        nick_mov2.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(nick_mov2, 'tStopRefresh')  # time at next scr refresh
                        nick_mov2.setAutoDraw(False)
                
                # *skip4* updates
                waitOnFlip = False
                if skip4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    skip4.frameNStart = frameN  # exact frame index
                    skip4.tStart = t  # local t and not account for scr refresh
                    skip4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(skip4, 'tStartRefresh')  # time at next scr refresh
                    skip4.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(skip4.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(skip4.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if skip4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > skip4.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        skip4.tStop = t  # not accounting for scr refresh
                        skip4.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(skip4, 'tStopRefresh')  # time at next scr refresh
                        skip4.status = FINISHED
                if skip4.status == STARTED and not waitOnFlip:
                    theseKeys = skip4.getKeys(keyList=['s'], waitRelease=False)
                    _skip4_allKeys.extend(theseKeys)
                    if len(_skip4_allKeys):
                        skip4.keys = _skip4_allKeys[-1].name  # just the last key pressed
                        skip4.rt = _skip4_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in movie_nickComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "movie_nick"-------
            for thisComponent in movie_nickComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # check responses
            if other_n_btn.keys in ['', [], None]:  # No response was made
                other_n_btn.keys = None
            MC_other_loop2.addData('other_n_btn.keys',other_n_btn.keys)
            if other_n_btn.keys != None:  # we had a response
                MC_other_loop2.addData('other_n_btn.rt', other_n_btn.rt)
            MC_other_loop2.addData('other_n_btn.started', other_n_btn.tStartRefresh)
            MC_other_loop2.addData('other_n_btn.stopped', other_n_btn.tStopRefresh)
            nick_mov2.stop()
            
            # ------Prepare to start Routine "finish_trial"-------
            continueRoutine = True
            routineTimer.add(30.000000)
            # update component parameters for each repeat
            break_btn.keys = []
            break_btn.rt = []
            _break_btn_allKeys = []
            # keep track of which components have finished
            finish_trialComponents = [break_txt, break_btn]
            for thisComponent in finish_trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            finish_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "finish_trial"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = finish_trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=finish_trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *break_txt* updates
                if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_txt.frameNStart = frameN  # exact frame index
                    break_txt.tStart = t  # local t and not account for scr refresh
                    break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                    break_txt.setAutoDraw(True)
                if break_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_txt.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_txt.tStop = t  # not accounting for scr refresh
                        break_txt.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_txt, 'tStopRefresh')  # time at next scr refresh
                        break_txt.setAutoDraw(False)
                
                # *break_btn* updates
                waitOnFlip = False
                if break_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_btn.frameNStart = frameN  # exact frame index
                    break_btn.tStart = t  # local t and not account for scr refresh
                    break_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_btn, 'tStartRefresh')  # time at next scr refresh
                    break_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(break_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(break_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if break_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_btn.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_btn.tStop = t  # not accounting for scr refresh
                        break_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_btn, 'tStopRefresh')  # time at next scr refresh
                        break_btn.status = FINISHED
                if break_btn.status == STARTED and not waitOnFlip:
                    theseKeys = break_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _break_btn_allKeys.extend(theseKeys)
                    if len(_break_btn_allKeys):
                        break_btn.keys = _break_btn_allKeys[-1].name  # just the last key pressed
                        break_btn.rt = _break_btn_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in finish_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "finish_trial"-------
            for thisComponent in finish_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MC_other_loop2.addData('break_txt.started', break_txt.tStartRefresh)
            MC_other_loop2.addData('break_txt.stopped', break_txt.tStopRefresh)
            # check responses
            if break_btn.keys in ['', [], None]:  # No response was made
                break_btn.keys = None
            MC_other_loop2.addData('break_btn.keys',break_btn.keys)
            if break_btn.keys != None:  # we had a response
                MC_other_loop2.addData('break_btn.rt', break_btn.rt)
            MC_other_loop2.addData('break_btn.started', break_btn.tStartRefresh)
            MC_other_loop2.addData('break_btn.stopped', break_btn.tStopRefresh)
            thisExp.nextEntry()
            
        # completed orders1[2][outer_loop2.thisN] repeats of 'MC_other_loop2'
        
        
        # set up handler to look after randomisation of conditions etc
        MM_other_loop2 = data.TrialHandler(nReps=orders1[3][outer_loop2.thisN], method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='MM_other_loop2')
        thisExp.addLoop(MM_other_loop2)  # add the loop to the experiment
        thisMM_other_loop2 = MM_other_loop2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMM_other_loop2.rgb)
        if thisMM_other_loop2 != None:
            for paramName in thisMM_other_loop2:
                exec('{} = thisMM_other_loop2[paramName]'.format(paramName))
        
        for thisMM_other_loop2 in MM_other_loop2:
            currentLoop = MM_other_loop2
            # abbreviate parameter names if possible (e.g. rgb = thisMM_other_loop2.rgb)
            if thisMM_other_loop2 != None:
                for paramName in thisMM_other_loop2:
                    exec('{} = thisMM_other_loop2[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "MM_other_instr"-------
            continueRoutine = True
            # update component parameters for each repeat
            advance3.keys = []
            advance3.rt = []
            _advance3_allKeys = []
            # keep track of which components have finished
            MM_other_instrComponents = [MM_other_title, MM_other_body, MM_other_nxt, advance3]
            for thisComponent in MM_other_instrComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            MM_other_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "MM_other_instr"-------
            while continueRoutine:
                # get current time
                t = MM_other_instrClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=MM_other_instrClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *MM_other_title* updates
                if MM_other_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_other_title.frameNStart = frameN  # exact frame index
                    MM_other_title.tStart = t  # local t and not account for scr refresh
                    MM_other_title.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_other_title, 'tStartRefresh')  # time at next scr refresh
                    MM_other_title.setAutoDraw(True)
                
                # *MM_other_body* updates
                if MM_other_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_other_body.frameNStart = frameN  # exact frame index
                    MM_other_body.tStart = t  # local t and not account for scr refresh
                    MM_other_body.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_other_body, 'tStartRefresh')  # time at next scr refresh
                    MM_other_body.setAutoDraw(True)
                
                # *MM_other_nxt* updates
                if MM_other_nxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    MM_other_nxt.frameNStart = frameN  # exact frame index
                    MM_other_nxt.tStart = t  # local t and not account for scr refresh
                    MM_other_nxt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MM_other_nxt, 'tStartRefresh')  # time at next scr refresh
                    MM_other_nxt.setAutoDraw(True)
                
                # *advance3* updates
                waitOnFlip = False
                if advance3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    advance3.frameNStart = frameN  # exact frame index
                    advance3.tStart = t  # local t and not account for scr refresh
                    advance3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(advance3, 'tStartRefresh')  # time at next scr refresh
                    advance3.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(advance3.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(advance3.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if advance3.status == STARTED and not waitOnFlip:
                    theseKeys = advance3.getKeys(keyList=['space'], waitRelease=False)
                    _advance3_allKeys.extend(theseKeys)
                    if len(_advance3_allKeys):
                        advance3.keys = [key.name for key in _advance3_allKeys]  # storing all keys
                        advance3.rt = [key.rt for key in _advance3_allKeys]
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in MM_other_instrComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "MM_other_instr"-------
            for thisComponent in MM_other_instrComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop2.addData('MM_other_title.started', MM_other_title.tStartRefresh)
            MM_other_loop2.addData('MM_other_title.stopped', MM_other_title.tStopRefresh)
            MM_other_loop2.addData('MM_other_body.started', MM_other_body.tStartRefresh)
            MM_other_loop2.addData('MM_other_body.stopped', MM_other_body.tStopRefresh)
            MM_other_loop2.addData('MM_other_nxt.started', MM_other_nxt.tStartRefresh)
            MM_other_loop2.addData('MM_other_nxt.stopped', MM_other_nxt.tStopRefresh)
            # check responses
            if advance3.keys in ['', [], None]:  # No response was made
                advance3.keys = None
            MM_other_loop2.addData('advance3.keys',advance3.keys)
            if advance3.keys != None:  # we had a response
                MM_other_loop2.addData('advance3.rt', advance3.rt)
            MM_other_loop2.addData('advance3.started', advance3.tStartRefresh)
            MM_other_loop2.addData('advance3.stopped', advance3.tStopRefresh)
            # the Routine "MM_other_instr" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_other_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_other_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_other_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_other_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_other_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_other_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_other_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_other_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_other_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "TR"-------
            continueRoutine = True
            # update component parameters for each repeat
            scan_inter.keys = []
            scan_inter.rt = []
            _scan_inter_allKeys = []
            # keep track of which components have finished
            TRComponents = [wait_scan_txt, still_txt2, scan_inter]
            for thisComponent in TRComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "TR"-------
            while continueRoutine:
                # get current time
                t = TRClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=TRClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *wait_scan_txt* updates
                if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    wait_scan_txt.frameNStart = frameN  # exact frame index
                    wait_scan_txt.tStart = t  # local t and not account for scr refresh
                    wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    wait_scan_txt.setAutoDraw(True)
                
                # *still_txt2* updates
                if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt2.frameNStart = frameN  # exact frame index
                    still_txt2.tStart = t  # local t and not account for scr refresh
                    still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
                    still_txt2.setAutoDraw(True)
                
                # *scan_inter* updates
                waitOnFlip = False
                if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    scan_inter.frameNStart = frameN  # exact frame index
                    scan_inter.tStart = t  # local t and not account for scr refresh
                    scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
                    scan_inter.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if scan_inter.status == STARTED and not waitOnFlip:
                    theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
                    _scan_inter_allKeys.extend(theseKeys)
                    if len(_scan_inter_allKeys):
                        scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
                        scan_inter.rt = _scan_inter_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TRComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "TR"-------
            for thisComponent in TRComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop2.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
            MM_other_loop2.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
            MM_other_loop2.addData('still_txt2.started', still_txt2.tStartRefresh)
            MM_other_loop2.addData('still_txt2.stopped', still_txt2.tStopRefresh)
            # the Routine "TR" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "scan_wait"-------
            continueRoutine = True
            # update component parameters for each repeat
            begin_scan_btn.keys = []
            begin_scan_btn.rt = []
            _begin_scan_btn_allKeys = []
            # keep track of which components have finished
            scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
            for thisComponent in scan_waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "scan_wait"-------
            while continueRoutine:
                # get current time
                t = scan_waitClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *begin_scan_txt* updates
                if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_txt.frameNStart = frameN  # exact frame index
                    begin_scan_txt.tStart = t  # local t and not account for scr refresh
                    begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_txt.setAutoDraw(True)
                
                # *still_txt* updates
                if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    still_txt.frameNStart = frameN  # exact frame index
                    still_txt.tStart = t  # local t and not account for scr refresh
                    still_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
                    still_txt.setAutoDraw(True)
                
                # *begin_scan_btn* updates
                waitOnFlip = False
                if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    begin_scan_btn.frameNStart = frameN  # exact frame index
                    begin_scan_btn.tStart = t  # local t and not account for scr refresh
                    begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
                    begin_scan_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
                if begin_scan_btn.status == STARTED and not waitOnFlip:
                    theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
                    _begin_scan_btn_allKeys.extend(theseKeys)
                    if len(_begin_scan_btn_allKeys):
                        begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
                        begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in scan_waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "scan_wait"-------
            for thisComponent in scan_waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop2.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
            MM_other_loop2.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
            MM_other_loop2.addData('still_txt.started', still_txt.tStartRefresh)
            MM_other_loop2.addData('still_txt.stopped', still_txt.tStopRefresh)
            # check responses
            if begin_scan_btn.keys in ['', [], None]:  # No response was made
                begin_scan_btn.keys = None
            MM_other_loop2.addData('begin_scan_btn.keys',begin_scan_btn.keys)
            if begin_scan_btn.keys != None:  # we had a response
                MM_other_loop2.addData('begin_scan_btn.rt', begin_scan_btn.rt)
            MM_other_loop2.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
            MM_other_loop2.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
            # the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "movie_nick"-------
            continueRoutine = True
            routineTimer.add(307.500000)
            # update component parameters for each repeat
            other_n_btn.keys = []
            other_n_btn.rt = []
            _other_n_btn_allKeys = []
            nick_mov2 = visual.MovieStim3(
                win=win, name='nick_mov2',
                noAudio = True,
                filename='nick_meditation.mp4',
                ori=0, pos=(0, 0), opacity=1,
                loop=False,
                depth=-1.0,
                )
            skip4.keys = []
            skip4.rt = []
            _skip4_allKeys = []
            # keep track of which components have finished
            movie_nickComponents = [other_n_btn, nick_mov2, skip4]
            for thisComponent in movie_nickComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            movie_nickClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "movie_nick"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = movie_nickClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=movie_nickClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *other_n_btn* updates
                waitOnFlip = False
                if other_n_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    other_n_btn.frameNStart = frameN  # exact frame index
                    other_n_btn.tStart = t  # local t and not account for scr refresh
                    other_n_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(other_n_btn, 'tStartRefresh')  # time at next scr refresh
                    other_n_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(other_n_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(other_n_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if other_n_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > other_n_btn.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        other_n_btn.tStop = t  # not accounting for scr refresh
                        other_n_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(other_n_btn, 'tStopRefresh')  # time at next scr refresh
                        other_n_btn.status = FINISHED
                if other_n_btn.status == STARTED and not waitOnFlip:
                    theseKeys = other_n_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _other_n_btn_allKeys.extend(theseKeys)
                    if len(_other_n_btn_allKeys):
                        other_n_btn.keys = [key.name for key in _other_n_btn_allKeys]  # storing all keys
                        other_n_btn.rt = [key.rt for key in _other_n_btn_allKeys]
                
                # *nick_mov2* updates
                if nick_mov2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    nick_mov2.frameNStart = frameN  # exact frame index
                    nick_mov2.tStart = t  # local t and not account for scr refresh
                    nick_mov2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(nick_mov2, 'tStartRefresh')  # time at next scr refresh
                    nick_mov2.setAutoDraw(True)
                if nick_mov2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > nick_mov2.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        nick_mov2.tStop = t  # not accounting for scr refresh
                        nick_mov2.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(nick_mov2, 'tStopRefresh')  # time at next scr refresh
                        nick_mov2.setAutoDraw(False)
                
                # *skip4* updates
                waitOnFlip = False
                if skip4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    skip4.frameNStart = frameN  # exact frame index
                    skip4.tStart = t  # local t and not account for scr refresh
                    skip4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(skip4, 'tStartRefresh')  # time at next scr refresh
                    skip4.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(skip4.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(skip4.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if skip4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > skip4.tStartRefresh + 307.5-frameTolerance:
                        # keep track of stop time/frame for later
                        skip4.tStop = t  # not accounting for scr refresh
                        skip4.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(skip4, 'tStopRefresh')  # time at next scr refresh
                        skip4.status = FINISHED
                if skip4.status == STARTED and not waitOnFlip:
                    theseKeys = skip4.getKeys(keyList=['s'], waitRelease=False)
                    _skip4_allKeys.extend(theseKeys)
                    if len(_skip4_allKeys):
                        skip4.keys = _skip4_allKeys[-1].name  # just the last key pressed
                        skip4.rt = _skip4_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in movie_nickComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "movie_nick"-------
            for thisComponent in movie_nickComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # check responses
            if other_n_btn.keys in ['', [], None]:  # No response was made
                other_n_btn.keys = None
            MM_other_loop2.addData('other_n_btn.keys',other_n_btn.keys)
            if other_n_btn.keys != None:  # we had a response
                MM_other_loop2.addData('other_n_btn.rt', other_n_btn.rt)
            MM_other_loop2.addData('other_n_btn.started', other_n_btn.tStartRefresh)
            MM_other_loop2.addData('other_n_btn.stopped', other_n_btn.tStopRefresh)
            nick_mov2.stop()
            
            # ------Prepare to start Routine "finish_trial"-------
            continueRoutine = True
            routineTimer.add(30.000000)
            # update component parameters for each repeat
            break_btn.keys = []
            break_btn.rt = []
            _break_btn_allKeys = []
            # keep track of which components have finished
            finish_trialComponents = [break_txt, break_btn]
            for thisComponent in finish_trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            finish_trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "finish_trial"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = finish_trialClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=finish_trialClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *break_txt* updates
                if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_txt.frameNStart = frameN  # exact frame index
                    break_txt.tStart = t  # local t and not account for scr refresh
                    break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                    break_txt.setAutoDraw(True)
                if break_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_txt.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_txt.tStop = t  # not accounting for scr refresh
                        break_txt.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_txt, 'tStopRefresh')  # time at next scr refresh
                        break_txt.setAutoDraw(False)
                
                # *break_btn* updates
                waitOnFlip = False
                if break_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_btn.frameNStart = frameN  # exact frame index
                    break_btn.tStart = t  # local t and not account for scr refresh
                    break_btn.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_btn, 'tStartRefresh')  # time at next scr refresh
                    break_btn.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(break_btn.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(break_btn.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if break_btn.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_btn.tStartRefresh + 30-frameTolerance:
                        # keep track of stop time/frame for later
                        break_btn.tStop = t  # not accounting for scr refresh
                        break_btn.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(break_btn, 'tStopRefresh')  # time at next scr refresh
                        break_btn.status = FINISHED
                if break_btn.status == STARTED and not waitOnFlip:
                    theseKeys = break_btn.getKeys(keyList=['space', '1'], waitRelease=False)
                    _break_btn_allKeys.extend(theseKeys)
                    if len(_break_btn_allKeys):
                        break_btn.keys = _break_btn_allKeys[-1].name  # just the last key pressed
                        break_btn.rt = _break_btn_allKeys[-1].rt
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in finish_trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "finish_trial"-------
            for thisComponent in finish_trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            MM_other_loop2.addData('break_txt.started', break_txt.tStartRefresh)
            MM_other_loop2.addData('break_txt.stopped', break_txt.tStopRefresh)
            # check responses
            if break_btn.keys in ['', [], None]:  # No response was made
                break_btn.keys = None
            MM_other_loop2.addData('break_btn.keys',break_btn.keys)
            if break_btn.keys != None:  # we had a response
                MM_other_loop2.addData('break_btn.rt', break_btn.rt)
            MM_other_loop2.addData('break_btn.started', break_btn.tStartRefresh)
            MM_other_loop2.addData('break_btn.stopped', break_btn.tStopRefresh)
            thisExp.nextEntry()
            
        # completed orders1[3][outer_loop2.thisN] repeats of 'MM_other_loop2'
        
        thisExp.nextEntry()
        
    # completed orders2[1][big_guy.thisN] repeats of 'outer_loop2'
    
    thisExp.nextEntry()
    
# completed 2 repeats of 'big_guy'


# ------Prepare to start Routine "Rest_instr"-------
continueRoutine = True
# update component parameters for each repeat
advance_0.keys = []
advance_0.rt = []
_advance_0_allKeys = []
# keep track of which components have finished
Rest_instrComponents = [Rest_title, rest_body, rest_nxt, advance_0]
for thisComponent in Rest_instrComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Rest_instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Rest_instr"-------
while continueRoutine:
    # get current time
    t = Rest_instrClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Rest_instrClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Rest_title* updates
    if Rest_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Rest_title.frameNStart = frameN  # exact frame index
        Rest_title.tStart = t  # local t and not account for scr refresh
        Rest_title.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Rest_title, 'tStartRefresh')  # time at next scr refresh
        Rest_title.setAutoDraw(True)
    
    # *rest_body* updates
    if rest_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        rest_body.frameNStart = frameN  # exact frame index
        rest_body.tStart = t  # local t and not account for scr refresh
        rest_body.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest_body, 'tStartRefresh')  # time at next scr refresh
        rest_body.setAutoDraw(True)
    
    # *rest_nxt* updates
    if rest_nxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        rest_nxt.frameNStart = frameN  # exact frame index
        rest_nxt.tStart = t  # local t and not account for scr refresh
        rest_nxt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest_nxt, 'tStartRefresh')  # time at next scr refresh
        rest_nxt.setAutoDraw(True)
    
    # *advance_0* updates
    waitOnFlip = False
    if advance_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        advance_0.frameNStart = frameN  # exact frame index
        advance_0.tStart = t  # local t and not account for scr refresh
        advance_0.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(advance_0, 'tStartRefresh')  # time at next scr refresh
        advance_0.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(advance_0.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(advance_0.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if advance_0.status == STARTED and not waitOnFlip:
        theseKeys = advance_0.getKeys(keyList=['space', '1'], waitRelease=False)
        _advance_0_allKeys.extend(theseKeys)
        if len(_advance_0_allKeys):
            advance_0.keys = _advance_0_allKeys[-1].name  # just the last key pressed
            advance_0.rt = _advance_0_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Rest_instrComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Rest_instr"-------
for thisComponent in Rest_instrComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Rest_title.started', Rest_title.tStartRefresh)
thisExp.addData('Rest_title.stopped', Rest_title.tStopRefresh)
thisExp.addData('rest_body.started', rest_body.tStartRefresh)
thisExp.addData('rest_body.stopped', rest_body.tStopRefresh)
thisExp.addData('rest_nxt.started', rest_nxt.tStartRefresh)
thisExp.addData('rest_nxt.stopped', rest_nxt.tStopRefresh)
# check responses
if advance_0.keys in ['', [], None]:  # No response was made
    advance_0.keys = None
thisExp.addData('advance_0.keys',advance_0.keys)
if advance_0.keys != None:  # we had a response
    thisExp.addData('advance_0.rt', advance_0.rt)
thisExp.addData('advance_0.started', advance_0.tStartRefresh)
thisExp.addData('advance_0.stopped', advance_0.tStopRefresh)
thisExp.nextEntry()
# the Routine "Rest_instr" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "TR"-------
continueRoutine = True
# update component parameters for each repeat
scan_inter.keys = []
scan_inter.rt = []
_scan_inter_allKeys = []
# keep track of which components have finished
TRComponents = [wait_scan_txt, still_txt2, scan_inter]
for thisComponent in TRComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "TR"-------
while continueRoutine:
    # get current time
    t = TRClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=TRClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *wait_scan_txt* updates
    if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        wait_scan_txt.frameNStart = frameN  # exact frame index
        wait_scan_txt.tStart = t  # local t and not account for scr refresh
        wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
        wait_scan_txt.setAutoDraw(True)
    
    # *still_txt2* updates
    if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        still_txt2.frameNStart = frameN  # exact frame index
        still_txt2.tStart = t  # local t and not account for scr refresh
        still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
        still_txt2.setAutoDraw(True)
    
    # *scan_inter* updates
    waitOnFlip = False
    if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        scan_inter.frameNStart = frameN  # exact frame index
        scan_inter.tStart = t  # local t and not account for scr refresh
        scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
        scan_inter.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if scan_inter.status == STARTED and not waitOnFlip:
        theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
        _scan_inter_allKeys.extend(theseKeys)
        if len(_scan_inter_allKeys):
            scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
            scan_inter.rt = _scan_inter_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in TRComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "TR"-------
for thisComponent in TRComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
thisExp.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
thisExp.addData('still_txt2.started', still_txt2.tStartRefresh)
thisExp.addData('still_txt2.stopped', still_txt2.tStopRefresh)
# the Routine "TR" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "TR"-------
continueRoutine = True
# update component parameters for each repeat
scan_inter.keys = []
scan_inter.rt = []
_scan_inter_allKeys = []
# keep track of which components have finished
TRComponents = [wait_scan_txt, still_txt2, scan_inter]
for thisComponent in TRComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "TR"-------
while continueRoutine:
    # get current time
    t = TRClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=TRClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *wait_scan_txt* updates
    if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        wait_scan_txt.frameNStart = frameN  # exact frame index
        wait_scan_txt.tStart = t  # local t and not account for scr refresh
        wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
        wait_scan_txt.setAutoDraw(True)
    
    # *still_txt2* updates
    if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        still_txt2.frameNStart = frameN  # exact frame index
        still_txt2.tStart = t  # local t and not account for scr refresh
        still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
        still_txt2.setAutoDraw(True)
    
    # *scan_inter* updates
    waitOnFlip = False
    if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        scan_inter.frameNStart = frameN  # exact frame index
        scan_inter.tStart = t  # local t and not account for scr refresh
        scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
        scan_inter.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if scan_inter.status == STARTED and not waitOnFlip:
        theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
        _scan_inter_allKeys.extend(theseKeys)
        if len(_scan_inter_allKeys):
            scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
            scan_inter.rt = _scan_inter_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in TRComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "TR"-------
for thisComponent in TRComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
thisExp.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
thisExp.addData('still_txt2.started', still_txt2.tStartRefresh)
thisExp.addData('still_txt2.stopped', still_txt2.tStopRefresh)
# the Routine "TR" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "TR"-------
continueRoutine = True
# update component parameters for each repeat
scan_inter.keys = []
scan_inter.rt = []
_scan_inter_allKeys = []
# keep track of which components have finished
TRComponents = [wait_scan_txt, still_txt2, scan_inter]
for thisComponent in TRComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "TR"-------
while continueRoutine:
    # get current time
    t = TRClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=TRClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *wait_scan_txt* updates
    if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        wait_scan_txt.frameNStart = frameN  # exact frame index
        wait_scan_txt.tStart = t  # local t and not account for scr refresh
        wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
        wait_scan_txt.setAutoDraw(True)
    
    # *still_txt2* updates
    if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        still_txt2.frameNStart = frameN  # exact frame index
        still_txt2.tStart = t  # local t and not account for scr refresh
        still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
        still_txt2.setAutoDraw(True)
    
    # *scan_inter* updates
    waitOnFlip = False
    if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        scan_inter.frameNStart = frameN  # exact frame index
        scan_inter.tStart = t  # local t and not account for scr refresh
        scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
        scan_inter.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if scan_inter.status == STARTED and not waitOnFlip:
        theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
        _scan_inter_allKeys.extend(theseKeys)
        if len(_scan_inter_allKeys):
            scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
            scan_inter.rt = _scan_inter_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in TRComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "TR"-------
for thisComponent in TRComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
thisExp.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
thisExp.addData('still_txt2.started', still_txt2.tStartRefresh)
thisExp.addData('still_txt2.stopped', still_txt2.tStopRefresh)
# the Routine "TR" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "TR"-------
continueRoutine = True
# update component parameters for each repeat
scan_inter.keys = []
scan_inter.rt = []
_scan_inter_allKeys = []
# keep track of which components have finished
TRComponents = [wait_scan_txt, still_txt2, scan_inter]
for thisComponent in TRComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
TRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "TR"-------
while continueRoutine:
    # get current time
    t = TRClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=TRClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *wait_scan_txt* updates
    if wait_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        wait_scan_txt.frameNStart = frameN  # exact frame index
        wait_scan_txt.tStart = t  # local t and not account for scr refresh
        wait_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(wait_scan_txt, 'tStartRefresh')  # time at next scr refresh
        wait_scan_txt.setAutoDraw(True)
    
    # *still_txt2* updates
    if still_txt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        still_txt2.frameNStart = frameN  # exact frame index
        still_txt2.tStart = t  # local t and not account for scr refresh
        still_txt2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(still_txt2, 'tStartRefresh')  # time at next scr refresh
        still_txt2.setAutoDraw(True)
    
    # *scan_inter* updates
    waitOnFlip = False
    if scan_inter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        scan_inter.frameNStart = frameN  # exact frame index
        scan_inter.tStart = t  # local t and not account for scr refresh
        scan_inter.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(scan_inter, 'tStartRefresh')  # time at next scr refresh
        scan_inter.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(scan_inter.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(scan_inter.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if scan_inter.status == STARTED and not waitOnFlip:
        theseKeys = scan_inter.getKeys(keyList=['equal'], waitRelease=False)
        _scan_inter_allKeys.extend(theseKeys)
        if len(_scan_inter_allKeys):
            scan_inter.keys = _scan_inter_allKeys[-1].name  # just the last key pressed
            scan_inter.rt = _scan_inter_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in TRComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "TR"-------
for thisComponent in TRComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('wait_scan_txt.started', wait_scan_txt.tStartRefresh)
thisExp.addData('wait_scan_txt.stopped', wait_scan_txt.tStopRefresh)
thisExp.addData('still_txt2.started', still_txt2.tStartRefresh)
thisExp.addData('still_txt2.stopped', still_txt2.tStopRefresh)
# the Routine "TR" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "scan_wait"-------
continueRoutine = True
# update component parameters for each repeat
begin_scan_btn.keys = []
begin_scan_btn.rt = []
_begin_scan_btn_allKeys = []
# keep track of which components have finished
scan_waitComponents = [begin_scan_txt, still_txt, begin_scan_btn]
for thisComponent in scan_waitComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
scan_waitClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "scan_wait"-------
while continueRoutine:
    # get current time
    t = scan_waitClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=scan_waitClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *begin_scan_txt* updates
    if begin_scan_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        begin_scan_txt.frameNStart = frameN  # exact frame index
        begin_scan_txt.tStart = t  # local t and not account for scr refresh
        begin_scan_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(begin_scan_txt, 'tStartRefresh')  # time at next scr refresh
        begin_scan_txt.setAutoDraw(True)
    
    # *still_txt* updates
    if still_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        still_txt.frameNStart = frameN  # exact frame index
        still_txt.tStart = t  # local t and not account for scr refresh
        still_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(still_txt, 'tStartRefresh')  # time at next scr refresh
        still_txt.setAutoDraw(True)
    
    # *begin_scan_btn* updates
    waitOnFlip = False
    if begin_scan_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        begin_scan_btn.frameNStart = frameN  # exact frame index
        begin_scan_btn.tStart = t  # local t and not account for scr refresh
        begin_scan_btn.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(begin_scan_btn, 'tStartRefresh')  # time at next scr refresh
        begin_scan_btn.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(begin_scan_btn.clock.reset)  # t=0 on next screen flip
    if begin_scan_btn.status == STARTED and not waitOnFlip:
        theseKeys = begin_scan_btn.getKeys(keyList=['equal'], waitRelease=False)
        _begin_scan_btn_allKeys.extend(theseKeys)
        if len(_begin_scan_btn_allKeys):
            begin_scan_btn.keys = [key.name for key in _begin_scan_btn_allKeys]  # storing all keys
            begin_scan_btn.rt = [key.rt for key in _begin_scan_btn_allKeys]
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in scan_waitComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "scan_wait"-------
for thisComponent in scan_waitComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('begin_scan_txt.started', begin_scan_txt.tStartRefresh)
thisExp.addData('begin_scan_txt.stopped', begin_scan_txt.tStopRefresh)
thisExp.addData('still_txt.started', still_txt.tStartRefresh)
thisExp.addData('still_txt.stopped', still_txt.tStopRefresh)
# check responses
if begin_scan_btn.keys in ['', [], None]:  # No response was made
    begin_scan_btn.keys = None
thisExp.addData('begin_scan_btn.keys',begin_scan_btn.keys)
if begin_scan_btn.keys != None:  # we had a response
    thisExp.addData('begin_scan_btn.rt', begin_scan_btn.rt)
thisExp.addData('begin_scan_btn.started', begin_scan_btn.tStartRefresh)
thisExp.addData('begin_scan_btn.stopped', begin_scan_btn.tStopRefresh)
thisExp.nextEntry()
# the Routine "scan_wait" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "Rest_scan"-------
continueRoutine = True
routineTimer.add(307.500000)
# update component parameters for each repeat
skip1.keys = []
skip1.rt = []
_skip1_allKeys = []
# keep track of which components have finished
Rest_scanComponents = [trial_crosshair_rest, skip1]
for thisComponent in Rest_scanComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Rest_scanClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Rest_scan"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = Rest_scanClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Rest_scanClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *trial_crosshair_rest* updates
    if trial_crosshair_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        trial_crosshair_rest.frameNStart = frameN  # exact frame index
        trial_crosshair_rest.tStart = t  # local t and not account for scr refresh
        trial_crosshair_rest.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(trial_crosshair_rest, 'tStartRefresh')  # time at next scr refresh
        trial_crosshair_rest.setAutoDraw(True)
    if trial_crosshair_rest.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > trial_crosshair_rest.tStartRefresh + 307.5-frameTolerance:
            # keep track of stop time/frame for later
            trial_crosshair_rest.tStop = t  # not accounting for scr refresh
            trial_crosshair_rest.frameNStop = frameN  # exact frame index
            win.timeOnFlip(trial_crosshair_rest, 'tStopRefresh')  # time at next scr refresh
            trial_crosshair_rest.setAutoDraw(False)
    
    # *skip1* updates
    waitOnFlip = False
    if skip1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        skip1.frameNStart = frameN  # exact frame index
        skip1.tStart = t  # local t and not account for scr refresh
        skip1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(skip1, 'tStartRefresh')  # time at next scr refresh
        skip1.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(skip1.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(skip1.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if skip1.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > skip1.tStartRefresh + 307.5-frameTolerance:
            # keep track of stop time/frame for later
            skip1.tStop = t  # not accounting for scr refresh
            skip1.frameNStop = frameN  # exact frame index
            win.timeOnFlip(skip1, 'tStopRefresh')  # time at next scr refresh
            skip1.status = FINISHED
    if skip1.status == STARTED and not waitOnFlip:
        theseKeys = skip1.getKeys(keyList=['s'], waitRelease=False)
        _skip1_allKeys.extend(theseKeys)
        if len(_skip1_allKeys):
            skip1.keys = _skip1_allKeys[-1].name  # just the last key pressed
            skip1.rt = _skip1_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Rest_scanComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Rest_scan"-------
for thisComponent in Rest_scanComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('trial_crosshair_rest.started', trial_crosshair_rest.tStartRefresh)
thisExp.addData('trial_crosshair_rest.stopped', trial_crosshair_rest.tStopRefresh)

# ------Prepare to start Routine "End_txt"-------
continueRoutine = True
# update component parameters for each repeat
end_press.keys = []
end_press.rt = []
_end_press_allKeys = []
# keep track of which components have finished
End_txtComponents = [end, end_press]
for thisComponent in End_txtComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
End_txtClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "End_txt"-------
while continueRoutine:
    # get current time
    t = End_txtClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=End_txtClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *end* updates
    if end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        end.frameNStart = frameN  # exact frame index
        end.tStart = t  # local t and not account for scr refresh
        end.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(end, 'tStartRefresh')  # time at next scr refresh
        end.setAutoDraw(True)
    
    # *end_press* updates
    waitOnFlip = False
    if end_press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        end_press.frameNStart = frameN  # exact frame index
        end_press.tStart = t  # local t and not account for scr refresh
        end_press.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(end_press, 'tStartRefresh')  # time at next scr refresh
        end_press.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(end_press.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(end_press.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if end_press.status == STARTED and not waitOnFlip:
        theseKeys = end_press.getKeys(keyList=['space', '1'], waitRelease=False)
        _end_press_allKeys.extend(theseKeys)
        if len(_end_press_allKeys):
            end_press.keys = _end_press_allKeys[-1].name  # just the last key pressed
            end_press.rt = _end_press_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in End_txtComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "End_txt"-------
for thisComponent in End_txtComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('end.started', end.tStartRefresh)
thisExp.addData('end.stopped', end.tStopRefresh)
# check responses
if end_press.keys in ['', [], None]:  # No response was made
    end_press.keys = None
thisExp.addData('end_press.keys',end_press.keys)
if end_press.keys != None:  # we had a response
    thisExp.addData('end_press.rt', end_press.rt)
thisExp.addData('end_press.started', end_press.tStartRefresh)
thisExp.addData('end_press.stopped', end_press.tStopRefresh)
thisExp.nextEntry()
# the Routine "End_txt" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
