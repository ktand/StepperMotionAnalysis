#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import numpy as np

from scipy.interpolate import interp1d
from scipy import signal
from matplotlib.widgets import MultiCursor

filename = 'analysis.csv'

axes = {
  'X' : { 
    'index': 0,
    'mm_per_step': 1/100.0,    
    'visible': True,
    'result': dict()
    },
  'Y' : { 
    'index': 1,
    'mm_per_step': 1/100.0,
    'visible': True,
    'result': dict()
    },
  'Z' : { 
    'index': 2,
    'mm_per_step': 1/400,
    'visible': True,
    'result': dict()
    }
}

visible_axes = {k:v for k,v in axes.items() if v['visible']}

show_motion = True

fs = 5000.0  # (Re-)sampling frequency

def filter(i, fs, fc):
  w = fc / (fs / 2) # Normalize the frequency
  b, a = signal.butter(3, w, 'low')
  return signal.filtfilt(b, a, i)

def savgol_filter(i):
  return signal.savgol_filter(i, 3, 1)

def velocity(x, y):
  return np.append([0], np.diff(y)*fs) 

def acceleration(x, y):
  v = np.append([0], np.diff(y)*fs)
  a = np.append([0], np.diff(v)*fs)
  return a

def maxabs(a):
  return max(np.max(a), abs(np.min(a)))

def calc_axis(axis_data, time, axis_name, time_start, time_end, fs):
  t,p,s,n = zip(*axis_data)
  
  pos = interp1d(t, p, kind='nearest', bounds_error=False, fill_value='extrapolate')(time)  # OK

  vel = velocity(t, pos)
  acc = acceleration(t, pos)

  vel_filtered = filter(vel, fs, 200)
  acc_filtered = filter(acc, fs, 50)

  vel_max = maxabs(vel_filtered)
  acc_max = maxabs(acc_filtered)

  pulse_min = np.min(s)*1000*1000
  pulse_avg = np.mean(s)*1000*1000
  pulse_max = np.max(s)*1000*1000
  
  if len(s) > 0:
    print ('[Axis {}]: Pulse width (min, avg, max): {:.3f}us, {:.3f}us, {:.3f}us. Max velocity = {:f}. Max acceleration = {:f}'.format(axis_name, pulse_min, pulse_avg, pulse_max, vel_max, acc_max))

  steps = []
  for step_time, step_width, step_dir in zip(t, s, n):
    steps.append([step_time, step_dir])
    steps.append([step_time+step_width, 0])

  return { 'pos': pos, 'vel': vel, 'vel_filtered': vel_filtered, 'acc': acc, 'acc_filtered': acc_filtered, 'steps': np.array(steps), 'vel_max': vel_max, 'acc_max': acc_max, 'pulse_min': pulse_min, 'pulse_avg': pulse_avg, 'pulse_max' : pulse_max }

def extract_data(data, timeIdx, stepIdx, dirIdx, mm_per_step):
    position = 0
    pulseStart = None

    for (t, s, d) in zip(data[:,timeIdx], data[:,stepIdx], data[:,dirIdx]):        
      if s:
        if not pulseStart:
          pulseStart = t
          step = -1 if d else 1
          position += mm_per_step * step
      else:
        if pulseStart != None:
          yield (pulseStart, position, t-pulseStart, step)
          pulseStart = None

print ('Loading dataset...')
dtype_dict = {'Time[s]':'float64','X STEP':'bool', 'X DIR':'bool','Y STEP':'bool','Y DIR':'bool','Z STEP':'bool','Z DIR':'bool'}

dataFrame = pd.read_csv(filename, delimiter = ',', skipinitialspace=True, header=0)

data = dataFrame.get_values()

print ('Sample count =', len(data))

time_start = data[0][0]
time_end = data[-1][0]

reg_t = np.linspace(time_start, time_end, int((time_end - time_start) // (1/fs) + 1))
print ('Interpolated steps =', len(reg_t))

for axis_name, axis in visible_axes.items():
  axis['result'] = calc_axis(extract_data(data, 0, axis['index']*2+1, axis['index']*2+2, axis['mm_per_step']), reg_t, axis_name, time_start, time_end, fs)

can_show_motion = show_motion and axes['X']['result'] and axes['Y']['result']

rows = 4
cols = (3 if can_show_motion else 2)

figure = plt.figure('Stepper Motion analys')

# POSITION
plot_pos = plt.subplot2grid((rows, cols), (0, 0), colspan = 2)
plot_pos.set_ylabel('position ($mm$)')
plot_pos.set_xlabel('time ($s$)')


# VELOCITY
plot_vel = plt.subplot2grid((rows, cols), (1, 0), colspan = 2, sharex=plot_pos)
plot_vel.set_ylabel('velocity ($mm/s$)')
plot_vel.set_xlabel('time ($s$)')


# ACCELERATION
plot_acc = plt.subplot2grid((rows, cols), (2, 0), colspan = 2, sharex=plot_pos)
plot_acc.set_ylabel('acceleration ($mm/s^2$)')
plot_acc.set_xlabel('time ($s$)')


# STEPS
plot_steps = plt.subplot2grid((rows, cols), (3, 0), colspan = 2, sharex=plot_pos)
plot_steps.set_ylim(-2, 2)
plot_steps.set_ylabel('steps')
plot_steps.set_xlabel('time ($s$)')


# Plot motion
if can_show_motion:
  plot_graph = plt.subplot2grid((rows, cols), (1, 2), colspan = 1, rowspan = 2)
  plt.step(axes['X']['result']['pos'], axes['Y']['result']['pos'])
  plot_graph.set_ylabel('motion ($y$)')
  plot_graph.set_xlabel('motion ($x$)')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)

# Plot visible axes
for axis_name, axis in visible_axes.items():
    result = axis['result']
    plot_pos.step(reg_t, result['pos'], label=axis_name)  
    # plot_vel.plot(reg_t, result['vel'], label=axis_name)
    plot_vel.plot(reg_t, result['vel_filtered'], label=axis_name)
    # plot_acc.plot(reg_t, result['acc'], label=axis_name)
    plot_acc.plot(reg_t, result['acc_filtered'], label=axis_name)
    plot_steps.step(result['steps'][:,0], result['steps'][:,1], label=axis_name, where='post')

plot_pos.legend()
plot_pos.grid(True)

plot_vel.legend()
plot_vel.grid(True)

plot_acc.legend()
plot_acc.grid(True)

plot_steps.legend()
plot_steps.grid(True)

plt.subplots_adjust(hspace=0.38, wspace=0.15, left=0.05, top=0.95, bottom=0.05, right=0.98)

multi = MultiCursor(figure.canvas, (plot_pos, plot_vel, plot_acc, plot_steps), useblit=True, color='r', lw=1)

plt.show()
