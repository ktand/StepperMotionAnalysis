import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
import numpy as np

from scipy.interpolate import interp1d
from scipy import signal

filename = 'analysis.csv'

axes = {
  'X' : { 
    'index': 0,
    'mm_per_step': 1/160.0,
    'visible': True,
    'result': None
    },
  'Y' : { 
    'index': 1,
    'mm_per_step': 1/160.0,
    'visible': True,
    'result': None
    },
  'Z' : { 
    'index': 2,
    'mm_per_step': 1/2133.33,
    'visible': True,
    'result': None
    }
}

visible_axes = {k:v for k,v in axes.iteritems() if v['visible']}

show_motion = True

fs = 5000.0  # (Re-)sampling frequency
fc = 10.0 # Cut-off frequency of the filter

w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(1, w, 'low')

def interpolate(x, y, t):
  return interp1d(x, y, kind='zero', bounds_error=False, fill_value='extrapolate')(t)

def filter(i):
  return signal.filtfilt(b, a, i)

def velocity(x, y):
  return np.append([0], np.diff(y)*fs) 

def acceleration(x, y):
  v = np.append([0], np.diff(y)*fs)
  a = np.append([0], np.diff(v)*fs)
  return a

def calc_axis(axis_data, time, axis_name):
  t,p,s,n = zip(*axis_data)
  
  pos = interpolate(t, p, time)

  pos_filtered = filter(pos)
  vel = velocity(t, pos_filtered)
  acc = acceleration(t, pos_filtered)

  if len(s) > 0:
    print '[Axis {}]: Pulse width (min, avg, max): {:.3f}us, {:.3f}us, {:.3f}us'.format(axis_name, np.min(s)*1000*1000, np.mean(s)*1000*1000, np.max(s)*1000*1000)

  steps = []
  for step_time, step_width, step_dir in zip(t, s, n):
    steps.append([step_time, step_dir])
    steps.append([step_time+step_width, 0])

  return { 'pos': pos, 'pos_filtered': pos_filtered, 'vel': vel, 'acc': acc, 'steps': np.array(steps) }

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

print 'Loading dataset...'
data = pd.read_csv(filename, delimiter = ',', header=0).get_values()
print 'Sample count =', len(data)

time_start = data[0][0]
time_end = data[-1][0]

reg_t = np.linspace(time_start, time_end, round((time_end - time_start) / (1/fs) + 1))
print 'Interpolated steps =', len(reg_t)

for axis_name, axis in visible_axes.iteritems():
  axis['result'] = calc_axis(extract_data(data, 0, axis['index']*2+1, axis['index']*2+2, axis['mm_per_step']), reg_t, axis_name)

can_show_motion = show_motion and axes['X']['result'] and axes['Y']['result']

rows = 4
cols = (3 if can_show_motion else 2)

figure = plt.figure('Stepper Motion analys')

# POSITION
plot_pos = plt.subplot2grid((rows, cols), (0, 0), colspan = 2)
plot_pos.set_ylabel('position ($mm$)')
plot_pos.set_xlabel('time ($s$)')
plot_pos.legend()
plot_pos.grid(True)

# VELOCITY
plot_vel = plt.subplot2grid((rows, cols), (1, 0), colspan = 2, sharex=plot_pos)
plot_vel.set_ylabel('velocity ($mm/s$)')
plot_vel.set_xlabel('time ($s$)')
plot_vel.legend()
plot_vel.grid(True)

# ACCELERATION
plot_acc = plt.subplot2grid((rows, cols), (2, 0), colspan = 2, sharex=plot_pos)
plot_acc.set_ylabel('acceleration ($mm/s^2$)')
plot_acc.set_xlabel('time ($s$)')
plot_acc.legend()
plot_acc.grid(True)

# STEPS
plot_steps = plt.subplot2grid((rows, cols), (3, 0), colspan = 2, sharex=plot_pos)
plot_steps.set_ylim(-2, 2)
plot_steps.set_ylabel('steps')
plot_steps.set_xlabel('time ($s$)')
plot_steps.legend()
plot_steps.grid(True)

# Plot motion
if can_show_motion:
  plot_graph = plt.subplot2grid((rows, cols), (1, 2), colspan = 1, rowspan = 2)
  plt.step(axes['X']['result']['pos'], axes['Y']['result']['pos'])
  plot_graph.set_ylabel('motion ($y$)')
  plot_graph.set_xlabel('motion ($x$)')

# Plot visible axes
for axis_name, axis in visible_axes.iteritems():
    plot_pos.step(reg_t, axis['result']['pos'], label=axis_name)  
    plot_vel.plot(reg_t, axis['result']['vel'], label=axis_name)
    plot_acc.plot(reg_t, axis['result']['acc'], label=axis_name)
    plot_steps.step(axis['result']['steps'][:,0], axis['result']['steps'][:,1], label=axis_name, where='post')

plt.subplots_adjust(hspace=0.38, wspace=0.15, left=0.05, top=0.95, bottom=0.05, right=0.98)
plt.show()
