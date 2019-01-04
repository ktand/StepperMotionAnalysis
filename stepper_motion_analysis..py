import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
import numpy as np

from scipy.interpolate import interp1d
from scipy import signal

filename = 'analysis.csv'

x_visible = True
y_visible = True
z_visible = True
motion_visible = True

x_mm_per_step = 1/160.0
y_mm_per_step = 1/160.0
z_mm_per_step = 1/2133.33

fs = 500.0  # (Re-)sampling frequency
fc = 10.0 # Cut-off frequency of the filter

w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(5, w, 'low')

def interpolate(x, y, t):
  i = interp1d(x, y, kind='zero', bounds_error=False, fill_value='extrapolate', assume_sorted=True)(t)

  return signal.filtfilt(b, a, i)

def velocity(x, y):
  return np.append([0], np.diff(y)*fs) 

def acceleration(x, y):
  v = np.append([0], np.diff(y)*fs)
  a = np.append([0], np.diff(v)*fs)
  return a

print 'Loading dataset...'
data = pd.read_csv(filename, delimiter = ',', header=0).get_values()
print 'Sample count =', len(data)

time_start = data[0][0]
time_end = data[-1][0]

#reg_t = np.arange(times[0], times[-1], 1/fs)
reg_t = np.linspace(time_start, time_end, round((time_end - time_start) / (1/fs) + 1))
print 'Interpolated steps =', len(reg_t)

x_position = 0.0
y_position = 0.0
z_position = 0.0

x_high = 0
y_high = 0
z_high = 0

x_steps = [[time_start, 0]]
y_steps = [[time_start, 0]]
z_steps = [[time_start, 0]]

x_positions = [[time_start, 0]]
y_positions = [[time_start, 0]]
z_positions = [[time_start, 0]]

print 'Detecting movements...'

for time, xs, xd, ys, yd, zs, zd in data:
  if xs:
    if x_high == 0:
      x_high = 1
      x_position += x_mm_per_step * (-1 if xd else 1) 
      x_steps.append([time, -1 if xd else 1])
      x_positions.append([time, x_position])
  else:
    if x_high == 1:
      x_steps.append([time, 0])
      x_high = 0
  
  if ys:
    if y_high == 0:
      y_high = 1
      y_position += y_mm_per_step * (-1 if yd else 1) 
      y_steps.append([time, -1 if yd else 1])
      y_positions.append([time, y_position])
  else:
    if y_high == 1:
      y_steps.append([time, 0])
      y_high = 0

  if zs:
    if z_high == 0:
      z_high = 1
      z_position += z_mm_per_step * (1 if zd else -1) 
      z_steps.append([time, 1 if zd else -1])
      z_positions.append([time, z_position])
  else:
    if z_high == 1:
      z_steps.append([time, 0])
      z_high = 0

x_steps = np.array(x_steps)
y_steps = np.array(y_steps)
z_steps = np.array(z_steps)

x_positions = np.array(x_positions)
y_positions = np.array(y_positions)
z_positions = np.array(z_positions)

if x_visible:
  print 'Calculating X-axis...'
  pos_x = interpolate(x_positions[:,0], x_positions[:,1], reg_t)
  vel_x = velocity(reg_t, pos_x)
  acc_x = acceleration(reg_t, pos_x)

if y_visible:
  print 'Calculating Y-axis...'
  pos_y = interpolate(y_positions[:,0], y_positions[:,1], reg_t)
  vel_y = velocity(reg_t, pos_y)
  acc_y = acceleration(reg_t, pos_y)

if z_visible:
  print 'Calculating Z-axis...'
  pos_z = interpolate(z_positions[:,0], z_positions[:,1], reg_t)
  vel_z = velocity(reg_t, pos_z)
  acc_z = acceleration(reg_t, pos_z)

rows = 4
cols = (3 if motion_visible else 2)

plt.figure('Stepper Motion analys')

# POSITION
print 'Plotting position...'
plot_pos = plt.subplot2grid((rows, cols), (0, 0), colspan = 2)
if x_visible:
  plt.plot(reg_t, pos_x, label='x')  
if y_visible:
  plt.plot(reg_t, pos_y, label='y')
if z_visible:
  plt.plot(reg_t, pos_z, label='z')
plot_pos.set_ylabel('position ($mm$)')
plot_pos.set_xlabel('time ($s$)')
plt.legend()
plt.grid(True)

# VELOCITY
print 'Plotting velocity...'
plot_vel = plt.subplot2grid((rows, cols), (1, 0), colspan = 2)
if x_visible:
  plt.plot(reg_t, vel_x, label='x')
if y_visible:
  plt.plot(reg_t, vel_y, label='y')
if z_visible:
  plt.plot(reg_t, vel_z, label='z')
plot_vel.set_ylabel('velocity ($mm/s$)')
plot_vel.set_xlabel('time ($s$)')
plt.legend()
plt.grid(True)

# ACCELERATION
print 'Plotting acceleration...'
plot_acc = plt.subplot2grid((rows, cols), (2, 0), colspan = 2, sharex=plot_pos)
if x_visible:
  plt.plot(reg_t, acc_x, label='x')
if y_visible:
  plt.plot(reg_t, acc_y, label='y')
if z_visible:
  plt.plot(reg_t, acc_z, label='z')
plot_acc.set_ylabel('acceleration ($mm/s^2$)')
plot_acc.set_xlabel('time ($s$)')
plt.legend()
plt.grid(True)

# STEPS
print 'Plotting steps...'
plot_steps = plt.subplot2grid((rows, cols), (3, 0), colspan = 2, sharex=plot_pos)
plot_steps.set_ylim(-2, 2)
if x_visible:
  plt.step(x_steps[:,0], x_steps[:,1], label='x', where='post')
if y_visible:
  plt.step(y_steps[:,0], y_steps[:,1], label='y', where='post')
if z_visible:
  plt.step(z_steps[:,0], z_steps[:,1], label='x', where='post')
plot_steps.set_ylabel('steps')
plot_steps.set_xlabel('time ($s$)')
plt.legend()
plt.grid(True)

# STEPS
if motion_visible:
  print 'Plotting motion...'
  plot_graph = plt.subplot2grid((rows, cols), (1, 2), colspan = 1, rowspan = 2)
  if x_visible and y_visible:
    plt.plot(pos_x, pos_y)
  # plot_graph.adjustable = 'box'
  # plot_graph.num = 1
  plot_graph.set_ylabel('motion ($y$)')
  plot_graph.set_xlabel('motion ($x$)')

print 'Showing plot...'
plt.subplots_adjust(hspace=0.38, wspace=0.15, left=0.05, top=0.95, bottom=0.05, right=0.98)
plt.show()