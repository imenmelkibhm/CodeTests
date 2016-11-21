# Load the required libraries:
#   * scipy
#   * numpy
#   * matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import sys


def show_info(aname, a, samplerate):
    print "Array", aname
    print "shape", a.shape
    print "dtype", a.dtype
    print "min,max:", a.min(), a.max()
    print "length ", len(a)
    print "samplerate", samplerate
    print "time ", len(a)/float(samplerate)


# Load the data and calculate the time of each sample
samplerate, data = wavfile.read('/opt/exe/textocr/demo/output-audio.wav')
#samplerate, data = wavfile.read('/home/imen/Downloads/a2002011001-e02.wav')
show_info("data", data, samplerate)
times = np.arange(len(data))/float(samplerate)
#sys.exit(0)


# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))
#plt.fill_between(times, data[:,0],data[:,1], color='k')
plt.plot(times, data, color='k')
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.savefig('plot.png', dpi=50)
plt.show()