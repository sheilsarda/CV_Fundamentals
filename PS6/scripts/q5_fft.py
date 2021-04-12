import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift

def sine_wave(f,overSampRate,phase,nCyl):
	"""
	Generate sine wave signal with the following parameters
	Parameters:
		f : frequency of sine wave in Hertz
		overSampRate : oversampling rate (integer)
		phase : desired phase shift in radians
		nCyl : number of cycles of sine wave to generate
	Returns:
		(t,g) : time base (t) and the signal g(t) as tuple
	Example:
		f=10; overSampRate=30;
		phase = 1/3*np.pi;nCyl = 5;
		(t,g) = sine_wave(f,overSampRate,phase,nCyl)
	"""
	fs = overSampRate*f # sampling frequency
	t = np.arange(0,nCyl*1/f-1/fs,1/fs) # time base
	g = np.sin(2*np.pi*f*t+phase) # replace with cos if a cosine wave is desired
	return (t,g) # return time base and signal g(t) as tuple

def box_wave(f,overSampRate,phase,nCyl):
	"""
	Generate sine wave signal with the following parameters
	Parameters:
		f : frequency of sine wave in Hertz
		overSampRate : oversampling rate (integer)
		phase : desired phase shift in radians
		nCyl : number of cycles of sine wave to generate
	Returns:
		(t,g) : time base (t) and the signal g(t) as tuple
	Example:
		f=10; overSampRate=30;
		phase = 1/3*np.pi;nCyl = 5;
		(t,g) = sine_wave(f,overSampRate,phase,nCyl)
	"""
	fs = overSampRate*f # sampling frequency
	t = np.arange(0,nCyl*1/f-1/fs,1/fs) # time base
	g = np.cos(2*np.pi*f*t+phase) # replace with cos if a cosine wave is desired
	return (t,np.fmax(np.sign(g), 0.0)) # return time base and signal g(t) as tuple

def cos_wave(f,overSampRate,phase,nCyl):
	"""
	Generate sine wave signal with the following parameters
	Parameters:
		f : frequency of sine wave in Hertz
		overSampRate : oversampling rate (integer)
		phase : desired phase shift in radians
		nCyl : number of cycles of sine wave to generate
	Returns:
		(t,g) : time base (t) and the signal g(t) as tuple
	Example:
		f=10; overSampRate=30;
		phase = 1/3*np.pi;nCyl = 5;
		(t,g) = sine_wave(f,overSampRate,phase,nCyl)
	"""
	fs = overSampRate*f # sampling frequency
	t = np.arange(0,nCyl*1/f-1/fs,1/fs) # time base
	g = np.cos(2*np.pi*f*t+phase) # replace with cos if a cosine wave is desired
	return (t,g) # return time base and signal g(t) as tuple


"""
Parameters for function to model
"""
nCyl = 1.5 # desired number of cycles of the sine wave
f = nCyl/128 #frequency = 10 Hz
overSampRate = 20 #oversammpling rate
fs = f*overSampRate #sampling frequency
phase = np.pi * 0.65 # 1/3*np.pi #phase shift in radians

"""
Select function generator
"""
# (t,x) = cos_wave(f,overSampRate,phase,nCyl) #function call
# (t,x) = sine_wave(f,overSampRate,phase,nCyl) #function call
(t,x) = box_wave(f,overSampRate,phase,nCyl) #function call

"""
Uncomment this section to plot the raw function
"""
# plt.plot(t,x) # plot using pyplot library from matplotlib package
# plt.title('Sine wave f='+str(f)+' Hz') # plot title
# plt.xlabel('Time (s)') # x-axis label
# plt.ylabel('Amplitude') # y-axis label
# plt.show() # display the figure
# exit()


"""
This section does the FFT and FFTshift
"""
NFFT=1024     
X=fftshift(fft(x,NFFT))
fig4, ax = plt.subplots(nrows=1, ncols=1) #create figure handle
fVals=np.arange(start = -NFFT/2,stop = NFFT/2)*fs/NFFT
ax.plot(fVals,np.abs(X),'b')
ax.set_title('Double Sided FFT - with FFTShift')
ax.set_xlabel('Frequency (Hz)')         
ax.set_ylabel('|DFT Values|')
ax.set_xlim(-0.25,0.25)
ax.set_xticks(np.arange(-0.25, 0.25,10))
fig4.show()
plt.show() # display the figure
