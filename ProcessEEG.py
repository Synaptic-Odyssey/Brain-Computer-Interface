import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from scipy import signal
import sys
import time

class EEGVisualizer:
    def __init__(self):
        # Constants
        self.timeScale = 50
        self.normalScale = 50
        self.alphaScale = 100
        self.freqAvgScale = 50
        self.alphaCenter = 12
        self.alphaBandwidth = 2
        self.betaCenter = 24
        self.betaBandwidth = 2
        
        # Data buffers
        self.timeLength = 240
        self.timeSignal = np.zeros(self.timeLength)
        self.fs = 256  # Sampling frequency
        
        # Setup FFT
        self.fft_size = 256
        self.freqs = np.fft.rfftfreq(self.fft_size, 1/self.fs)
        
        # Filters
        self.notch_freq = 60
        self.notch_filter = self.create_notch_filter(self.notch_freq, self.fs)
        self.alpha_filter = self.create_bandpass(self.alphaCenter, self.alphaBandwidth, self.fs)
        self.beta_filter = self.create_bandpass(self.betaCenter, self.betaBandwidth, self.fs)
        
        # Visualization setup
        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget(title="EEG Visualizer")
        self.win.resize(1000, 800)
        
        # Time domain plot
        self.timePlot = self.win.addPlot(title="Time Domain")
        self.timeCurve = self.timePlot.plot(pen='w')
        
        # Frequency domain plot
        self.win.nextRow()
        self.freqPlot = self.win.addPlot(title="Frequency Domain")
        self.freqBars = pg.BarGraphItem(x=self.freqs[:30], height=np.zeros(30), width=0.5, brush='b')
        self.freqPlot.addItem(self.freqBars)
        
        # Brainwave averages plot
        self.win.nextRow()
        self.avgPlot = self.win.addPlot(title="Brainwave Averages")
        self.avgBars = pg.BarGraphItem(x=np.arange(6), height=np.zeros(6), width=0.5)
        self.avgPlot.addItem(self.avgBars)
        
        # Color bars for brainwaves
        colors = ['b', 'm', 'r', 'c', 'y', 'g']  # delta to high beta
        for i, color in enumerate(colors):
            self.avgBars.opts['brushes'][i] = pg.mkBrush(color)
        
        # Audio setup
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.fs,
            callback=self.audio_callback,
            blocksize=self.timeLength
        )
        
        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)  # ~30fps
    
    def create_notch_filter(self, notch_freq, fs, quality=30):
        nyq = 0.5 * fs
        freq = notch_freq / nyq
        b, a = signal.iirnotch(freq, quality)
        return (b, a)
    
    def create_bandpass(self, center, bandwidth, fs):
        low = (center - bandwidth/2) / (0.5 * fs)
        high = (center + bandwidth/2) / (0.5 * fs)
        b, a = signal.butter(4, [low, high], btype='band')
        return (b, a)
    
    def apply_filter(self, data, filter_coeffs):
        b, a = filter_coeffs
        return signal.lfilter(b, a, data)
    
    def audio_callback(self, indata, frames, time, status):
        self.timeSignal = indata[:, 0]
    
    def update(self):
        # Process data
        if len(self.timeSignal) == 0:
            return
            
        # Apply notch filter
        filtered = self.apply_filter(self.timeSignal, self.notch_filter)
        
        # Compute FFT
        fft_data = np.abs(np.fft.rfft(filtered, n=self.fft_size))
        
        # Update plots
        self.timeCurve.setData(filtered * self.timeScale)
        self.freqBars.setOpts(height=fft_data[:30])
        
        # Calculate brainwave averages
        bands = [
            (0, 4),    # Delta
            (4, 8),    # Theta
            (8, 12),   # Alpha
            (12, 20),  # Low Beta
            (20, 30),  # High Beta
            (30, 45)   # Gamma
        ]
        avg_heights = []
        for low, high in bands:
            mask = (self.freqs >= low) & (self.freqs <= high)
            avg_heights.append(np.mean(fft_data[mask]))
        
        self.avgBars.setOpts(height=avg_heights)
    
    def run(self):
        self.win.show()
        self.stream.start()
        self.app.exec_()

if __name__ == "__main__":
    visualizer = EEGVisualizer()
    visualizer.run()