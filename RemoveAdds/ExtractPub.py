import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import subprocess
import argparse
import time
import os


def extract_audio_signal(args):
    chunckname_wextension=os.path.basename(args.input)
    chunckname=chunckname_wextension.split('.')[0]
    subprocess.call('ffmpeg -i ' + args.input + ' /opt/exe/textocr/demo/Audio/' + chunckname +'.wav', shell=True)

def Process_Audio_Signal(args):
    chunckname_wextension=os.path.basename(args.input)
    chunckname=chunckname_wextension.split('.')[0]
    spf = wave.open('/opt/exe/textocr/demo/Audio/' + chunckname +'.wav','r')

    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')

    np.savetxt(signal, '/opt/exe/textocr/demo/Audio/' + chunckname +'.txt' )


    #If Stereo
    if spf.getnchannels() == 2:
        print 'Just mono files'
        sys.exit(0)

    plt.figure(1)
    plt.title('Signal Wave...')
    plt.plot(signal)
    plt.show()


def main():

    parser = argparse.ArgumentParser(description='Extract the audio signal of a video')
    parser.add_argument("-i", "--input", help="input video file")

    args = parser.parse_args()

    starttime = time.time()

    extract_audio_signal(args)
    Process_Audio_Signal(args)

    endtime = time.time()
    logging.warning('Processing the video in %f' % (endtime-starttime))
    sys.exit(0)

if __name__ == "__main__":

    main()
