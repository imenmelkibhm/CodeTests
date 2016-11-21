#Exemple: python ExtractAdds.py -i /opt/exe/textocr/demo/LCI_20161017_18062014_18112014_SD.ts -o /opt/exe/textocr/demo/AddsFrames_LCI -f 12 -b 190 -e 194

import logging
import os
import numpy as np
import cv2
import sys
import argparse
import pylab as pl
import time
import math
import operator
#Read the video file
#vinput = '/opt/exe/textocr/demo/iTele_20161017_18244715_18294715_SD.ts'
#add_frame =


#Get and store the wanted frames
def extract_adds_ref_images(args, vinput, outputpath, bese, endse, fintv):

    cap = cv2.VideoCapture(vinput)
    logging.warning('Opening the video for frames dumping')

    fps = cap.get(5)  # frame per second in video
    befr = int(bese * fps)  # begining frame
    endfr = int(endse * fps)  # ending frame

    if cap.isOpened():  # if video is opened, process frames
        for i in xrange(befr, endfr, int(np.round(fps / fintv))):
            # 1.----------read the specific frames in the video-----------
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # get the specific frames
            #logging.warning('Read the %d frame' % i)
            ret, frame = cap.read()
            #2.-----------store the read image
            if ret == True:
                cv2.imwrite(os.path.join(outputpath, args.outputname + "_frame%d.png" % i),frame) # save the frame as a JPEG File


def dump_GT(args, vinput, outputpath, bese, endse, fintv):

    cap = cv2.VideoCapture(vinput)
    logging.warning('Opening the video for ground truth dumping')

    fps = cap.get(5)  # frame per second in video
    befr = int(bese * fps)  # begining frame
    endfr = int(endse * fps)  # ending frame
    frn = int(cap.get(7))  # frame number
    ground_truth = []
    frames =[]
    if cap.isOpened():  # if video is opened, process frames
        for i in xrange(0, frn, int(np.round(fps / fintv))):
            print('i = %d' %i)
            print('i = %d' %i + '/ %d' %frn)
            if i>=befr and i<endfr:
                ground_truth.append(1)
                frames.append(i)
            else:
                ground_truth.append(0)
                frames.append(i)

    pl.figure(figsize=(30, 4))
    pl.plot(frames,ground_truth, 'g')
    pl.xlabel('time (s)')
    pl.ylabel('amplitude')

    if int(args.pubs) == 1 :
        pl.savefig(os.path.join(outputpath,  args.outputname + "_Pub_GroundTruth.jpg"), dpi=50)
        np.savetxt(os.path.join(outputpath,  args.outputname + "_Pub_GroundTruth.txt"), ground_truth)
    else:
        pl.savefig(os.path.join(outputpath,  args.outputname + "_Meteo_GroundTruth.jpg"), dpi=50)
        np.savetxt(os.path.join(outputpath,  args.outputname + "_Meteo_GroundTruth.txt"), ground_truth)


#the main function
def main():
    #Read input arguments
    parser = argparse.ArgumentParser(description='Extract the adds frames')
    parser.add_argument("-i", "--input", help="input video file")
    parser.add_argument("-a", "--action", help="action to do: (1) To dump the adds announcement frames (2) To dumps the adds ground truth file")
    parser.add_argument("-p", "--pubs", help="want to extract adds (0) or meteo (1)")
    parser.add_argument("-o", "--outputpath", help="the path to save the adds frames")
    parser.add_argument("-f", "--frequency", default=1.0, help="the frequency of extracting and processing video frames")
    parser.add_argument("-n", "--outputname", help="the name to save the adds frames")
    parser.add_argument("-b", "--beginning", nargs='?', type=int, help="the adds beginning time")
    parser.add_argument("-e", "--end", nargs='?', type=int, help="the adds end time")

    args = parser.parse_args()
    vinput = args.input #input video
    if not os.path.isfile(vinput):
        logging.error('---video does not exist---')
        sys.exit(1)

    outputpath = args.outputpath
    if outputpath != '' and not os.path.exists(outputpath):
        os.makedirs(outputpath)


    bese = int(args.beginning)
    endse = int(args.end)
    fintv = float(args.frequency)

    if int(args.action) == 1 :
        #extract the adds announcement frames
        extract_adds_ref_images(args, vinput,outputpath , bese,endse, fintv )
    else:
        #Dumps the ground truth of the adds
        dump_GT(args, vinput,outputpath , bese,endse, fintv)

    sys.exit(0)



if __name__ == "__main__":

    main()

#Compare the current frame to the adds frame


