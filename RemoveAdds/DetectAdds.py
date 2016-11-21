#example python DetectAdds.py -i /opt/exe/textocr/demo/iTele_20161017_18244715_18294715_SD.ts -af /opt/exe/textocr/demo/AddsFrames_ITELE/frame2660.jpg -o /opt/exe/textocr/demo/AddsFrames_IT -f 12 -b 104 -e 109

from skimage.measure import structural_similarity as ssim
import logging
import os
import numpy as np
import cv2
import sys
import argparse
import numpy as np
import pylab as pl
import time
import math
import operator
import subprocess


def compare_images_distance_Gray(imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum(np.sqrt((imageA.astype("float") - imageB.astype("float")) ** 2))
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

def compare_images_distance_RGB(imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum(np.sqrt(((imageA[:,:,0].astype("float") - imageB[:,:,0].astype("float")) ** 2)+ (imageA[:,:,1].astype("float") - imageB[:,:,1].astype("float")) ** 2 + (imageA[:,:,2].astype("float") - imageB[:,:,2].astype("float")) ** 2))
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err


def compare_histo_HSV(imageA, imageB):
    # compare histgram of the lower center part of the image
    # hsv1 = cv2.cvtColor(imageA[int(np.round(imageB.shape[0] * 0.75)):,
    #                     int(np.round(imageB.shape[0] * 0.4)):int(np.round(imageB.shape[0] * 0.6))], cv2.COLOR_BGR2HSV)
    # hsv2 = cv2.cvtColor(imageB[int(np.round(imageB.shape[0] * 0.75)):,
    #                     int(np.round(imageB.shape[0] * 0.4)):int(np.round(imageB.shape[0] * 0.6))], cv2.COLOR_BGR2HSV)
    #Convert BRG (color iameg format of openCV) to HSV
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)

    h1 = cv2.calcHist([imageA], [0], None, [16], [0, 256])
    h2 = cv2.calcHist([imageB], [0], None, [16], [0, 256])

    sim = math.sqrt(reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, h1, h2))) / len(h1))
    return sim


def compare_histo_Gray(imageA, imageB):
    # compare histgram of the lower center part of the image
    # hsv1 = cv2.cvtColor(imageA[int(np.round(imageB.shape[0] * 0.75)):,
    #                     int(np.round(imageB.shape[0] * 0.4)):int(np.round(imageB.shape[0] * 0.6))], cv2.COLOR_BGR2HSV)
    # hsv2 = cv2.cvtColor(imageB[int(np.round(imageB.shape[0] * 0.75)):,
    #                     int(np.round(imageB.shape[0] * 0.4)):int(np.round(imageB.shape[0] * 0.6))], cv2.COLOR_BGR2HSV)
    #Convert BRG (color iameg format of openCV) to grayscale
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    h1 = cv2.calcHist([imageA], [0], None, [16], [0, 256])
    h2 = cv2.calcHist([imageB], [0], None, [16], [0, 256])

    sim = math.sqrt(reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, h1, h2))) / len(h1))
    return sim


def compare_histo_RGB(imageA, imageB):
    # compare histgram of the lower center part of the image
    # hsv1 = cv2.cvtColor(imageA[int(np.round(imageB.shape[0] * 0.75)):,
    #                     int(np.round(imageB.shape[0] * 0.4)):int(np.round(imageB.shape[0] * 0.6))], cv2.COLOR_BGR2HSV)
    # hsv2 = cv2.cvtColor(imageB[int(np.round(imageB.shape[0] * 0.75)):,
    #                     int(np.round(imageB.shape[0] * 0.4)):int(np.round(imageB.shape[0] * 0.6))], cv2.COLOR_BGR2HSV)
    #Convert BRG (color iameg format of openCV) to grayscale
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)

    h1 = cv2.calcHist([imageA], [0,1,2], None, [16, 16, 16], [0, 256,0, 256,0, 256])
    h2 = cv2.calcHist([imageB], [0,1,2], None, [16, 16, 16], [0, 256,0, 256,0, 256])

    sim = cv2.compareHist(h1,h2,cv2.HISTCMP_CHISQR)#math.sqrt(reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, h1, h2))) / len(h1))
    return sim

Compare_Method = {
    0: compare_histo_HSV,
    1: compare_histo_Gray,
    2: compare_histo_RGB,
    3: compare_images_distance_Gray,
    4: compare_images_distance_RGB
}


def compare_images(imageA, imageB, method):

    if int(method) in Compare_Method:
        return Compare_Method[int(method)](imageA, imageB)
    else:
        logging.error("The comparing method is not defined! Use the default method instead: compare_images_distance_Gray")
        return Compare_Method[int(1)](imageA, imageB)



#Get and store the wanted frames
def detect_adds_images_from_video(args):

        Debug = int(args.debug)
        vinput = args.input  # input video
        if not os.path.isfile(vinput):
            logging.error('---video does not exist---')
            sys.exit(1)

        cap = cv2.VideoCapture(vinput)
        logging.warning('***************************************Opening the video: ' + args.input+ ' for commercial adds detection**********************************************')
        fintv = float(args.frequency)
        fps = cap.get(5)  # frame per second in video
        frn = int(cap.get(7))  # frame number


        outputpath = args.outputpath
        if outputpath != '' and not os.path.exists(outputpath):
            os.makedirs(outputpath)

        # verify beginning and end time
        if args.beginning is None:
            bese = 0
        else:
            bese = args.beginning
        if args.end is None:
            endse = (frn/fps)
        else:
            endse = args.end
        if bese >= endse or bese < 0 or endse > (frn/fps):
            logging.error('wrong arguments of beginning and end time')
            sys.exit(1)

        logging.info('process each segment of video {0}'.format(args.input))
        befr = int(bese * fps)  # begining frame
        endfr = int(endse * fps)  # ending frame
        ms_array = []
        ssim_array = []
        histo_array = []
        frames = []
        if cap.isOpened():  # if video is opened, process frames
            ret, frame = cap.read()
            counter = 0
            #print('endfr = %d' % endfr + 'endse %d ' % endse + 'fps %d' %fps + 'frn %d' %frn )
            for i in xrange(befr, endfr, int(np.round(fps / fintv))):
                #print('i = %d' %i + '/ %d' %frn)
                while (counter != i):
                    #print('counter = %d' %counter)
                    ret, frame = cap.read()
                    counter +=1

                #logging.warning('Read the %d frame' % i)

                #2. Compare the current frame to the adds frame
                if ret == True:
                    reference = cv2.imread(args.addsframe)
                    current = frame
                    hist = compare_histo_Gray(reference,current)
                    # ssimeas = ssim(reference,current)
                    # mserr = mse(reference,current)
                    # ms_array.append(mserr)
                    # ssim_array.append(ssimeas)
                    #logging.warning('histo = %d' % hist)
                    if hist < 500 :
                        print('Pubs frame detected at frame %d' % i)
                        # cv2.imshow('image', current)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    histo_array.append(hist)
                    frames.append(int(i/fps))

            if Debug==1:
                pl.figure(figsize=(30, 4))
                chunckname_wextension=os.path.basename(args.input)
                chunckname=chunckname_wextension.split('.')[0]
                if not os.path.isfile('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname + '_Pub_GroundTruth.txt'):
                    logging.warning('No ground Truth file found for commercial adds detection')
                    pl.plot(frames,histo_array, 'r')
                else:
                    GT = np.loadtxt('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname + '_Pub_GroundTruth.txt')
                    GT = GT* (max(histo_array))
                    print('GT dimension %d' % np.shape(GT))
                    print('histo_array dimension %d' % np.shape(histo_array))
                    #pl.plot(frames,ms_array, 'r',frames, ssim_array,'b' ,frames,histo_array, 'g')
                    pl.plot(frames,histo_array, 'r', label='similarity')
                    pl.plot(frames, GT, 'g', label='Ground Truth')
                    pl.legend()

                pl.xlabel('time (s)')
                pl.ylabel('amplitude')
                pl.savefig(os.path.join(outputpath, args.outputname + "_Pub_histo_freq_"+ args.frequency + ".jpg"), dpi=50)
                pl.show()


#Get and store the wanted frames
def Compute_Shot_Cuts(args):

        Debug = int(args.debug)
        vinput = args.input  # input video
        if not os.path.isfile(vinput):
            logging.error('---video does not exist---')
            sys.exit(1)

        cap = cv2.VideoCapture(vinput)
        logging.warning('***************************************Opening the video: ' + args.input+ ' for commercial adds detection**********************************************')
        fintv = float(args.frequency)
        fps = cap.get(5)  # frame per second in video
        frn = int(cap.get(7))  # frame number

        outputpath = args.outputpath
        if outputpath != '' and not os.path.exists(outputpath):
            os.makedirs(outputpath)

        # verify beginning and end time
        if args.beginning is None:
            bese = 0
        else:
            bese = args.beginning
        if args.end is None:
            endse = (frn/fps)
        else:
            endse = args.end
        if bese >= endse or bese < 0 or endse > (frn/fps):
            logging.error('wrong arguments of beginning and end time')
            sys.exit(1)

        logging.info('process each segment of video {0}'.format(args.input))
        befr = int(bese * fps)  # begining frame
        endfr = int(endse * fps)  # ending frame
        histo_array1 = []
        histo_array2 = []
        histo_array3 = []
        frames = []
        previous = None
        if cap.isOpened():  # if video is opened, process frames
            ret, current = cap.read()
            counter = 0
            for i in xrange(befr, endfr, int(np.round(fps / fintv))):
                #print('i = %d' %i + '/ %d' %frn)
                while (counter != i):
                    ret, current = cap.read()
                    counter +=1

              #2. Compare the current frame to the adds frame

                if previous is not None:
                    ret, current = cap.read()
                    # convert the images to grayscale
                    if current is not None:
                        #hist_RGB  = compare_histo_RGB(previous,current)
                        hist_GRAY = compare_histo_Gray(previous,current)
                        hist_HSV  = compare_histo_HSV(previous,current)
                        #histo_array1.append(hist_RGB)
                        histo_array2.append(hist_GRAY)
                        histo_array3.append(hist_HSV)
                        frames.append(i)
                previous = current

            if Debug==1:
                pl.figure(figsize=(30, 4))
                chunckname_wextension=os.path.basename(args.input)
                chunckname=chunckname_wextension.split('.')[0]
                #pl.plot(frames, histo_array1,'r',label='RGB')
                pl.plot( frames, histo_array2, 'g',label='GRAY')
                pl.plot( frames, histo_array3,'b', label='HSV')
                pl.legend()
                # if not os.path.isfile('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname + '_Pub_GroundTruth.txt'):
                #     logging.warning('No ground Truth file found for commercial adds detection')
                #
                # else:
                #     GT = np.loadtxt('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname + '_Pub_GroundTruth.txt')
                #     GT = GT* (max(histo_array))
                #     print('GT dimension %d' % np.shape(GT))
                #     print('histo_array dimension %d' % np.shape(histo_array))
                #     #pl.plot(frames,ms_array, 'r',frames, ssim_array,'b' ,frames,histo_array, 'g')
                #     pl.plot(frames,histo_array, 'r', frames, GT, 'g')

                pl.xlabel('time (s)')
                pl.ylabel('amplitude')
                #pl.savefig(os.path.join(outputpath, args.outputname + "_shotcuts_histo_freq_"+ args.frequency + ".jpg"), dpi=50)
                pl.show()

#Get and store the wanted frames
def Compute_Shot_Cuts_from_video_(args):

        Debug = int(args.debug)
        vinput = args.input  # input video
        if not os.path.isfile(vinput):
            logging.error('---video does not exist---')
            sys.exit(1)

        cap = cv2.VideoCapture(vinput)
        logging.warning('Opening the video')
        fintv = float(args.frequency)
        fps = cap.get(5)  # frame per second in video
        frn = int(cap.get(7))  # frame number
        duration = int(frn / fps)

        outputpath = args.outputpath
        if outputpath != '' and not os.path.exists(outputpath):
            os.makedirs(outputpath)

        # verify beginning and end time
        if args.beginning is None:
            bese = 0
        else:
            bese = args.beginning
        if args.end is None:
            endse = duration
        else:
            endse = args.end
        if bese >= endse or bese < 0 or endse > duration:
            logging.error('wrong arguments of beginning and end time')
            sys.exit(1)

        logging.info('process each segment of video {0}'.format(args.input))
        befr = int(bese * fps)  # begining frame
        endfr = int(endse * fps)  # ending frame
        histo_array1 = []
        histo_array2 = []
        histo_array3 = []
        frames = []
        previous = None
        if cap.isOpened():  # if video is opened, process frames
            for i in xrange(befr, endfr, int(np.round(fps / fintv))):
                # 1.----------read the specific frames in the video-----------
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # get the specific frames
                ret, frame = cap.read()
                if ret == True:
                    current = frame
                    if previous is not None:
                        dist_RGB = compare_images(previous,current,4)
                        #dist_Gray = compare_images(previous,current,3)
                        #histo_array1.append(hist_RGB)
                        histo_array2.append(dist_RGB)
                        #histo_array3.append(dist_Gray)
                        frames.append(i)
                    else:
                        histo_array2.append(0)
                        #histo_array3.append(0)
                        frames.append(i)
                    previous = current

            if Debug==1:
                pl.figure(figsize=(30, 4))
                chunckname_wextension=os.path.basename(args.input)
                chunckname=chunckname_wextension.split('.')[0]
                if not os.path.isfile('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname + '_Pub_GroundTruth.txt'):
                    logging.warning('No ground Truth file found for commercial adds detection')
                    #pl.plot(frames, histo_array1,'r',label='RGB')
                    pl.plot( frames, histo_array2, 'g',label='GRAY')
                    #pl.plot( frames, histo_array3,'b', label='HSV')
                    pl.legend()

                else:
                    GT = np.loadtxt('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname + '_Pub_GroundTruth.txt')
                    GT = GT* (max(histo_array2))
                    print('GT dimension %d' % np.shape(GT))
                    print('histo_array dimension %d' % np.shape(histo_array2))
                    #pl.plot( frames, GT, 'y', label='GT')
                    pl.plot( frames, histo_array2, 'g',label='GRAY')
                    #pl.plot( frames, histo_array3,'b', label='HSV')
                    pl.legend()


                pl.xlabel('time (s)')
                pl.ylabel('amplitude')
                pl.savefig(os.path.join(outputpath, args.outputname + "_shotcuts_histo_freq_"+ args.frequency + ".jpg"), dpi=50)
                #pl.show()


#Get and store the wanted frames
def detect_adds_images_from_video_(args):

        vinput = args.input  # input video
        if not os.path.isfile(vinput):
            logging.error('---video does not exist---')
            sys.exit(1)

        cap = cv2.VideoCapture(vinput)
        logging.warning('Opening the video')
        fintv = float(args.frequency)
        fps = cap.get(5)  # frame per second in video
        frn = int(cap.get(7))  # frame number
        duration = int(frn / fps)

        outputpath = args.outputpath
        if outputpath != '' and not os.path.exists(outputpath):
            os.makedirs(outputpath)

        # verify beginning and end time
        if args.beginning is None:
            bese = 0
        else:
            bese = args.beginning
        if args.end is None:
            endse = duration
        else:
            endse = args.end
        if bese >= endse or bese < 0 or endse > duration:
            logging.error('wrong arguments of beginning and end time')
            sys.exit(1)
        duration = min(endse - bese, duration)


        logging.info('process each segment of video {0}'.format(args.input))
        befr = int(bese * fps)  # begining frame
        endfr = int(endse * fps)  # ending frame
        ms_array = []
        ssim_array = []
        histo_array = []
        frames = []
        if cap.isOpened():  # if video is opened, process frames
            for i in xrange(befr, endfr, int(np.round(fps / fintv))):
                #1.----------read the specific frames in the video-----------
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # get the specific frames
                logging.warning('Read the %d frame' % i)
                ret, frame = cap.read()

                #2. Compare the current frame to the adds frame
                if ret == True:
                    reference = cv2.imread(args.addsframe)
                    current = frame
                    # convert the images to grayscale
                    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
                    current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
                    hist = compare_histo_Gray(reference,current)
                    #logging.warning('histo = %d' % hist)
                    if hist < 500:
                        print('Pubs frame detected at frame %d' % i)
                        cv2.imshow('image', current)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    # ssimeas = ssim(reference,current)
                    # mserr = mse(reference,current)
                    # ms_array.append(mserr)
                    # ssim_array.append(ssimeas)
                    histo_array.append(hist)
                    frames.append(i)

            pl.figure(figsize=(30, 4))
            #pl.plot(frames,ms_array, 'r',frames, ssim_array,'b' ,frames,histo_array, 'g')
            pl.plot(frames,histo_array, 'g')
            pl.xlabel('time (s)')
            pl.ylabel('amplitude')
            pl.savefig(os.path.join(outputpath, args.outputname + "histo.jpg"), dpi=50)
            #pl.show()


#Get and store the wanted frames
def detect_adds_images_from_video_images(args):

        Debug = int(args.debug)
        vinput = args.input  # input video
        if not os.path.isfile(vinput):
            logging.error('---video does not exist---')
            sys.exit(1)

        cap = cv2.VideoCapture(vinput)
        logging.warning('Opening the video')
        fintv = float(args.frequency)
        fps = cap.get(5)  # frame per second in video
        frn = int(cap.get(7))  # frame number
        duration = frn / fps

        outputpath = args.outputpath
        if outputpath != '' and not os.path.exists(outputpath):
            os.makedirs(outputpath)

        # verify beginning and end time
        if args.beginning is None:
            bese = 0
        else:
            bese = args.beginning
        if args.end is None:
            endse = duration
        else:
            endse = args.end
        if bese >= endse or bese < 0 or endse > duration:
            logging.error('wrong arguments of beginning and end time')
            sys.exit(1)
        duration = min(endse - bese, duration)
        maxframe = int(fintv*frn/fps)
        #dump the video images on the disk

        subprocess.call('ffmpeg -i ' + args.input + ' -vf fps=' + args.frequency + ' ' + args.dumprepo + '/frame-%d.jpg' + ' -threads 0', shell=True) #faster
        #subprocess.call('ffmpeg -i ' + args.input + ' -r ' + args.frequency + ' -f image2 /opt/exe/textocr/demo/temp_images/frame-%d.jpg' + ' -threads 0', shell=True)

        logging.info('process each segment of video {0}'.format(args.input))
        befr = int(bese * fps)  # begining frame
        endfr = int(endse * fps)  # ending frame
        ms_array = []
        ssim_array = []
        histo_array = []
        frames = []

        for i in xrange(0,maxframe+1,1):
        #for i in xrange(befr, endfr, int(np.round(fps / fintv))):
            im = cv2.imread("/opt/exe/textocr/demo/temp_images/frame-%d.jpg" %(i+1))
            if im is None:
                logging.error('Image not found!! : /opt/exe/textocr/demo/temp_images/frame-%d.jpg' %i)
                continue

            # 2. Compare the current frame to the commercials announcement frames
            reference = cv2.imread(args.addsframe)
            current = im
            # convert the images to grayscale
            reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            hist = compare_histo(reference, current)
            # ssimeas = ssim(reference,current)
            # mserr = mse(reference,current)
            # ms_array.append(mserr)
            # ssim_array.append(ssimeas)
            histo_array.append(hist)
            frames.append(i)

        # Remove the dumped images
        subprocess.call('rm -f '+ args.dumprepo + '/*', shell=True)


        if Debug==1:
            pl.figure(figsize=(30, 4))
            chunckname_wextension=os.path.basename(args.input)
            chunckname=chunckname_wextension.split('.')[0]
            GT = np.loadtxt('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname +'_Pub_GroundTruth.txt')
            GT = GT* (max(histo_array))
            print('GT dimension %d' % np.shape(GT))
            print('histo_array dimension %d' % np.shape(histo_array))
            #pl.plot(frames,ms_array, 'r',frames, ssim_array,'b' ,frames,histo_array, 'g')
            pl.plot(frames,histo_array, 'r', GT, 'g')
            pl.xlabel('time (s)')
            pl.ylabel('amplitude')
            pl.savefig(os.path.join(outputpath, args.outputname + "histo.jpg"), dpi=50)
            pl.show()


        # pl.figure(figsize=(30, 4))
        # #pl.plot(frames,ms_array, 'r',frames, ssim_array,'b' ,frames,histo_array, 'g')
        # pl.plot(frames,histo_array, 'g')
        # pl.xlabel('time (s)')
        # pl.ylabel('amplitude')
        # pl.savefig(os.path.join(outputpath, args.outputname + "histo.jpg"), dpi=50)
        # pl.show()
        # print('Read the %d frame' % i + '/opt/exe/textocr/demo/temp_images/frame-%d.jpg' % i)



            # for i in xrange(befr, endfr, int(np.round(fps / fintv))):
            #     # 1.----------read the specific frames in the video-----------
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # get the specific frames
            #     #logging.warning('Read the %d frame' % i)
            #     ret, frame = cap.read()
            #     #print("CAP_PROP_POS_FRAMES= %d" % cap.get(cv2.CAP_PROP_POS_FRAMES))
            #     # while cap.get(cv2.CAP_PROP_POS_FRAMES) != i:
            #     #     ret, frame = cap.read()
            #     #2. Compare the current frame to the adds frame
            #     if ret == True:
            #         reference = cv2.imread(args.addsframe)
            #         current = frame
            #
            #         # convert the images to grayscale
            #         reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            #         current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            #         hist = compare_histo(reference,current)
            #         ssimeas = ssim(reference,current)
            #         mserr = mse(reference,current)
            #         ms_array.append(mserr)
            #         ssim_array.append(ssimeas)
            #         histo_array.append(hist)
            #         frames.append(i)
            #         #print("the error is %f" % m)
            #         #cv2.imwrite(os.path.join(outputpath, "%d.jpg" % i),frame) # save the frame as a JPEG File
            #
            # pl.figure(figsize=(30, 4))
            # pl.plot(frames,ms_array, 'r',frames, ssim_array,'b' ,frames,histo_array, 'g')
            # pl.xlabel('time (s)')
            # pl.ylabel('amplitude')
            # pl.savefig(os.path.join(outputpath, args.outputname + "histo.jpg"), dpi=50)
            # pl.show()


#Get and store the wanted frames
def detect_Weather_Forcast_images_from_video(args):

        Debug = int(args.debug)
        vinput = args.input  # input video
        if not os.path.isfile(vinput):
            logging.error('---video does not exist---')
            sys.exit(1)

        cap = cv2.VideoCapture(vinput)
        logging.warning('***************************************Opening the video: ' + args.input+ ' for weather forecast detection**********************************************')
        fintv = float(args.frequency)
        fps = cap.get(5)  # frame per second in video
        frn = int(cap.get(7))  # frame number


        outputpath = args.outputpath
        if outputpath != '' and not os.path.exists(outputpath):
            os.makedirs(outputpath)

        # verify beginning and end time
        if args.beginning is None:
            bese = 0
        else:
            bese = args.beginning
        if args.end is None:
            endse = (frn/fps)
        else:
            endse = args.end
        if bese >= endse or bese < 0 or endse > (frn/fps):
            logging.error('wrong arguments of beginning and end time')
            sys.exit(1)

        logging.info('process each segment of video {0}'.format(args.input))
        befr = int(bese * fps)  # begining frame
        endfr = int(endse * fps)  # ending frame
        ms_array = []
        ssim_array = []
        histo_array = []
        frames = []
        if cap.isOpened():  # if video is opened, process frames
            ret, frame = cap.read()
            counter = 0
            #print('endfr = %d' % endfr + 'endse %d ' % endse + 'fps %d' %fps + 'frn %d' %frn )
            for i in xrange(befr, endfr, int(np.round(fps / fintv))):
                #print('i = %d' %i + '/ %d' %frn)
                while (counter != i):
                    #print('counter = %d' %counter)
                    ret, frame = cap.read()
                    counter +=1

                #logging.warning('Read the %d frame' % i)

                #2. Compare the current frame to the adds frame
                if ret == True:
                    reference = cv2.imread(args.addsframe)
                    current = frame
                    # convert the images to grayscale
                    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
                    current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
                    hist = compare_histo(reference,current)
                    # ssimeas = ssim(reference,current)
                    # mserr = mse(reference,current)
                    # ms_array.append(mserr)
                    # ssim_array.append(ssimeas)
                    #logging.warning('histo = %d' % hist)
                    if hist < 500 :
                        print('Weather forecast frame detected at frame %d' % i)
                        # cv2.imshow('image', current)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    histo_array.append(hist)
                    frames.append(i)

            if Debug==1:
                pl.figure(figsize=(30, 4))
                chunckname_wextension=os.path.basename(args.input)
                chunckname=chunckname_wextension.split('.')[0]
                if not os.path.isfile('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname +'_Meteo_GroundTruth.txt'):
                    logging.warning('No ground Truth file found for weather forecast adds detection')
                    pl.plot(frames,histo_array, 'r')
                else:
                    GT = np.loadtxt('/opt/exe/textocr/demo/Chunks/GroundTruth/' + chunckname + '_Meteo_GroundTruth.txt')
                    GT = GT* (max(histo_array))
                    print('GT dimension %d' % np.shape(GT))
                    print('histo_array dimension %d' % np.shape(histo_array))
                    #pl.plot(frames,ms_array, 'r',frames, ssim_array,'b' ,frames,histo_array, 'g')
                    pl.plot(frames,histo_array, 'r', frames, GT, 'g')

                pl.xlabel('time (s)')
                pl.ylabel('amplitude')
                pl.savefig(os.path.join(outputpath, args.outputname + "_Meteo_histo_freq_"+ args.frequency + ".jpg"), dpi=50)
                #pl.show()





#def video_to_images(args):

#the main function
def main():
    #Read input arguments
    parser = argparse.ArgumentParser(description='Extract the adds frames')
    parser.add_argument("-i", "--input", help="input video file")
    parser.add_argument("-al", "--algo", help="detect (1) Weather Forecast or (2) Commercial adds")
    parser.add_argument("-af", "--addsframe", help="the frame of the adds anouncement")
    parser.add_argument("-dr", "--dumprepo", help="the temporary folder used to dump the video images")
    parser.add_argument("-o", "--outputpath", help="the path to save the adds frames")
    parser.add_argument("-n", "--outputname", help="the name to save the adds frames")
    parser.add_argument("-f", "--frequency", default=1.0, help="the frequency of extracting and processing video frames")
    parser.add_argument("-b", "--beginning", nargs='?', type=int, help="the adds beginning time")
    parser.add_argument("-e", "--end", nargs='?', type=int, help="the adds end time")
    parser.add_argument("-d", "--debug", nargs='?', type=int, help="Debug mode")
    args = parser.parse_args()

    starttime = time.time()
    algo = int(args.algo)

    if (algo==1):
        detect_Weather_Forcast_images_from_video(args)
    else:
        Compute_Shot_Cuts_from_video_(args)
        #detect_adds_images_from_video(args)

    endtime = time.time()
    logging.warning('Processing the video in %f' % (endtime-starttime))
    sys.exit(0)



if __name__ == "__main__":

    main()




