#!/bin/bash



echo Detecting commercial adds
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18244715_18294715_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/frame2660.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18244715_18294715_SD -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18494715_18544715_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/frame2660.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18494715_18544715_SD -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18594715_19044715_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/frame2660.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18594715_19044715_SD -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18544715_18594715_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/frame2660.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18544715_18594715_SD -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18344715_18394715_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/frame2660.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18344715_18394715_SD -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18394715_18444715_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/frame2660.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18394715_18444715_SD -f 12 -d 1

echo Detecting commercial adds
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI/LCI_20161017_17562014_18012014_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/4764.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  LCI_20161017_17562014_18012014_SD_e -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI/LCI_20161017_17562014_18012014_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/1526.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  LCI_20161017_17562014_18012014_SD_s -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI/LCI_20161017_18062014_18112014_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/4764.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  LCI_20161017_18062014_18112014_SD_e -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI/LCI_20161017_18062014_18112014_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/1526.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  LCI_20161017_18062014_18112014_SD_s -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI/LCI_20161017_18012014_18062014_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/4764.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  LCI_20161017_18012014_18062014_SD_e -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI/LCI_20161017_18012014_18062014_SD.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/1526.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  LCI_20161017_18012014_18062014_SD_s -f 12 -d 1


exit


#Channel parameters
dir=/opt/exe/textocr/demo/Chunks/France3
channel=France3
logo=/opt/exe/textocr/demo/AddsReferenceFrames_France3/LogoF3.png
x1=40
x2=103
y1=625
y2=685

echo Detecting Logo in $channel chunks
cd $dir
chunk_list=`ls | grep $channel`
cd /opt/exe/Code/RemoveAdds
for chunck in $chunk_list
do

    name=`basename $chunck .ts`
    python ChannelLogoDetection.py -i $dir/$chunck -l  $logo -x1 $x1  -x2  $x2 -y1 $y1 -y2 $y2 -o /opt/exe/Code/RemoveAdds/LogoMatchs -n $name -f 1 -d 0
done

exit
python ChannelLogoDetection.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18494715_18544715_SD.ts -l /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/LogoITELE.png -x1 0  -x2 120  -y1 0 -y2 120 -o /opt/exe/Code/RemoveAdds/ -n iTele_20161017_18494715_18544715_SD -f 12 -d 0


python ChannelLogoDetection.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18394715_18444715_SD.ts -l /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/LogoITELE.png -o /opt/exe/Code/RemoveAdds/ -n iTele_20161017_18394715_18444715_SD -f 1 -d 0
python ChannelLogoDetection.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18494715_18544715_SD.ts -l /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/LogoITELE.png -o /opt/exe/Code/RemoveAdds/ -n iTele_20161017_18494715_18544715_SD -f 1 -d 0
python ChannelLogoDetection.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18594715_19044715_SD.ts -l /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/LogoITELE.png -o /opt/exe/Code/RemoveAdds/ -n iTele_20161017_18594715_19044715_SD -f 1 -d 0
python ChannelLogoDetection.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18544715_18594715_SD.ts -l /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/LogoITELE.png -o /opt/exe/Code/RemoveAdds/ -n iTele_20161017_18544715_18594715_SD -f 1 -d 0
python ChannelLogoDetection.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18344715_18394715_SD.ts -l /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/LogoITELE.png -o /opt/exe/Code/RemoveAdds/ -n iTele_20161017_18344715_18394715_SD -f 1 -d 0
python ChannelLogoDetection.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18244715_18294715_SD.ts -l /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/LogoITELE.png -o /opt/exe/Code/RemoveAdds/ -n iTele_20161017_18244715_18294715_SD -f 1 -d 0
python ChannelLogoDetection.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18294715_18344715_SD.ts -l /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/LogoITELE.png -o /opt/exe/Code/RemoveAdds/ -n iTele_20161017_18294715_18344715_SD -f 1 -d 0
python ChannelLogoDetection.py -i /opt/exe/textocr/demo/Chunks/iTele/iTele_20161017_18444715_18494715_SD.ts -l /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/LogoITELE.png -o /opt/exe/Code/RemoveAdds/ -n  iTele_20161017_18444715_18494715_SD -f 1 -d 0


exit


python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI/D161017-0689.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/4764.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  D161017-0689_e -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI/D161017-0689.ts -al 2 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/1526.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  D161017-0689_s -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI/D161017-0689.ts -al 1 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/weather_frame5006.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  D161017-0689 -f 12 -d 1

exit


echo Detecting weather forecast
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18244715_18294715_SD.ts -al 1 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/weather_frame4655.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18244715_18294715_SD -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18494715_18544715_SD.ts -al 1 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/weather_frame4655.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18494715_18544715_SD -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18594715_19044715_SD.ts -al 1 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/weather_frame4655.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18594715_19044715_SD -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18544715_18594715_SD.ts -al 1 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/weather_frame4655.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18544715_18594715_SD -f 12  -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18344715_18394715_SD.ts -al 1 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/weather_frame4655.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18344715_18394715_SD -f 12  -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18394715_18444715_SD.ts -al 1 -af /opt/exe/textocr/demo/AddsReferenceFrames_ITELE/weather_frame4655.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_ITELE/  -n  iTele_20161017_18394715_18444715_SD -f 12  -d 1



echo Detecting weather forecast
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI_20161017_17562014_18012014_SD.ts -al 1 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/weather_frame5006.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  LCI_20161017_17562014_18012014_SD_s -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI_20161017_18062014_18112014_SD.ts -al 1 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/weather_frame5006.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  LCI_20161017_18062014_18112014_SD_s -f 12 -d 1
python DetectAdds.py -i /opt/exe/textocr/demo/Chunks/LCI_20161017_18012014_18062014_SD.ts -al 1 -af /opt/exe/textocr/demo/AddsReferenceFrames_LCI/weather_frame5006.jpg -o /opt/exe/textocr/demo/AddsFramesDetection_LCI -n  LCI_20161017_18012014_18062014_SD_s -f 12 -d 1

