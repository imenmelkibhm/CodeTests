#!/bin/bash


#LCI
echo extract GT for LCI chuncks...
python ExtractAdds.py -i /opt/exe/textocr/demo/Chunks/LCI_20161017_18062014_18112014_SD.ts   -a 2 -p 1 -o /opt/exe/textocr/demo/Chunks/GroundTruth -n LCI_20161017_18062014_18112014_SD -f 12 -b 0 -e 101
python ExtractAdds.py -i /opt/exe/textocr/demo/Chunks/LCI_20161017_18062014_18112014_SD.ts   -a 2 -p 0 -o /opt/exe/textocr/demo/Chunks/GroundTruth -n LCI_20161017_18062014_18112014_SD -f 12 -b 200 -e 300

#ITele
echo extract GT for iTele chuncks...
python ExtractAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18544715_18594715_SD.ts -a 2 -p 0 -o /opt/exe/textocr/demo/Chunks/GroundTruth -n iTele_20161017_18544715_18594715_SD  -f 12 -b 43 -e 94
python ExtractAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18494715_18544715_SD.ts -a 2 -p 1 -o /opt/exe/textocr/demo/Chunks/GroundTruth -n iTele_20161017_18494715_18544715_SD  -f 12 -b 46 -e 236
python ExtractAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18394715_18444715_SD.ts -a 2 -p 1 -o /opt/exe/textocr/demo/Chunks/GroundTruth -n iTele_20161017_18394715_18444715_SD  -f 12 -b 0 -e 21
python ExtractAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18244715_18294715_SD.ts -a 2 -p 1 -o /opt/exe/textocr/demo/Chunks/GroundTruth -n iTele_20161017_18244715_18294715_SD  -f 12 -b 0 -e 101
python ExtractAdds.py -i /opt/exe/textocr/demo/Chunks/iTele_20161017_18244715_18294715_SD.ts -a 2 -p 0 -o /opt/exe/textocr/demo/Chunks/GroundTruth -n iTele_20161017_18244715_18294715_SD  -f 12 -b 185 -e 236


