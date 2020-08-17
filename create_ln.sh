#!/bin/sh
for i in NewData/[2,3,4,5]/* ; do
     cp $i input/$(basename $i) 
done
