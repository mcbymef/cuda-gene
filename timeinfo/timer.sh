#!/bin/bash

cd /work/swanson/carsoncrawford/project

for i in {1..10}
do
    (time ./optimized_cudadist $2) &> $1_$i.dat
done

cat $1_1.dat $1_2.dat $1_3.dat $1_4.dat $1_5.dat $1_6.dat $1_7.dat $1_8.dat $1_9.dat $1_10.dat > $1.out

rm $1_*
mv $1.out timeinfo/$1
