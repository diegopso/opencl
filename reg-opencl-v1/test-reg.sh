#!/bin/bash
TIME=${TIME:-`which time` -f%e}
DATA=${DATA:-simple-syntetic.cut44s.su}
PROG=${PROG:-./build/reg}



echo
echo "Test 2:"
echo

ARGS="4120 -480 1.94 0.005
-0.00088484  0.00111516 20 
-0.001194    0.000806   20 
 6.4e-07     8.4e-07    20 
 6.0e-10     8.0e-10    20 
 4.61e-08    6.61e-08   20"

DEVICE_TYPE=0
#$TIME $PROG $ARGS $DATA $DEVICE_TYPE
DEVICE_TYPE=1
$TIME $PROG $ARGS $DATA $DEVICE_TYPE



