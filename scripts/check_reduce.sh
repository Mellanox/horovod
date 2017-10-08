#!/bin/bash

rm -f $target
wait

target=out.tmp

cmd="/machineLearning/users/borisko/openmpi-2.1.1/vanilla/bin/mpirun -mca btl_openib_receive_queues P,128,32:P,2048,32:P,12288,32:P,131072,32      -launch-agent 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH /machineLearning/users/borisko/openmpi-2.1.1/vanilla/bin/orted' --allow-run-as-root -np 4 -H 10.143.119.21:2,10.143.119.27:2 -x LD_LIBRARY_PATH python reduce_validity_check.py > $target"

echo $cmd
eval $cmd





wait

i=0
b=1
match=0
s=`cat $target | egrep "^$i)" | wc -l`

margin=0.01
unchecked=0

while [[ `cat $target | egrep "^$i)" | wc -l` -gt 0 ]]; do
  if [[ `cat $target | egrep "^$i)" | wc -l` -lt $s ]]; then
    unchecked=$((unchecked+1))
  else
    a=`cat $target | egrep "^$i)" | cut -d" " -f2 | awk '{s+=$1} END {print s}'`
    m=`cat $target | egrep "^Sum $i: $f" | wc -l `
    if [[ `cat $target | egrep "^Sum $i:" | uniq  | wc -l` -gt 1 ]]; then
      echo "mismatch found at $i: Sums differ"
      exit 1
    fi
    b=`cat $target | egrep "^Sum $i:" | uniq | cut -d" " -f3 `
##    echo $a 
##    echo $b
    if [[ `echo "(($a) - ($b)) > $margin" | bc` -eq 1 ]]; then
       if [[ `echo "(($b) - ($a)) > $margin" | bc` -eq 1 ]]; then
          echo "mismatch found at $i: sum should be $a vs $b"
          exit 1
       fi
    fi
  fi
  i=$((i+1))
done
checked=$((i-unchecked))
echo "Success for $checked variables, $unchecked remain unchecked"
