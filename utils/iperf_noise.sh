#!/bin/bash

server=$1
worker=$2
datasize=$3
t=$4
random_time=$(( t + RANDOM % t ))
sleep $random_time

while true
do 
    echo $(date +"%T") "$worker to $server: $datasize"
    ssh $worker "iperf -c $server --parallel -n $datasize" &> /dev/null
    random_time=$(( t + RANDOM % t ))
    sleep $random_time
    echo $(date +"%T") "sleep for $random_time seconds"
done