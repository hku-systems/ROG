#!/bin/sh 

DOWNLINK_SPEED=$2
UPLINK_SPEED=$3
NET_DEV=$1

PKT_LIMIT=64

if [ $# -eq 1 ]; then
  tc qdisc del dev $NET_DEV root
  tc qdisc del dev $NET_DEV ingress
  ip link del dev ifb0
  exit
fi

# uplink
tc qdisc replace dev $NET_DEV root netem rate ${UPLINK_SPEED}kbit limit $PKT_LIMIT

# downlink
tc qdisc add dev $NET_DEV handle ffff: ingress >/dev/null 2>&1

ip link add name ifb0 type ifb >/dev/null 2>&1
ip link set up ifb0
tc filter replace dev $NET_DEV parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev ifb0

tc qdisc replace dev ifb0 root netem rate ${DOWNLINK_SPEED}kbit limit $PKT_LIMIT

