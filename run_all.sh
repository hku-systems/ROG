## outdoors
python3 scripts/run.py --library BSP --threshold 0 -e 2 -c outdoors -n 'BSP' && \ 
python3 scripts/run.py --library SSP --threshold 4 -e 2 -c outdoors -n 'SSP 4' && \
python3 scripts/run.py --library SSP --threshold 20 -e 2 -c outdoors -n 'SSP 20 ' && \ 
python3 scripts/run.py --library ROG --threshold 4 -e 2 -c outdoors -n 'ROG 4' && \
python3 scripts/run.py --library ROG --threshold 20 -e 2 -c outdoors -n 'ROG 20' && \
python3 scripts/test_checkpoints_on_server.py && \
python3 scripts/draw_smooth.py outdoors

## batchsize
python3 scripts/run.py --library BSP --threshold 0 -e 2 -c outdoors -b 1 -n 'BSP-Bx1' && \ 
python3 scripts/run.py --library BSP --threshold 0 -e 3 -c outdoors -b 2 -n 'BSP-Bx2' && \ 
python3 scripts/run.py --library BSP --threshold 0 -e 3 -c outdoors -b 4 -n 'BSP-Bx4' && \ 
python3 scripts/run.py --library SSP --threshold 4 -e 2 -c outdoors -b 1 -n 'SSP-Bx1' && \ 
python3 scripts/run.py --library SSP --threshold 4 -e 3 -c outdoors -b 2 -n 'SSP-Bx2' && \ 
python3 scripts/run.py --library SSP --threshold 4 -e 3 -c outdoors -b 4 -n 'SSP-Bx4' && \
python3 scripts/run.py --library ROG --threshold 4 -e 2 -c outdoors -b 1 -n 'ROG-Bx1' && \ 
python3 scripts/run.py --library ROG --threshold 4 -e 3 -c outdoors -b 2 -n 'ROG-Bx2' && \ 
python3 scripts/run.py --library ROG --threshold 4 -e 3 -c outdoors -b 4 -n 'ROG-Bx4' && \ 
python3 scripts/test_checkpoints_on_server.py && \
python3 scripts/draw_smooth.py batchsize

## threshold
python3 scripts/run.py --library ROG --threshold 4 -e 4 -c outdoors -n 'ROG 4' && \
python3 scripts/run.py --library ROG --threshold 20 -e 5 -c outdoors -n 'ROG 20' && \
python3 scripts/run.py --library ROG --threshold 30 -e 5 -c outdoors -n 'ROG 30' && \
python3 scripts/run.py --library ROG --threshold 40 -e 6 -c outdoors -n 'ROG 40' && \
python3 scripts/test_checkpoints_on_server.py && \
python3 scripts/draw_smooth.py threshold