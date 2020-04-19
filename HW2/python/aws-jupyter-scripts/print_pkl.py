#!/usr/bin/env python3
import sys
import pickle as pk
def print_pickle(pickle_fn):
	with open(pickle_fn,'rb') as pkl:
	    stat=pk.load(pkl)
	stat.keys()
	print(stat['creation_stats'])
	print(f"Output has {len(stat['Random_pokes'])} measurements.")
	for pkl in stat['Random_pokes']:
	    print(f"input size = {pkl['m']:.1e}.")
	    print(f"mean latency (sec) = {pkl['memory__mean']:.3e}.")
	    print(f"std latency (sec) = {pkl['memory__std']:.3e}.")
	    print(f"mean of largest 100 latencies (sec) = {np.mean(pkl['memory__largest']):.3e}.")
	    print('')

if __name__ == "__main__":
   print_pickle(sys.argv[1])