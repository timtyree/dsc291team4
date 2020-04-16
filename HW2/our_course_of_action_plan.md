# dsc291team4 - Our Course of Action Plan (OCAP) for HW2 is as follows:
Goal: How does the performance of random-poke.py relate to the size of the caches in the ec2 instance?
- step 1

Find the instances which have a significant difference in their L1, L2 and L3 cache sizes. (Currently not sure where we are going to find this info)

- step 2

Perform random-poke.py on these instances and investigate the distribution of latency times for different size arrays.

- step 3

Null Hypothesis: Cache sizes have no affect on latency times.

Alternative Hypothesis: Higher cache size leads to lower latency times.

Design a p-test to reject or retain the null hypothesis. (Ask the Professor about this)


Nota bene: much of the functionality for measurements is forked from the github repository located <url = https://github.com/yoavfreund/Public-DSC291/>here</url>.
