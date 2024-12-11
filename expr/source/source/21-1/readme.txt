1. compile:
mpicc power.c -o power

2. run:
mpirun -np 4 power

3. result:
Input of file "dataIn.txt"
4       5
1.000000        4.000000        -2.000000       3.000000        6.000000
2.000000        2.000000        0.000000        4.000000        2.000000
3.000000        0.000000        -1.000000       2.000000        1.000000
1.000000        2.000000        2.000000        -3.000000       8.000000

Output of running
 1 th  differ=48.000000
 2 th  differ=43.125000
 3 th  differ=1.227564
 4 th  differ=1.652985
 5 th  differ=2.370061
 6 th  differ=2.789871
 7 th  differ=3.555358
 8 th  differ=3.538388
 9 th  differ=3.609998
10 th  differ=3.235491
11 th  differ=2.653407
12 th  differ=2.354585
13 th  differ=1.957072
14 th  differ=1.723574
15 th  differ=1.446417
16 th  differ=1.266393
17 th  differ=1.069886
18 th  differ=0.932502
19 th  differ=0.791576
20 th  differ=0.687567
21 th  differ=0.585668
22 th  differ=0.507399
23 th  differ=0.433281
24 th  differ=0.374648
25 th  differ=0.320504
26 th  differ=0.276730
27 th  differ=0.237051
28 th  differ=0.204453
29 th  differ=0.175309
30 th  differ=0.151080
31 th  differ=0.129637
32 th  differ=0.111653
33 th  differ=0.095857
34 th  differ=0.082522
35 th  differ=0.070875
36 th  differ=0.060996
37 th  differ=0.052402
38 th  differ=0.045086
39 th  differ=0.038742
40 th  differ=0.033328
41 th  differ=0.028643
42 th  differ=0.024636
43 th  differ=0.021174
44 th  differ=0.018211
45 th  differ=0.015654
46 th  differ=0.013462
47 th  differ=0.011573
48 th  differ=0.009953
49 th  differ=0.008556
50 th  differ=0.007357
51 th  differ=0.006324
52 th  differ=0.005438
53 th  differ=0.004675
54 th  differ=0.004020
55 th  differ=0.003457
56 th  differ=0.002972
57 th  differ=0.002555
58 th  differ=0.002197
59 th  differ=0.001889
60 th  differ=0.001625
61 th  differ=0.001397
62 th  differ=0.001201
63 th  differ=0.001033
64 th  differ=0.000888
65 th  differ=0.000763
66 th  differ=0.000656
67 th  differ=0.000564
68 th  differ=0.000484
69 th  differ=0.000417
70 th  differ=0.000359
71 th  differ=0.000309
72 th  differ=0.000266
73 th  differ=0.000228
74 th  differ=0.000196
75 th  differ=0.000168
76 th  differ=0.000144
77 th  differ=0.000124
78 th  differ=0.000107
79 th  differ=0.000092
the envalue is 5.681839

Iteration num = 79
Whole running time    = 0.083758 s
Distribute data time  = 0.002005 s
Parallel compute time = 0.081753 s
