Example:
编译：mpicc -o box box.c -lm
运行：(group_size为处理器个数)
　mpirun -np group_size box 这里运行mpirun –np 8 box
运行结果：
input a[8]:
0.200000
0.500000
0.400000
0.200000
0.300000
0.300000
0.100000
0.800000
d 0 1:1
d 0 2:3
d 0 3:6
d 0 4:8
processor number is 8
my_rank 4
number 4 goods put into box 2
processor number is 8
my_rank 1
number 1 goods put into box 1
processor number is 8
my_rank 5
number 5 goods put into box 2
processor number is 8
my_rank 7
number 7 goods put into box 3
processor number is 8
my_rank 2
number 2 goods put into box 1
processor number is 8
my_rank 3
number 3 goods put into box 2
processor number is 8
my_rank 8
number 8 goods put into box 4
processor number is 8
my_rank 6
number 6 goods put into box 3

说明:物品的体积存在文件data1中