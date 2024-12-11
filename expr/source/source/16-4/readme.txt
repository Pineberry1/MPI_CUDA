Example:
编译：mpicc -o beibao beibao.c -lm
运行：(group_size为处理器个数)
　mpirun -np group_size box 这里运行mpirun –np 5 beibao
运行结果：
my_rank 0
knapscack of capacity = 4
Enter number of values:
6
please input p:
16 34 20 25 30 26
please input w:
2 3 5 6 4 1
 my rank is 1
 my rank is 3
 my rank is 4
 the result:
z 1:0
z 2:1
z 3:0
z 4:0
z 5:0
z 6:1
 my rank is 2
说明:背包的容量为处理器个数-1
