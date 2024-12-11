Example:
编译：mpicc including.c –o including
运行：可以使用命令 mpirun –np SIZE including来运行该串匹配程序，其中SIZE是所使用的处理器个数。本实例中使用了SIZE=4个处理器。
mpirun –np 4 including
运行结果：
请输入点的个数:6
请输入各点的坐标
0：1 0
1：2 1
2：4 0
3：2 4
4：1 2
5：0 3
请输入要判断点的坐标
2 3
结果：vertex p is in polygon
说明：输入顶点的(x,y)坐标时中间用空格隔开。
