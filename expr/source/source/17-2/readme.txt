编译：mpicc intersect.c –o intersect
运行：可以使用命令 mpirun –np SIZE intersect来运行该串匹配程序，其中SIZE是所使用的处理器个数，本实例中使用了SIZE=3个处理器。
　　mpirun –np 3 intersect
运行结果：
输入：  
please input first polygon
please input the count of vetex:3
please input the vertex
3   2
2   5
4   3
please input second polygon
please input the count of vetex:6
please input the vertex
0 1
1 -1
1 -2
6   2
5   4
1   4
输出：
两多边形相交
说明：输入顶点的(x,y)坐标时中间用空格隔开。
