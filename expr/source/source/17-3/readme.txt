Example:
编译：mpicc c.c –o convexity -lm
运行：可以使用命令 mpirun –np SIZE convexity来运行该串匹配程序，其中SIZE是所使用的处理器个数。本实例中使用了SIZE=4个处理器。
　　mpirun –np 4 convexity
运行结果：
please input all the vertexes!
first is number!
please input the Number:14
please input the vertex:
1 0
3 0
0 2
0 3.6
8 2
8 6
3 4
3 8
6 8
7.5 7.8
7.8 7.5
7.5 7.5
0.4 0.3
0.3 0.4
输出的是第0部分的点
0.00,2.00
0.30,0.40
0.40,0.30
1.00,0.00
输出的是第3部分的点
3.00,8.00
0.00,3.60
0.00,2.00
输出的是第2部分的点
8.00,6.00
7.80,7.50
7.50,7.80
6.00,8.00
输出的是第1部分的点
3.00,0.00
8.00,2.00
说明：输入了14个点，其中的8个点是极点，2个点在凸壳的内部，另外4个点在四个角上。将四个部分上的点按照序号连接起来就可以得到最小的凸多边形。	

