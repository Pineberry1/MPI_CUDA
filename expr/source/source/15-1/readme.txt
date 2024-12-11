Example:
编译：mpicc -o closure closure.c -lm
运行：可以使用命令 mpirun –np SIZE closure来运行生成的默认文件closure，其中SIZE是所使用的处理器个数。本实例中使用了4个处理器。
      mpirun –np 4 closure
运行结果：
Input the size of matrix:8
Input the matrix:
0 0 0 0 0 0 0 1
0 0 0 0 0 1 1 0
0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 1
0 0 0 0 0 0 1 1
0 1 0 1 0 0 0 0
0 1 0 0 1 0 0 0
1 0 0 1 1 0 0 0
最终输出结果：
loop4:
1 1 0 1 1 1 1 1
1 1 0 1 1 1 1 1
0 0 1 0 0 0 0 0
1 1 0 1 1 1 1 1
1 1 0 1 1 1 1 1
1 1 0 1 1 1 1 1
1 1 0 1 1 1 1 1
1 1 0 1 1 1 1 1
说明：结果表明除去顶点2为孤立顶点，其余各顶点两两可达。

