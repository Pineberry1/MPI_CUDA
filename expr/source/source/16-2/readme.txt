Example:
编译：mpicc sat.c -o sat
例1.变量数为5，子句数目为3，子句中文字数目为3
		(X1 V X2 V X3) ^ (~X1 V X2 V X4) ^ (X0 V ~X2 V ~X4) 
　　文件sample1内容为
　　1 2 3 0 -1 2 4 0 0 -2 -4 0
　　运行：
　　mpirun -np 3 sat sample1
　　NVARS=10 NCLAUSES=3 LENGTH_CLAUSE=3
　　　Satisfied! time0.024580
例2.变量数为5，子句数目为3，子句中文字数目为3
		(X1 V X2 V X3) ^ (~X1 V X2 V X4) ^ (X0 V ~X2 V ~X4) 
　　文件sample2内容为
　　1 2 3 0 -1 2 4 0 0 -2 -4 0
　　运行：
　　mpirun -np 4 test sample
　　NVARS=10 NCLAUSES=3 LENGTH_CLAUSE=3
　　Satisfied! time0.018852
例3.变量数为5，子句数目为3，子句中文字数目为3
		(X1 V X1 V X1) ^ (~X1 V ~X1 V ~X1) ^ (X1 V X1 V X1) 
　　文件sample3内容为
		1 1 1 0 -1 -1 -1 0 1 1 1 0
　　运行：
　　mpirun -np 6 test error
　　　NVARS=10 NCLAUSES=3 LENGTH_CLAUSE=3
　　　Unsatisfied!
　　　
