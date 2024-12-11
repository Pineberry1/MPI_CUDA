Example:
编译：mpicc app_match.c –o app_match
运行：可以使用命令 mpirun –np SIZE app_match m n k来运行该串匹配程序，其中SIZE是所使用的处理器个数，m表示文本串长度，n为模式串长度，k为允许误差长度。本实例中使用了SIZE=3个处理器，m=7，n=2，k=1。
　　mpirun –np 3 app_match 7 2 1
运行结果：
on node 0 the text is as
on node 0 the pattern is as
Total 2 matched on node 0
on node 1 the text is as
on node 1 the pattern is as
Total 2 matched on node 1
on node 2 the text is bsb
on node 2 the pattern is as
Total 2 matched on node 2
说明：该运行实例中，令文本串长度为7，随机产生的文本串为asasbsb，分布在3个节点上；模式串长度为2，随机产生的模式串为as。最后，节点0、1和2上分别得到两个近似匹配位置。

