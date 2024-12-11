Example:
编译：mpicc rand_match.c –o rand_match
运行：可以使用命令 mpirun –np SIZE rand_match m n来运行该串匹配程序，其中SIZE是所使用的处理器个数，m表示文本串长度，n为模式串长度。本实例中使用了SIZE=3个处理器，m=7，n=2。
　　mpirun –np 3 rand_match 7 2
运行结果：
one node 0 n=7,m=2,p=41
one node 0 Text=00
one node 0 Pattern=01
on node 0 match pos is 2
one node 1 n=7,m=2,p=41
one node 1 Text=10
one node 1 Pattern=01
on node 1 match pos is 2
one node 2 n=7,m=2,p=41
one node 2 Text=100
one node 2 Pattern=01
说明：该运行实例中，令文本串长度为7，随机产生的文本串为0010100，分布在3个节点上；模式串长度为2，随机产生的模式串为01。最后，节点0和1上分别得到一个匹配位置。
