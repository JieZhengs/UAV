环境说明：pytorch：2.2.1，python版本：3.10，cuda版本：12.1，GPU：RTX 4090，CPU：Intel(R) Xeon(R) Gold 5418Y * 12核，内存：120GB
UAV-Main标准环境即可运行，ctrgcn、tdgcn和tegcn均需要添加KAN模型。

数据位置：
.data/ 文件夹下

运行方式：
UAV-Main网络运行方式：
1. 该代码使用的是joint、joint_bone和bone模态的数据，需要对官方给的joint模态的数据进行处理成joint_bone、bone进行复现。

2.  进入UAV-Main。

3. 激活你的环境。

4. 分别切换train.config里model的配置改成ctrgcn、tdgcn和tegcn（此外，由于joint（3通道）、joint_bone（6通道）和bone（三通道）分别还需要对ctrgcn、tdgcn和tegcn里的通道代码进行特定修改，具体是tegcn源码只需要修改Model类的channel参数，ctrgcn、tdgcn分别修改Model类的channel参数和TDGC类的in_channels参数（在if语句增加或修改in_channels == 3或6））。

5. 13次运行训练模型请在终端执行：python main.py --config ./config/train.yaml --work-dir ./work_dir/train -model_saved_name ./runs/train --device 0 --batch-size 16 --test-batch-size 16 --warm_up_epoch 0 --only_train_epoch 100（根据你训练的模型修改对应的--work-dir -model_saved_name 路径）。

6. 其中tegcn，tdgcn和ctrgcn运行运行多次joint（4次）、joint_bone（5次）和bone（4次）模态训练，根据所给参数进行训练（直接用参数设置和训练日志文件夹下的内容替换源代码中的文件进行训练即可复现结果）。

7. 将得到的最佳权重（见训练日志文件）进行val测试集上的测试拿到置信度文件共计13种，文件以.npy结尾。


8. 获取融合权重最佳参数分配：
	1. 分别将13种在val上得到的置信度文件路径分别加入到Search_Force.py、Serach_PSO.py和Serach_Nearly.py源码中（具体见源码的参数添加部分）。

	2. 首先运行Search_Force.py代码进行权重分配的初始定位操作（所有初始权重分配搜先采用[1,1,1,1,1,1,1,1,1,1,1,1,1]），然后根据最佳准确率，反复更新初始权值分配参数搜索直至准确率上不去为止则停止搜索，确定最佳权重分配的大致定位（此时，最佳权重分配参数全为整数）。
	
	3. 然后运行Serach_PSO.py代码进行最优第2步得到的权重分配参数进行进一步的最优（可能是局部也可能是全局）搜索，这里的可调参数包括容忍度，PSO惯性权重，全局最佳因子和局部最佳因子，参数调节根据人工观测准确度变化进行可调参数调节。这里将会得到浮点数的权重分配参数，同样的道理，将得到的最佳准确率的权重分配参数反复进行搜索直至准确率上不去为止则停止搜索（此时，最佳权重分配参数全为浮点数）。
	
	4. 最后运行Serach_Nearly.py代码对PSO搜索到的最优权重分配参数做进行最后的调整，将得到的最优权重分配参数进行反复震荡式搜索（第三步最优解附近），可调参数包括容忍度，根据人工观测准确率进行可调参数调节。这里将会得到浮点数的权重分配参数，同样的道理，将得到的最佳准确率的权重分配参数反复进行搜索直至准确率上不去为止则停止搜索（此时，最佳权重分配参数全为浮点数）。
	
9. 制作一个全为1的test_label.npy文件的假标签，用最佳的Train_val权重进行test_joint.npy、test_joint_bone.npy和test_bone.npy（官方给的测试集，用假标签预测）生成对应的置信度文件共计13种。

10. 运行源代码下的Ensemble_MixGCN.py代码，将第8步得到的融合最佳权重参数分配加入，得到test测试集的最终预测置信度文件（本次最佳融合权重参数为：5.873476202521488, 4.678717740853027, 6.603675832620936, 0.2643313882902165, 0.27986643043507275,
-1.1023075930863073, 2.043657545258659, 3.187101109269344, 3.4465320125689645, -1.3164161768069882, 8.067121711896956,
1.2074025263468688, 2.1962578433865425） ，在work_dir中得到预测结果（pred.npy）

11. 我们项目的代码公开在github上：https://github.com/JieZhengs/UAV

12. 注意！！！！！ 所有获取融合权重最佳参数分配的源码为丝滑稀饭组自主设计！！！！！
