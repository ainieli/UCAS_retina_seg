## 执行顺序
1. 确定数据集是DRIVE还是CHASEDB1，对不同文件的init_xxx()进行选择

2. 设定数据划分(DRIVE 40张/CHASEDB1 28张分为训练集，验证集和测试集)，并运行UNet_preprocess.py

3. 在代码里面修改注释项来选择模型是UNet还是UNet++

4. 在代码里面修改注释项来选择模型是否使用数据增强

5. 运行代码UNet_train.py(合理修改训练参数，如epochs/weight decay/learning rate)
    <p style="text-indent:2em">ps:目前没有加上学习力调整schedule，可后续补上</p>
    
6. 运行代码UNet_predict.py，获取验证结果
    <p style="text-indent:2em">ps:目前的metrics只设置了Dice系数</p>
    
## PS
1. 当前的验证集和测试集是从所有测试数据中划分得到的，最终写报告的时候的指标应该是所有测试图像的测试结果

2. 目前的代码不够精简，读文件部分偷懒了沿用的以前的菜鸡代码_(:з」∠)_，导致出现了多个文件复制副本，且对文件名、路径名相当敏感；且继续偷懒使用global变量对某些值进行定义可能在不知名的地方出一些bug，欢迎修改\(^o^)/~ 

3. UNet++代码逻辑（训练与测试）是否通顺未进行测试，欢迎测试

4. CHASEDB1测试逻辑是否通顺未进行测试，欢迎测试

5. 有什么想法和建议可以发在群里大家一起讨论下