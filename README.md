# Packer-Classifier

## `./gadgets/`

一些小功能的实现。

* `calc_md5.py`：计算文件MD5并将文件名改为对应MD5
* `enable_check.py`：用于求出并删去某两目录的差集
* `ggs.py`：部分重要函数的实现，如求准确率、早停法、混淆矩阵等
* `yara_check.py`：匹配yara规则
* `packer.yar`：部分壳的yara规则

## `./my_models/`

一些神经网络模型的实现。

* `En_De.py`：编码器-解码器框架，编码器主要为GRU，解码器为带软注意的全连接
* `ODEnet.py`：自己的ODEnet实现
* `SE_ResNet`：自己的SE-net实现，最后没采用
* `my_transfomer.py`：一种transfomer的实现，最后没采用

## `./my_sandbox/`

使用虚拟机做类沙箱操作，用于对PE文件预处理，也就是提取前32*32条指令信息。

* `run_my_sandbox.py`：主机执行的脚本
* `my_sandbox_script.py`：放在虚拟机中的脚本
* `get_features.py`：对脚本中特征提取步骤的类实现

## `./pin/`

编写的Pintools。

* `itrace.cpp`：源码
* `itrace_x86.dll`：32位
* `itrace_x64.dll`：64位

## `./Datasets/`

数据集的实现，包含词汇表、序列化词汇表、向量器和torch的数据集实现。

* `img_datasets.py`：机器码数据集的单独实现
* `ins_datasets.py`：反汇编指令数据集的单独实现
* `datasets.py`：两种数据集的整合

## `./pre_data/`

对从沙箱中提取的信息进行预处理。

* `settings.py`：一些参数的设置
* `get_csv.py`：预处理的脚本
* `preprocess.py`：对脚本中预处理步骤的类实现

## `./experiments/`

实验中模型参数、向量器和各种作图的存储位置。

## `./old_temp/`

在实现过程中被淘汰的脚本。

## `./images/`

模块流程图。

## `./`

* `img_train.py`：机器码模型的单独训练脚本
* `ins_train.py`：反汇编指令模型的单独训练脚本
* `main_train.py`：模型整合的训练脚本
* `main_train.ipynb`：谷歌Colaboratory训练的IPython notebook
* `presentation.ipynb`：系统演示的IPython notebook