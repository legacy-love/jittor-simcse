# SimCSE-Jittor
## 环境配置
- jittor安装：
```
sudo apt install python3.7-dev libomp-dev
python3.7 -m pip install jittor
python3.7 -m jittor.test.test_example
# 如果您电脑包含Nvidia显卡，检查cudnn加速库
python3.7 -m jittor.test.test_cudnn_op
```
- 其它安装包配置：
```
pip install -r requirements.txt
```
- 数据集导入
```
cd data
bash download_nli.sh
bash download_wiki.sh
```
注意数据集配置可能出现网络问题，可以在浏览器中下载后将`nli_for_simcse.csv`和`wiki1m_for_simcse.txt`放到`data`目录下
- 模型导入  

将预训练模型导入到`model`文件夹下

## 训练
进入`src`目录下
- 无监督
```
bash run_unsup.sh
```
- 有监督
```
bash run_sup.sh
```

## 测试
进入`src`目录下，修改`evaluate.sh`脚本的`model_name_or_path`字段为对应的模型文件
```
bash evaluate.sh
```

## 在使用Jittor时发现的问题
在使用Jittor的`Dataset`和`DataLoader`类时我们发现了一些问题  
按照使用Pytorch的习惯，我们会在继承`Dataset`类实现自己的类时实现`__len__`函数类似如下：
```
def __len__(self):
    return len(self.data)
```
然而，在使用`DataLoader`封装后，`DataLoader`返回的length与`DataSet`一样，而并没有依据batch size改变
这是因为`DataLoader`是定义的一个函数而非封装好的一个类，其定义如下：
```
def DataLoader(dataset: Dataset, *args, **kargs):
    return dataset.set_attrs(*args, **kargs)
```
这就导致直接使用`tqdm`打印进度条时，进度条长度和实际迭代次数不一致的问题，目前的解决方法是向`tqdm`传入`total`的参数设置正确的进度条长度