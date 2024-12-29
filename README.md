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
