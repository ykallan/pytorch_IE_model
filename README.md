# pytorch_information_extract_model
# 基于pytorch的信息抽取模型

 # 环境要求：
* python>=3.7，3.7、3.8测试无问题
* cuda>=10.1，建议10.2

# 安装依赖：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host tuna.tsinghua.edu.cn
```

# 下载预训练模型:
- 中文bert、albert的pytorch版本下载地址：<https://huggingface.co/>
- 下载的文件存放搭配model_file文件夹下，并修改embedding下对应的 "*embeddeing.py"文件开头的中定义的目录
- word2vec预训练词向量下载地址：<https://github.com/Embedding/Chinese-Word-Vectors>
- 下载的word2vec预训练可能太大导致内存不足，运行下面的语句进行处理（提取出数据集中包含的字向量，其他的词向量不要）
```bash
python ./utils/pertrain_to_numpy.py
```

# 数据集下载： 
- <https://aistudio.baidu.com/aistudio/datasetdetail/11384>
- 下载的数据集放到data文件夹下
  
# 运行:
```bash
# 先处理数据
python ./utils/process_raw_data.py
# 修改config配置
vi config.py
# 修改main.py确定运行哪个框架
vi main.py
# 运行
python main.py
    
```