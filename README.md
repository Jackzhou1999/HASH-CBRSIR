# HASH-CBRSIR
## 目录结构
CBRSIR-hash  
-Dataset  
  --test  
  --train  
-Model_save
-Model
  --NET.py
  --alexnet.pth
-Load_data.py
-Retravel_test.py
-Split_Dataset.py
-Train_Net.py


## 步骤
1. 运行Split_Dataset.py拆分数据集,train/test,每类80%训练集,20%测试集
2. 运行Train_Net.py训练模型
3. 运行Retrivel_test简单评估模型
