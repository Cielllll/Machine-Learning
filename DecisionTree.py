#!/usr/bin/env python
# coding: utf-8

# In[29]:


from math import log
import operator
import treePlotter

###数据预处理
count = len(open('traindata.txt','r').readlines())-2
#print("  训练数据集个数：",count)
A = []
A_row = 0        
f = open('traindata.txt')               
lines = f.readlines();          
for line in lines:#去掉头尾
    if line.strip() == 'traindata=[' or line.strip() == '];':
        continue
    else:     
        list = line.strip('\n').split('\t')#换行/制表
        y = []
        for x in list:
            x = float(x)
            y.append(x)
        A.append(y)                 
        A_row += 1    
#print("  特征个数：",(len(A[0]) - 1))
#print("  构造矩阵：",A)


###分裂节点选择

# 计算香农熵（信息熵）
def calcShannonEnt(dataSet):
    #返回数据集行数
    num=len(dataSet)
    #保存每个标签（label）出现次数的字典
    labelCounts={}
    #对每组特征向量进行统计
    for featVec in dataSet:
        currentLabel=featVec[-1]                     #提取分类标签
        if currentLabel not in labelCounts.keys():   #如果标签没有放入统计次数的字典，添加进去
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1                 #label计数
    shannonEnt=0.0                                   #信息熵
    #计算信息熵
    for key in labelCounts:
        p=float(labelCounts[key])/num                #选择该标签的概率
        shannonEnt-=p*log(p,2)                       #利用公式计算
    return shannonEnt                                #返回信息熵


# 按照给定特征（所在的下标）axis划分数据集，当指定特征axis等于value时返回去掉该特征的数据集
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 按照给定的特征（所在的下标）axis与数值value，将数据集分为不大于和大于两部分
def splitDataSetForSeries(dataSet, axis, value):
    eltDataSet = [] #不大于划分值的集合
    gtDataSet = []  #大于划分值的集合
    #进行划分，保留该特征值
    for feat in dataSet:
        if feat[axis] <= value:
            eltDataSet.append(feat)
        else:
            gtDataSet.append(feat)
    return eltDataSet, gtDataSet


# 计算连续值的信息增益率，i为对应的特征值下标，baseEntropy为基础信息熵，返回信息增益率和当前的划分点
def calcInfoGainForSeries(dataSet, i, baseEntropy):
    #记录最大的信息增益率
    maxInfoGainRate = 0.0
    #最好的划分点
    bestMid = -1
    #得到数据集中所有的当前特征值列表
    featList = [example[i] for example in dataSet]
    #得到分类列表
    classList = [example[-1] for example in dataSet]
    #组合
    dictList = dict(zip(featList, classList))
    #按照第一个元素的次序将其从小到大排序
    sortedFeatList = sorted(dictList.items(), key=operator.itemgetter(0))
    #计算连续值有多少个
    numberForFeatList = len(sortedFeatList)
    #计算划分点，保留三位小数
    midFeatList = [round((sortedFeatList[i][0] + sortedFeatList[i+1][0])/2.0, 3)for i in range(numberForFeatList - 1)]
    k = 1
    #计算出各个划分点信息增益率
    for mid in midFeatList:
        #将连续值划分为不大于当前划分点和大于当前划分点两部分
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, i, mid)
        #计算两部分的特征值熵和权重的乘积之和
        newEntropy = len(eltDataSet)/len(sortedFeatList)*calcShannonEnt(eltDataSet) + len(gtDataSet)/len(sortedFeatList)*calcShannonEnt(gtDataSet)
        #计算出信息增益
        infoGain = baseEntropy - newEntropy
        a=len(eltDataSet)/len(dataSet)
        b=len(gtDataSet)/len(dataSet)
        instrinsicInfo = -a*log(a,2)-b*log(b,2)
        InfoGainRate = infoGain/instrinsicInfo
       # print('      当前划分值为：' + str(mid) + '，此时的信息增益率为：' + str(InfoGainRate))
        if k == 1:
            bestMid = mid
            maxInfoGainRate = InfoGainRate
            k = 0
        if InfoGainRate > maxInfoGainRate:
            bestMid = mid
            maxInfoGainRate = InfoGainRate
    return maxInfoGainRate, bestMid


# 根据信息增益率来选择最好的数据集划分特征
def chooseBestFeatureToSplit(dataSet, labels):
    #得到数据的特征值总数
    numFeatures = len(dataSet[0]) - 1
    #计算出基础信息熵
    baseEntropy = calcShannonEnt(dataSet)
    #基础信息增益率为0.0
    bestInfoGainRate = 0.0
    #最好的特征值
    bestFeature = -1
    #最好的划分点
    bestSeriesMid = 0.0
    #对每个特征值进行求信息熵
    for i in range(numFeatures):
        #得到数据集中所有的当前特征值列表
        featList = [example[i] for example in dataSet]
       # print('  当前划分属性为：' + str(labels[i]))
        InfoGainRate, bestMid = calcInfoGainForSeries(dataSet, i, baseEntropy)
       # print('  当前特征值为：' + labels[i] + '，对应的信息增益率为：' + str(InfoGainRate))
        
        #如果当前的信息增益比原来的大
        if InfoGainRate > bestInfoGainRate or i==0:
            #最好的信息增益
            bestInfoGainRate = InfoGainRate
            #新的最好的用来划分的特征值
            bestFeature = i
            bestSeriesMid = bestMid
    print('\n 本次分裂特征节点：' + labels[bestFeature]+',最优分裂特征值为：'+ str(bestSeriesMid) + ' ')
    return bestFeature, bestSeriesMid


# 找到次数最多的类别标签
def majorityCnt(classList):
    classCount={}
    #统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
            classCount[vote]+=1
        #根据字典的值做降序排列
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]



### 生成决策树

# 创建决策树
def createTree(dataSet, labels):
    #拿到所有数据集的分类标签
    classList = [example[-1] for example in dataSet]
    #统计第一个标签出现的次数，与总标签个数比较，如果相等则说明当前列表中全部都是一种标签，此时停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #计算第一行有多少个数据，如果只有一个的话说明所有的特征属性都遍历完了，剩下的一个就是类别标签
    if len(dataSet[0]) == 1:
        #返回剩下标签中出现次数较多的那个
        return majorityCnt(classList)

    #选择最好的划分特征，得到该特征的下标
    bestFeat = chooseBestFeatureToSplit(dataSet=dataSet, labels=labels)
    #得到最好特征的名称
    bestFeatLabel = ''
    #记录此刻是连续值还是离散值,1连续，2离散
    flagSeries = 0
    #如果是连续值，记录连续值的划分点
    midSeries = 0.0
    #如果是元组的话，说明此时是连续值
    if isinstance(bestFeat, tuple):
        #重新修改分叉点信息
        bestFeatLabel = str(labels[bestFeat[0]]) + ' 小于 ' + str(bestFeat[1]) + '?'
        #得到当前的划分点
        midSeries = bestFeat[1]
        #得到下标值
        bestFeat = bestFeat[0]
        #连续值标志
        flagSeries = 1
    else:
        #得到分叉点信息
        bestFeatLabel = labels[bestFeat]
        #离散值标志
        flagSeries = 0

    #使用字典来存储树结构，分叉处为划分的特征名称
    myTree = {bestFeatLabel: {}}
    #得到当前特征标签的所有可能值
    featValues = [example[bestFeat] for example in dataSet]

    # 连续值处理
    if flagSeries:
        #将连续值划分为不大于当前划分点和大于当前划分点两部分
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, bestFeat, midSeries)
        #得到剩下的特征标签
        subLabels = labels[:]
        #递归处理小于划分点的子树
        subTree = createTree(eltDataSet, subLabels)
        myTree[bestFeatLabel]['如果小于'] = subTree
        #递归处理大于当前划分点的子树
        subTree = createTree(gtDataSet, subLabels)
        myTree[bestFeatLabel]['如果大于'] = subTree
        return myTree

    # 离散值处理
    else:
        #将本次划分的特征值从列表中删除掉
        del (labels[bestFeat])
        #唯一化，去掉重复的特征值
        uniqueVals = set(featValues)
        #遍历所有的特征值
        for value in uniqueVals:
            #得到剩下的特征标签
            subLabels = labels[:]
            #递归调用，将数据集中该特征等于当前特征值的所有数据划分到当前节点下，递归调用时需要先将当前的特征去除掉
            subTree = createTree(splitDataSet(dataSet=dataSet, axis=bestFeat, value=value), subLabels)
            #将子树归到分叉处下
            myTree[bestFeatLabel][value] = subTree
        return myTree


shannonEnt = calcShannonEnt(A)
#print("  数据集的信息熵为：",shannonEnt)
#print("\n")
labels=['x1','x2','x3','x4']
"""
myTree=createTree(A,labels)
print("\n")
print(myTree)
"""

if __name__ == '__main__':
    """
    处理连续值时候的决策树
    """
   # dataSet, labels, labels_full = createDataSet()
    # chooseBestFeatureToSplit(dataSet, labels)
    myTree = createTree(A, labels)
    #print(myTree)
    treePlotter.createPlot(myTree)


### 测试数据集


# 定义一个使用决策树分类的函数,输入字典格式的决策树inputTree和特征名featLabels，以及一项测试数据testVec
def classify(inputTree,featLabels,testVec):
    # 获取决策树节点
    firstStr = next(iter(inputTree))
    # 下一个字典
    secondDict = inputTree[firstStr]
    # 特征索引
    featIndex = 0
    # 检查字典键是否是特征名
    if firstStr not in featLabels:
        # 如果不是，通过空格对键进行切分，获取特征名firstStrs[0]和特征分裂值testValue，并查找特征索引featIndex
        firstStrs = firstStr.split(" ")
        featIndex = featLabels.index(firstStrs[0])
        testValue = float(firstStrs[2][:-1])
    else:
        # 如果是，特征名即为键名firstStr，并查找特征索引featIndex
        featIndex = featLabels.index(firstStr)

    # 分类标签
    classLabel = 0
    for key in secondDict.keys():
        if testVec[featIndex] == key             or (key == "如果小于" and testVec[featIndex] <= testValue)             or (key == "如果大于" and testVec[featIndex] > testValue):
            # 判断字典中是否还包含新的字典
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else: classLabel=secondDict[key]
    return classLabel   


count = len(open('testdata.txt','r').readlines())
count = count - 2
#print("测试数据集个数：",count)

#读取测试数据集，预处理
B = []
B_row = 0                   
lines = ( open('testdata.txt')).readlines();          
for line in lines:
    if line.strip() == 'testdata=[' or line.strip() == '];':
        continue
    else:     
        list = line.strip('\n').split('\t')
        y = []
        for x in list:
            x = float(x)
            y.append(x)
        B.append(y)                 
        B_row += 1 



#计算测试准确率
rNum = B_row
for i in range(B_row):
    if resultlist[i] != B[i][-1]:
        rNum -=1 
accurancy = rNum/B_row
print("\n Testing Accurancy: "'%.3f'%accurancy)

#打印结果
test_classes = []
for i in range(B_row):
    result = classify(myTree,['x1','x2','x3','x4'],B[i][:-1])
    resultlist.append(result)
    #print("第%d个数据类别为%d" % (i+1,result) )  
    


# In[ ]:




