#coding=utf-8
import operator
from math import log

#计算给定数据集的香农熵---度量数据集的无序程度
def calcshannonEnt(dataSet):
    # 首先计算数据集中实例的总数
    numEntries = len(dataSet)
    # 创建一个数据字典
    labelCounts = {}
    for featVec in dataSet:
        #字典的键值是最后一列的值
        currentLabel = featVec[-1]
        #如果当前键值不存在，就扩展字典并将当前键值加入字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0.0
            #每个键值都记录了当前类别出现的次数
        labelCounts[currentLabel] += 1.0
    shannonEnt = 0.0
    for key in labelCounts:
        #使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key]) / numEntries
        #用这个概率计算香农熵，统计所有类标签发生的次数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#函数功能：按照给定特征划分数据集
#输入：待划分的数据集dataSet 划分数据集的特征axis 需要返回的特征的值value
def splitDataSet(dataSet,axis,value):
    #为了不修改原始数据集，在这里创建一个新的列表对象
    retDataSet = []
    #遍历数据集中的每一个元素，一旦发现符合要求的值，将其添加到新创建的列表中
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#函数功能：选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    #计算整个数据集的原始香农熵
    baseEntropy = calcshannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    #遍历数据集中的所有特征
    for i in range(numFeatures):
        #将数据集中所有第i个特征值或者所有可能存在的值写入这个新list中
        featList = [example[i] for example in dataSet]
        #使用集合{set}数据类型，
        uniqueVals = set(featList)
        newEntropy = 0.0
        #计算每种划分方式的信息熵
        for value in uniqueVals:
            #对每个唯一属性划分一次数据集
            subDataSet = splitDataSet(dataSet, i, value)
            #prob相当于：包含该属性所取值的样本数/总样本数
            prob = len(subDataSet)/float(len(dataSet))
            #对所有唯一特征值得到的熵求和，得到新香农熵
            newEntropy += prob * calcshannonEnt(subDataSet)
        #信息增益是熵的减少（原始香农熵-新香农熵）
        infoGain = baseEntropy - newEntropy
        #计算最好的信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    #返回最好特征划分的索引值
    return bestFeature

#使用该函数挑选出出现次数最多的类别作为返回值
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #利用operator操作键值排序字典，并返回出现次数最多的分类名称
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#创建树的函数代码
#输入两个参数：数据集和标签列表
def createTree(dataSet,labels):
    #创建名为classList的列表变量，包含了数据集的所有类标签
    classList = [example[-1] for example in dataSet]
    #递归的第一个停止：所有类标签完全相同，返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #递归的第二个停止：使用完了所有特征但不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        #挑选出出现次数最多的类别作为返回值
        return majorityCnt(classList)
    #开始创建树
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLable = labels[bestFeat]
    #用myTree存储树的所有信息
    myTree = {bestFeatLable: {}}
    #将当前数据集选取的最好特征存储在变量bestFeat中，得到列表包含的所有属性值
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    #遍历当前所选特征包含的所有属性值，在每个数据集划分上递归调用函数createTree(),得到的返回值插入字典变量myTree中
    for value in uniqueVals:
        #复制类标签并将其存储在新列表变量subLabels中
        subLables = labels[:]
        myTree[bestFeatLable][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLables)
    return myTree


