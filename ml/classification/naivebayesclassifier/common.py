from numpy import *

def loadDataSet():#数据格式
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]# 1 侮辱性文字 ， 0 代表正常言论
    return postingList,classVec

def createVocabList(dataSet):#创建词汇表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建并集
        # print(len(vocabSet))
    return list(vocabSet)

def bagOfWord2VecMN(vocabList,inputSet):#根据词汇表，讲句子转化为向量
    # print(len(vocabList))
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    # print(len(returnVec))
    return returnVec

#训练
def trainNB0(trainMatrix,trainCategory):
    # print(trainMatrix)
    # print(trainCategory)
    numTrainDocs = len(trainMatrix)
    # print(sum(trainCategory))
    # print(float(numTrainDocs))
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # print(pAbusive)
    p0Num = ones(numWords);
    p1Num = ones(numWords)#计算频数初始化为1
    # print(p0Num)
    # print(p1Num)
    p0Denom = 2.0;p1Denom = 2.0                  #即拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            print("p1Num",trainMatrix[i])
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            print("p2Num", trainMatrix[i])
            p0Denom += sum(trainMatrix[i])
        # print(p0Num)
        print(p0Denom)
    p1Vect = log(p1Num/p1Denom)#注意
    print(p1Vect)
    p0Vect = log(p0Num/p0Denom)#注意
    print(p0Vect)
    return p0Vect,p1Vect,pAbusive#返回各类对应特征的条件概率向量
                                 #和各类的先验概率
#分类
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    print(vec2Classify)
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)#注意
    p0 = sum(vec2Classify * p0Vec) + log(1-pClass1)#注意
    if p1 > p0:
        return 1
    else:
        return 0