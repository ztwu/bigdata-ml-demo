from ml.classification.naivebayesclassifier import common

def testingNB():#流程展示
    listOPosts,listClasses = common.loadDataSet()#加载数据
    myVocabList = common.createVocabList(listOPosts)#建立词汇表
    # print(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(common.bagOfWord2VecMN(myVocabList,postinDoc))
    # print(trainMat)
    p0V,p1V,pAb = common.trainNB0(trainMat,listClasses)#训练
    #测试
    testEntry = ['love','my','dalmation']
    thisDoc = common.bagOfWord2VecMN(myVocabList,testEntry)
    print(testEntry,'classified as: ',common.classifyNB(thisDoc,p0V,p1V,pAb))

testingNB()