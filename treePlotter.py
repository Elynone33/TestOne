# -*- coding:UTF-8 -*-


import matplotlib.pyplot as plt

#使用文本注解绘制树节点
#定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


#首先创建了一个新图形并清空绘图区，然后在绘图区上绘制两个代表不同类型的树节点
def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot().ax1 = plt.subplot(111, frameon=False)
    plotNode(u'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(u'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

#绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt,
                            textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType,
                            arrowprops=arrow_args)



if __name__=="__main__":
    createPlot()