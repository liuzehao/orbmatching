'''
@Author: haoMax
@Github: https://github.com/liuzehao
@Blog: https://blog.csdn.net/liu506039293
@Date: 2019-10-22 17:42:53
@LastEditTime: 2019-10-23 10:20:08
@LastEditors: haoMax
@Description: 
'''
import cv2
import os
# 自定义计算两个图片相似度函数
def img_similarity(img1_path, img2_path):
    """
    :param img1_path: 图片1路径
    :param img2_path: 图片2路径
    :return: 图片相似度
    """
    try:
        # 读取图片
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)#trainning picture
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        w ,h = img1.shape
        w2,h2=img2.shape
        img1=cv2.resize(img1,(h2,w2))
        # img1 = img1[0 + 45:h]
        # img2 = img2[0 + 45:h1]
        cv2.imshow("img1", img1)
        cv2.imshow("img2", img2)
        cv2.waitKey(0)
 
 
        # 初始化ORB检测器
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        # 提取并计算特征点
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
 
        # knn筛选结果
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)
        cv2.imshow("img3:",img3)
        cv2.waitKey(0)
        # 查看最大匹配点数目
        good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
        # print(len(good))
        # print(len(matches))
        similary = float(len(good))/len(matches)
        if similary>0.1:
            print("判断为ture,两张图片相似度为:%s" % similary)
        else:
            print("判断为false,两张图片相似度为:%s" % similary)
        return similary
 
    except:
        print('无法计算两张图片相似度')
        return '0'
if __name__ == '__main__':
    name1='test_target.jpg'
    # similary 值为0-1之间,1表示相同
    jpg_path='./cmp'
    for i in os.listdir(jpg_path):
        print(i)
        name2=os.path.join(jpg_path,i)

        similary = img_similarity(name1, name2)