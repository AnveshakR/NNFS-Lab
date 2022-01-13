import cv2
import numpy as np
from numpy import random
from sklearn.cluster import KMeans
import os

path = r'C:\Users\anves\Documents\College\NNFS\braintumor\test_images'

img=cv2.imread(os.path.join(path,random.choice(os.listdir(path))),0)

height,width=img.shape
cv2.imshow("image",img)
cv2.waitKey(0)

img = img.reshape(height*width,1)

model = KMeans(n_clusters=4)
model.fit(img)

tumorLabel=3

test_img_names = os.listdir(path)

for test_img_name in test_img_names:

    img_path = os.path.join(path,test_img_name)
    img = cv2.imread(img_path,0)
    img_original=cv2.imread(img_path)
    height,width = img.shape
    imgFlatten = img.reshape(height*width,1)
    labels = model.predict(imgFlatten)
    labels2D = labels.reshape(height,width)

    mask = (labels2D == tumorLabel)

    tumorExtracted = np.bitwise_and(mask,img)
    contours,hierarchy = cv2.findContours(tumorExtracted,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print('No. of contours: ',len(contours))

    for index,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if(area>1000):
            img_original = cv2.drawContours(img_original,[cnt],-1,(0,255,255),2)

            x,y,w,h = cv2.boundingRect(cnt)
            img_original = cv2.rectangle(img_original,(x,y),(x+w,y+h),(0,255,0),2)
            img_original = cv2.rectangle(img_original,(x,y),(x+120,y-40),(0,255,0),-1)
            img_original = cv2.putText(img_original,"TUMOR",(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.imshow("predicted image",img_original)
    cv2.waitKey(0)

cv2.destroyAllWindows()