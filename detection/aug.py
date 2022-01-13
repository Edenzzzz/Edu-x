import albumentations as a
import cv2,os,random
path='E:\\pos_neg\\Positive'
write='E:\\augmented\\positive'
j=0
for file in os.listdir(path):
    img=cv2.imread(os.path.join(path,file))
    angle=random.randint(15,30)
    img=a.rotate(img,angle)
    cv2.imwrite(os.path.join(write,str(j)+file),img)
    j+=1
for file in os.listdir(path):
    img=cv2.imread(os.path.join(path,file))
    angle=random.randint(15,30)
    img=a.hflip(img)
    cv2.imwrite(os.path.join(write,str(j)+file),img)
    j+=1
for file in os.listdir(path):
    img=cv2.imread(os.path.join(path,file))
    angle=random.randint(15,30)
    img=a.rotate(img,angle)
    img=a.hflip(img)
    cv2.imwrite(os.path.join(write,str(j)+file),img)
    j+=1


