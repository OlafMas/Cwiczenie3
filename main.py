from PIL import Image
from pathlib import Path
import numpy as np
from skimage.feature import graycomatrix,graycoprops
import csv

patch_size = 128
dataset=[]


    classes = ['panele', 'tynk', 'laminat']
    root = Path(R,'C:\Users\Olaf\Tek')
    out_dir= root/'patches'
    for klasa in classes:
        klasa_path=root / klasa
        for filepath in klasa_path.iterdir():
            image = Image.open(str(filepath))
            for i in range(10):
                xo=np.random.randint(image.width)
                yo=np.random.randint(image.height)

                patch=Image.crop(xo,yo,patch_size,patch_size)
                patch.save(out_dir/klasa/filepath.stem,+,f'_{i:o2d})
                patch_gs=patch.convert('L')
                features=get_glcm(patch)
                features +=[klasa]
                dataset +=[features]

            header=['', '', 'category'] #wygenerowac cechy
            with open('output_data.csv','w',newline='')as out_csv:
                 csvwriter=csv.writer(out_csv,delimiter='\t')
                 csvwriter.writerow(header)
                 csvwriter.writerows(dataset)

                 feature_names=['dissimilrity','contrast','correlation']
                 angs=[0, 48, 90, 135]
                 dists=[1,3,5]


def get_glcm(patch: Image)->List[float]:
    patch_64 = np.array(patch)
    patch_64 /= np.max(patch_64)
    patch_64 *= 63
    patch_64 = patch_64.astype(np.unit8)
    graycomatrix(patch_64,dists,angs,64,True,True)
    feature_vector = []:
    for feature in features:
        data=graycoprops(glcm,feature).flatten()
        feature_vector.extends[data.tolist()]
    return feature_vector

'''
features = crop_and_get_features([128,128])

full_features_names=get_full_names()
full_features_names.append('Category')

df=DataFrame(data=features,columns=full_features_names)
df.to_csv('textures_data2.csv',sep=',',index=False)

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

classifier=svm.SVC()
data=np.array(features)
X = data[:,:-1]
Y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print(acc)
'''