from PIL import Image
from pathlib import Path
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import csv
import pandas as pd


patch_size = 128
dataset=[]
angs=[0, 48, 90, 135]
dists=[1,3,5]
feature_names = ['dissimilarity', 'contrast', 'correlation']

def get_glcm(patch: Image)->list[float]:
    patch_64 = np.array(patch)
    patch_64 = patch_64 / np.max(patch_64)
    patch_64 *= 63
    patch_64 = patch_64.astype(np.uint8)
    glcm = greycomatrix(patch_64,dists,angs,64,True,True)
    feature_vector = []
    for feature in feature_names:
        data = greycoprops(glcm,feature).flatten()
        feature_vector.extend(data.tolist())
    return feature_vector

classes = ['panele', 'tynk', 'laminat']
root = Path(R'C:\Users\Olaf\Tek')
out_dir= root/'patches'
for klasa in classes:
    klasa_path=root / klasa
    for filepath in klasa_path.iterdir():
        image = Image.open(str(filepath))
        for i in range(10):
            xo=np.random.randint(image.width)
            yo=np.random.randint(image.height)

            patch=image.crop((xo,yo,xo+patch_size,yo+patch_size))
            # patch.save(out_dir/klasa/f'{filepath.stem}_{i:02d}.jpg')
            patch_gs=patch.convert('L')
            features=get_glcm(patch_gs)
            features +=[klasa]
            dataset +=[features]

        header=['', '', 'category'] #wygenerowac cechy
        with open('output_data.csv','w',newline='')as out_csv:
             csvwriter=csv.writer(out_csv,delimiter='\t')
             # csvwriter.writerow(header)
             csvwriter.writerows(dataset)


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix

from matplotlib import cm


classifier=svm.SVC(gamma='auto')
data = pd.read_csv('output_data.csv', header=None, sep='\t')
data = np.array(data)
X = data[:,:-1]
Y = data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print(acc)

cmat = confusion_matrix(y_test, y_pred, normalize='true')

print(cmat)

import matplotlib.pyplot as plt
plot_confusion_matrix(classifier, x_test, y_test)
plt.show()
