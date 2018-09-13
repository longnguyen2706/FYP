import matplotlib.pyplot as plt
import numpy as np
import operator

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')


# cm =[[0.88284278, 0.0, 0.0, 0.0, 0.0, 0.10128641, 0.00973151, 0.0, 0.00613929, 0.0], [0.0, 0.96236999, 0.0, 0.0, 0.0, 0.0, 0.01483775, 0.0, 0.0, 0.02279226], [0.00166667, 0.0, 0.87893519, 0.0, 0.09740741, 0.01990741, 0.0, 0.0, 0.00208333, 0.0], [0.0, 0.0, 0.0, 0.99545455, 0.00454545, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.04275444, 0.0, 0.94510271, 0.0, 0.0, 0.0, 0.01214286, 0], [0.05175254, 0.0, 0.00434174, 0.0, 0.0, 0.85260748, 0.07942968, 0.0, 0.01186856, 0.0], [0.0, 0.0, 0.0, 0.00653699, 0.0, 0.0171304, 0.95709281, 0.0, 0.0192398, 0.0], [0.0, 0.0, 0.00750916, 0.0, 0.00208333, 0.0, 0.0, 0.99040751, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.01644097, 0.03204561, 0.0, 0.95151343, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
# cm = np.asarray(cm)
# class_names = ['lysosome', 'microtubules', 'golgi gpp', 'nucleus', 'golgi gia', 'endosome', 'er', 'nucleolus', 'mitochondria', 'actinfilaments']

#
# cm = [[9.66871323e-01, 3.31286766e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#       [0.00000000e+00, 9.69182351e-01, 2.26564611e-02, 0.00000000e+00,0.00000000e+00, 8.16118837e-03],
#       [0.00000000e+00, 2.83669553e-02, 9.02972593e-01, 5.08108570e-02,1.35998312e-02, 4.24976317e-03],
#       [0.00000000e+00, 1.58426836e-03, 3.94995606e-02, 9.48603044e-01,3.14178877e-03, 7.17133853e-03],
#       [0.00000000e+00, 9.13242009e-04, 7.19074023e-03, 7.97064646e-03,9.73654158e-01, 1.02712136e-02],
#       [2.74074074e-03, 5.17976513e-03, 2.89537182e-03, 1.10897524e-02,1.18646769e-02, 9.66229693e-01]]
# cm = np.asarray(cm)
# class_names = ['cytoplasmatic', 'coarse speckled', 'fine speckled', 'homogeneous', 'centromere', 'nucleolar']
#


unsorted_cm = [[0.99457215, 0., 0., 0., 0., 0., 0.00542785, 0., 0., 0., ],
               [0.02665258, 0.9350808,  0.,         0.,         0.,         0.,0.,         0.01575787, 0.02250875, 0.        ],
               [0.,         0.,         1.,         0.,         0.,         0.,0.,         0.,         0.,         0.        ],
               [0.,         0.,         0.,         0.99570707, 0.,         0.,  0.00429293, 0.,         0.,         0.        ],
               [0.,         0.00474334, 0.,         0.,         0.86694019, 0.,0.11547079, 0.00638889, 0.00645679, 0.        ],
               [0.,         0.01745976, 0.01122211, 0.,         0.,         0.97131813,0.,         0.,         0.,         0.        ],
               [0.,         0.,         0.,         0.,         0.02146309, 0.,0.97853691, 0.,         0.,         0.        ],
               [0.,         0.06281828, 0.,         0.,         0.,         0.,0.,         0.83832072, 0.00727395, 0.09158705],
               [0.,         0.05172419, 0.,         0.,         0.,         0.,0.,         0.02074346, 0.92753235, 0.        ],
               [0.,         0.00805676, 0.,         0.,         0.,         0.,0.,         0.09852115, 0.02050887, 0.87291322]]
unsorted_cm = np.asarray(unsorted_cm)
class_names = ['dna', 'er', 'actinfilaments', 'nucleolus', 'golgi gpp', 'microtubules', 'golgi gia', 'endosome', 'mitochondria', 'lysosome']

dict = []
for i in range (0, len(unsorted_cm)):
    cm_i = unsorted_cm[i]
    conf_dict = []
    for j in range(0, len(cm_i)):
        conf_dict.append({'ele': cm_i[j], 'label': class_names[j]})
    print(conf_dict)
    conf_dict.sort(key=operator.itemgetter('label'))
    print('sorted conf dict')
    print(conf_dict)

    conf_arr =[]
    for item in conf_dict:
        conf_arr.append(item['ele'])
    class_name = class_names[i]
    dict.append({'label': class_name, 'conf_arr': conf_arr})
print (dict)
dict.sort(key=operator.itemgetter('label'))
print('sorted dict')
print(dict)

sorted_cm = []
sorted_class_names = []
sorted_labels = []
for i, item in enumerate(dict):
    sorted_cm.append(item['conf_arr'])
    sorted_labels.append(item['label'])
    sorted_class_names.append("C"+str(i+1))
sorted_cm = np.asarray(sorted_cm)
print(sorted_cm)
print(sorted_class_names)
print(sorted_labels)
plt.figure()
plot_confusion_matrix(sorted_cm, sorted_class_names)
plt.show()