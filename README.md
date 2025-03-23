# ISE-coursework
Required packages:\
  pandas\
  numpy\
  scipy\
  re\
  nltk\
  sktlearn\

How to run the code:
  1. python SVM.py
  2. choose a file name (caffe, incubator-mxnet, keras, pytorch, tensorflow) to check the result.

How to replicate the results reported:\
  For each chosen file (caffe, incubator-mxnet, keras, pytorch, tensorflow), the result of the
  execution will show the AUC metric of SVM and NB. The result of checking if there is a
  statistical significance will also be shown.\
  In this way, all results for 5 files can be replicated.\

  For easily comparing the results instead of checking the screenshots, they are listed here:\

|         | AUC_SVM | AUC_NB |
|---------|---------|--------|
|  caffe  | 0.7586  | 0.6513 |
|  mxnet  | 0.9112  | 0.7471 |
|  keras  | 0.9072  | 0.6955 |
| pytorch | 0.8585  | 0.7581 |
|    tf   | 0.9263  | 0.7110 |
