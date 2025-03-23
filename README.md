# ISE-coursework
Required packages:

&nbsp;&nbsp;&nbsp;&nbsp;pandas\
&nbsp;&nbsp;&nbsp;&nbsp;numpy\
&nbsp;&nbsp;&nbsp;&nbsp;scipy\
&nbsp;&nbsp;&nbsp;&nbsp;re\
&nbsp;&nbsp;&nbsp;&nbsp;nltk\
&nbsp;&nbsp;&nbsp;&nbsp;sktlearn

How to run the code:

  1. python SVM.py
  2. choose a file name (caffe, incubator-mxnet, keras, pytorch, tensorflow) to check the result.

How to replicate the results reported:

&nbsp;&nbsp;&nbsp;&nbsp;For each chosen file (caffe, incubator-mxnet, keras, pytorch, tensorflow), the result of the
  execution will show the AUC metric of SVM and NB. The result of checking if there is a
  statistical significance will also be shown.
  
&nbsp;&nbsp;&nbsp;&nbsp;In this way, all results for 5 files can be replicated.

  For easily comparing the results instead of checking the screenshots, they are listed here:

|         | AUC_SVM | AUC_NB |
|---------|---------|--------|
|  caffe  | 0.7586  | 0.6513 |
|  mxnet  | 0.9112  | 0.7471 |
|  keras  | 0.9072  | 0.6955 |
| pytorch | 0.8585  | 0.7581 |
|    tf   | 0.9263  | 0.7110 |
