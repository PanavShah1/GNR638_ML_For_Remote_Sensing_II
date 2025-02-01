# GNR638 Assignments

## Assignment 1

### Task - SIFT Classification of the UC Merced dataset
Use the UC Merced dataset.
Take 70% data per class for training, remaining 10% for validation (deciding what should be the optimal number of codewords), and testing on the remaining 20%
You can use the k-fold cross-validation strategy.
You need to report the classification accuracy, a graph showing how does the accuracy changes as you use different number of codewords in clustering, a t-SNE visualization of the keypoints (each is 128 dimension in SIFT)

### How to run the code
- Download the UC Merced dataset from http://weegee.vision.ucmerced.edu/datasets/landuse.html and save it in the <code>/datasets</code> directory<br>
```
  datasets/
  |--UCMerced_LandUse/
     |--Images/
```
- Go to assignment 1 directory <code>cd assignment-1</code>
- In <code>main.py</code> change <code>VOCAB_SIZE</code> to the desirable Bag of Words Vocabulary size
- Run main.py by <code>python3 main.py</code>
