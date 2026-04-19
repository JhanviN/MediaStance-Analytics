# Bilateral sentiment — test set evaluation

## Baseline (TF-IDF + Logistic Regression)

**Accuracy:** 0.8754

### Classification report

```
              precision    recall  f1-score   support

 adversarial     0.8741    0.9020    0.8878       908
 cooperative     0.8584    0.9022    0.8798       961
     neutral     0.8974    0.8210    0.8575       916

    accuracy                         0.8754      2785
   macro avg     0.8766    0.8750    0.8750      2785
weighted avg     0.8763    0.8754    0.8750      2785
```

### Confusion matrix (rows=true, cols=pred)

| | adversarial | cooperative | neutral |
|---|---|---|---|
| adversarial | 819 | 53 | 36 |
| cooperative | 44 | 867 | 50 |
| neutral | 74 | 90 | 752 |

## Advanced (fine-tuned DistilBERT)

**Accuracy:** 0.8786

### Classification report

```
              precision    recall  f1-score   support

 adversarial     0.8804    0.8921    0.8862       908
 cooperative     0.8919    0.8928    0.8924       961
     neutral     0.8627    0.8504    0.8565       916

    accuracy                         0.8786      2785
   macro avg     0.8783    0.8784    0.8784      2785
weighted avg     0.8785    0.8786    0.8786      2785
```

### Confusion matrix (rows=true, cols=pred)

| | adversarial | cooperative | neutral |
|---|---|---|---|
| adversarial | 810 | 42 | 56 |
| cooperative | 35 | 858 | 68 |
| neutral | 75 | 62 | 779 |
