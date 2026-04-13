# Bilateral sentiment — test set evaluation

## Baseline (TF-IDF + Logistic Regression)

**Accuracy:** 0.5893

### Classification report

```
              precision    recall  f1-score   support

 adversarial     0.6538    0.6538    0.6538        26
 cooperative     0.5000    0.5714    0.5333        14
     neutral     0.5714    0.5000    0.5333        16

    accuracy                         0.5893        56
   macro avg     0.5751    0.5751    0.5735        56
weighted avg     0.5918    0.5893    0.5893        56
```

### Confusion matrix (rows=true, cols=pred)

| | adversarial | cooperative | neutral |
|---|---|---|---|
| adversarial | 17 | 6 | 3 |
| cooperative | 3 | 8 | 3 |
| neutral | 6 | 2 | 8 |

## Advanced (fine-tuned DistilBERT)

**Accuracy:** 0.4643

### Classification report

```
              precision    recall  f1-score   support

 adversarial     0.4643    1.0000    0.6341        26
 cooperative     0.0000    0.0000    0.0000        14
     neutral     0.0000    0.0000    0.0000        16

    accuracy                         0.4643        56
   macro avg     0.1548    0.3333    0.2114        56
weighted avg     0.2156    0.4643    0.2944        56
```

### Confusion matrix (rows=true, cols=pred)

| | adversarial | cooperative | neutral |
|---|---|---|---|
| adversarial | 26 | 0 | 0 |
| cooperative | 14 | 0 | 0 |
| neutral | 16 | 0 | 0 |
