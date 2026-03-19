# Task Readiness Report

- Total samples in label sheet: **20**

## Readiness by Task

### food_type
- task_type: `multiclass`
- labeled_samples: **0**
- class_distribution: `{}`
- trainable_now: **False**
- rule: multiclass requires each class >= 4 samples

### chew_side_label
- task_type: `binary`
- labeled_samples: **0**
- class_distribution: `{}`
- trainable_now: **False**
- rule: binary requires each class >= 5 samples
- notes: balanced can remain unlabeled for now

### rhythm_class
- task_type: `binary`
- labeled_samples: **0**
- class_distribution: `{}`
- trainable_now: **False**
- rule: binary requires each class >= 5 samples

### pause_style
- task_type: `binary`
- labeled_samples: **0**
- class_distribution: `{}`
- trainable_now: **False**
- rule: binary requires each class >= 5 samples

### chew_rate_target
- task_type: `regression`
- labeled_samples: **0**
- variance: `nan`
- trainable_now: **False**
- rule: regression requires >= 15 labels with non-trivial variance
- notes: treat as direct analytics unless external ground truth exists
