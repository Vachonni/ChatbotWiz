from transformers import pipeline

labels_str = {'code password log new': 0, 
              'printer print scan attached': 1, 
              'ticket follow': 2}

classifier = pipeline('zero-shot-classification')
candidate_labels = list(labels_str.keys())

pipeline_pred_complete = classifier(list(X_test.values), candidate_labels, hypothesis_template="This is probably a conversation on the topic of {}")
pipeline_pred = [labels_str[pred['labels'][0]] for pred in pipeline_pred_complete]
print(metrics.classification_report(y_test, pipeline_pred))
print(metrics.confusion_matrix(y_test, pipeline_pred))


