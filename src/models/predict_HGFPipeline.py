from transformers import pipeline
from sklearn import metrics


labels_str = {'code password log new': 0, 
              'printer print scan attached': 1, 
              'ticket follow': 2}

classifier = pipeline('zero-shot-classification')
candidate_labels = list(labels_str.keys())

pipeline_pred_complete = classifier(list(X_test.values), candidate_labels, hypothesis_template="This is probably a conversation on the topic of {}")
pipeline_pred = [labels_str[pred['labels'][0]] for pred in pipeline_pred_complete]
print(metrics.classification_report(y_test, pipeline_pred))
print(metrics.confusion_matrix(y_test, pipeline_pred))






def evaluation(model, metric, cv, X_train, X_test, y_train, y_test):

    # Traditional 
    predicted = model.predict(X_test)
    accuracy = np.mean(predicted == y_test)
    print("**BASIC:")
    print(accuracy)
    print(model.predict(X_test))
    print(X_test)
    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))

    # Cross va
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
    print(f"\n**CV SCORES BASIC: {scores}, mean {scores.mean()} and std {scores.std()} \n")