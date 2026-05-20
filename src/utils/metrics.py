from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

def calculate_metrics(targets, predictions):
    acc = accuracy_score(targets, predictions)
    recall = recall_score(targets, predictions, average='macro', zero_division=0)
    f1 = f1_score(targets, predictions, average='macro', zero_division=0)
    
    return {'accuracy': acc, 'recall_macro': recall, 'f1_macro': f1}