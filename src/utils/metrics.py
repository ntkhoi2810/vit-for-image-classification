from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def calculate_metrics(targets, predictions):
    """Tính toán Accuracy và F1-Score Macro."""
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='macro')
    return {'accuracy': acc, 'f1_macro': f1}