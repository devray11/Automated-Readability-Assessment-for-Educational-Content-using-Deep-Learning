import json
from Model import CognitiveLoadClassifier

try:
    clf = CognitiveLoadClassifier()
    clf.load_model('cognitiveload_model.pkl')
    sample = "The little cat sits on the warm windowsill in the morning sun. She watches the birds fly around outside."
    result = clf.predict(sample)
    print(json.dumps(result, indent=2))
except Exception as e:
    print('Error during test prediction:', str(e))
    import traceback
    traceback.print_exc()
