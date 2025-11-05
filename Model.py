# Updated Model.py for Book_Dataset.csv and Index.html integration
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Try to import NLTK with fallback
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available. Using basic text processing.")
    NLTK_AVAILABLE = False

# Try to import textstat with fallback
try:
    from textstat import flesch_kincaid_grade
    TEXTSTAT_AVAILABLE = True
except ImportError:
    print("Warning: textstat not available. Using simplified readability calculations.")
    TEXTSTAT_AVAILABLE = False

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Error: scikit-learn not available. Please install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

import pickle

# Define the new 6-level classification labels
NEW_LABELS = [
    'Grade till 6th',
    'Grade 6-8',
    'Grade 9-10',
    'Grade 11-12',
    'Undergraduate',
    'Postgraduate'
]

def simple_sentence_tokenize(text):
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def simple_word_tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def get_stop_words():
    if NLTK_AVAILABLE:
        try:
            return set(stopwords.words('english'))
        except:
            pass
    return { 'the','and','to','of','a','in','for','is','on','that','it','with','as','by','this','I','you' }

class TextFeatureExtractor:
    def __init__(self):
        self.stop_words = get_stop_words()
    def extract_features(self, text):
        if not text or not text.strip(): return self._default_features()
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
            words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
        else:
            sentences = simple_sentence_tokenize(text)
            words = simple_word_tokenize(text)
        if not sentences or not words: return self._default_features()
        f={}
        f['avg_sentence_length']=len(words)/len(sentences)
        f['type_token_ratio']=len(set(words))/len(words)
        f['lexical_sophistication']=sum(len(w) for w in words)/len(words)
        concrete=['the','and','to','of','a']
        f['word_concreteness']=5.0-(sum(1 for w in words if w in concrete)/len(words))*2.0
        f['mean_dependency_distance']=sum(len(simple_word_tokenize(s))/3.0 for s in sentences)/len(sentences)
        sub_conj=['although','because','if','when','while','unless','until']
        f['subordination_ratio']=sum(text.lower().count(c) for c in sub_conj)/len(sentences)
        if TEXTSTAT_AVAILABLE:
            try: f['flesch_kincaid_grade_level']=flesch_kincaid_grade(text)
            except: f['flesch_kincaid_grade_level']=0.0
        else:
            f['flesch_kincaid_grade_level']=f['avg_sentence_length']*0.39+11.8*(sum(len(re.findall(r'[aeiou]',w)) for w in words)/len(words))-15.59
        if len(sentences)<3: f['smog_index']=5.0
        else:
            comp=[w for w in words if len(w)>=7]
            f['smog_index']=1.043*np.sqrt(len(comp)*(30/len(sentences)))+3.1291
        conn=['and','but','or','so','however','therefore']
        f['connective_frequency']=(sum(1 for w in words if w in conn)/len(words))*100
        ref=['the','this','that','he','she','it','they','his','her','their']
        ws=words
        f['referential_cohesion']=sum(1 for w in ws if w in ref)/len(ws)
        return f
    def _default_features(self): return dict.fromkeys(['avg_sentence_length','type_token_ratio','lexical_sophistication','word_concreteness','mean_dependency_distance','subordination_ratio','flesch_kincaid_grade_level','smog_index','connective_frequency','referential_cohesion'],0.0)

class CognitiveLoadClassifier:
    def __init__(self):
        if not SKLEARN_AVAILABLE: raise ImportError
        self.fe=TextFeatureExtractor()
        self.le=LabelEncoder()
        self.scaler=StandardScaler()
        self.rf=RandomForestClassifier(n_estimators=100,random_state=42)
        self.gb=GradientBoostingClassifier(n_estimators=100,random_state=42)
        self.trained=False
    def train(self,csv_path='Book_Dataset.csv'):
        df=pd.read_csv(csv_path)
        texts,labels=df['text'].tolist(),df['difficulty_level'].tolist()
        feats=[list(self.fe.extract_features(t).values()) for t in texts]
        X=np.array(feats)
        self.le.fit(NEW_LABELS)
        y=self.le.transform(labels)
        Xs=self.scaler.fit_transform(X)
        Xt,xt,yt,yt_=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
        self.rf.fit(Xt,yt); self.gb.fit(Xt,yt)
        self.trained=True
    def predict(self,text):
        if not self.trained: raise ValueError
        fv=np.array([list(self.fe.extract_features(text).values())])
        fv_s=self.scaler.transform(fv)
        rp,rb=self.rf.predict(fv_s)[0],self.rf.predict_proba(fv_s)[0]
        gp,gbp=self.gb.predict(fv_s)[0],self.gb.predict_proba(fv_s)[0]
        prob=(rb+gbp)/2; idx=prob.argmax()
        return {'prediction':self.le.inverse_transform([idx])[0],'confidence':float(prob[idx]),'features':self.fe.extract_features(text),'probabilities':{lbl:float(p) for lbl,p in zip(NEW_LABELS,prob)}}
    def save_model(self,path='cognitiveload_model.pkl'):
        pickle.dump({'rf':self.rf,'gb':self.gb,'le':self.le,'scaler':self.scaler},open(path,'wb'))
    def load_model(self,path='cognitiveload_model.pkl'):
        d=pickle.load(open(path,'rb'))
        self.rf, self.gb, self.le, self.scaler=d['rf'],d['gb'],d['le'],d['scaler']
        self.trained=True

if __name__=='__main__':
    clf=CognitiveLoadClassifier()
    clf.train('Book_Dataset.csv')
    clf.save_model()