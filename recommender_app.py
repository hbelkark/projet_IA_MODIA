import argparse
import gradio as gr
import pickle
from nltk import word_tokenize          
from nltk.stem import SnowballStemmer
import re


# Interface lemma tokenizer from nltk with sklearn
class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, doc):
        doc = doc.lower()
        return [self.stemmer.stem(t) for t in word_tokenize(re.sub("[^a-z' ]", "", doc)) if t not in self.ignore_tokens]


def sentimentPrediction(text):
    prediction = model.predict([text])
    if prediction[0] == 1:
        return "Positive"
    else:
        return "Negative"

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='model_path')
    args = parser.parse_args()
    path = args.model_path

    model = pickle.load(open(path, 'rb'))
    
    gr.Interface(fn=sentimentPrediction, 
                inputs=gr.Textbox(placeholder="Enter a positive or negative sentence here..."),
                outputs="text", 
                interpretation="shap",
                num_shap=5,
                title="Sentiment Analysis",
                description="Enter a review and the model will predict whether it is positive or negative.",
                ).launch(debug=True, share=True);
