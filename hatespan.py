from flask import Flask, request, jsonify, render_template
import joblib 

app = Flask(__name__)
crf_model = joblib.load("crf_model.pkl")


@app.route('/')
def front():
    return render_template('front.html')

@app.route('/api/detect-hate-span', methods=['POST'])
def detect_hate_span():
    data = request.get_json()
    sentence = data.get('sentence', '')

    predicted_tags = predict_hate_span(sentence)

    result = {
        "sentence": sentence,
        "hateSpeechWords": predicted_tags
    }

    return jsonify(result)
def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def predict_hate_span(sentence):
    sentence_tokens = sentence.split()
    
    sentence_features = [sent2features(sentence_tokens)]
    
    predicted_tags = crf_model.predict(sentence_features)[0]
    
    return predicted_tags


if __name__ == '__main__':
    app.run(port=5500)
