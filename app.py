from flask import Flask, request, jsonify, render_template, redirect, url_for
from main import preprocess, TfidVectorizer, cosine_similarity, n_grams
import numpy as np

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


def read_file(file):
    return file.read().decode("utf-8")


def compute_scores(student_processed, memo_processed):

    student_tokens = student_processed.split()
    memo_tokens    = memo_processed.split()

   
    student_bigrams   = n_grams(student_processed, 2)
    memo_bigrams      = n_grams(memo_processed, 2)
    bigram_vectorizer = TfidVectorizer()
    bigram_vectorizer.fit([student_bigrams, memo_bigrams])
    bigram_score = round(float(cosine_similarity(
        bigram_vectorizer.transform(student_bigrams),
        bigram_vectorizer.transform(memo_bigrams))) * 100, 2)

   
    student_trigrams   = n_grams(student_processed, 3)
    memo_trigrams      = n_grams(memo_processed, 3)
    trigram_vectorizer = TfidVectorizer()
    trigram_vectorizer.fit([student_trigrams, memo_trigrams])
    trigram_score = round(float(cosine_similarity(
        trigram_vectorizer.transform(student_trigrams),
        trigram_vectorizer.transform(memo_trigrams))) * 100, 2)

  
    fusion_score = round((bigram_score + trigram_score) / 2, 2)
    verdict = "This answer is flagged for evaluation." if fusion_score < 50 else "This is the proposed grade."

    return {
        
        "bigram_score":      bigram_score,
        "trigram_score":     trigram_score,
        "fusion_score":      fusion_score,
        "student_processed": student_processed,
        "memo_processed":    memo_processed,
        "verdict":           verdict
    }


@app.route("/upload/json", methods=["POST"])
def upload_json():
    student_text = read_file(request.files["student"])
    memo_text    = read_file(request.files["memo"])
    results      = compute_scores(preprocess(student_text), preprocess(memo_text))
    return jsonify(results)


@app.route("/upload", methods=["POST"])
def upload():
    student_text = read_file(request.files["student"])
    memo_text    = read_file(request.files["memo"])
    results      = compute_scores(preprocess(student_text), preprocess(memo_text))
    return render_template("results.html", **results)


if __name__ == "__main__":
    app.run(debug=True)