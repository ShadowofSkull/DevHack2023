# Import necessary libraries
import os
import fitz
import spacy
from flask import Flask, request, render_template, redirect, url_for
from sentence_transformers import SentenceTransformer, util
from string import punctuation
from collections import Counter
from heapq import nlargest
from spacy.matcher import Matcher
from spacy.lang.en.stop_words import STOP_WORDS
from datasets import load_dataset, Dataset 

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load the Sentence Transformer model
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Load the dataset
job = load_dataset("jacob-hugging-face/job-descriptions")

app = Flask(__name__)

# Convert PDF to text function
def pdf_to_text(pdf_path):
    text = ''
    with fitz.open(pdf_path) as pdf_document:

      for page_num in range(pdf_document.page_count):
          page = pdf_document[page_num]
          text += page.get_text()

    return text

# Define your summarizer function (you can reuse your existing code)
def summarizer(text):
    text = text.strip().lower().replace("\n", "")
    doc = nlp(text)
    
    # To get nouns
    matcher = Matcher(nlp.vocab)
    pattern = [
        {"POS": "PROPN", "OP": "+"},
        {"POS": "NOUN", "OP": "+"},
    ]
    
    pattern1 = [
        {"POS": "NOUN", "OP": "+"},
        {"POS": "PROPN", "OP": "+"},
    ]
    pattern2 = [
        {"POS": "NOUN", "OP": "+"},
    ]
    pattern3 = [
      {"POS": "PROPN", "OP": "+"},
    ]
    matcher.add("label", [pattern, pattern1, pattern2, pattern3], greedy="LONGEST")
    matches = matcher(doc)
    # To sort in the order of occurance in the resume
    matches.sort(key = lambda x: x[1])
    # print(len(matches))
    keyword = []
    stopwords = list(STOP_WORDS)

    # remove stopwords and punctuation
    for mat in matches:
      text = doc[mat[1]: mat[2]]
      if not (text.text in stopwords or text.text in punctuation):
        keyword.append(text.text)
        # print(mat, text.text)
    # print(keyword)

    freq_word = Counter(keyword)
    # print(freq_word.most_common())
    max_freq = Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():
        freq_word[word] = (freq_word[word]/max_freq)
    
    # print(freq_word.most_common(20))
    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]
    # print(sent_strength)
    
    summarized_sentences = nlargest(10, sent_strength, key=sent_strength.get)
    # print(summarized_sentences)

    final_sentences = [ w.text for w in summarized_sentences ]
    summary = ' '.join(final_sentences)
    return(summary)

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # job = load_dataset("jacob-hugging-face/job-descriptions")
        # model = SentenceTransformer('BAAI/bge-large-en-v1.5')

        res = request.form['resume']

    
        res = res.strip().lower()
        res = summarizer(res)

        job_descs = []
        resumes = []
        for i in range(25):
          sum_des = summarizer(job["train"][i]["job_description"])
          job_descs.append(sum_des)
          resumes.append(res)

        #Compute embedding for both lists
        embeddings1 = model.encode(job_descs, normalize_embeddings=True)
        embeddings2 = model.encode(resumes, normalize_embeddings=True)

        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        #Output the pairs with their score
        #for i in range(len(job_descs)):
            #print(f"Score: {cosine_scores[i][i]*100:.2f}, Index: {i}")


        scores = []
        for i in range(25):
          scores.append(str(cosine_scores[i][i])[7:13])
        
        top_scores = nlargest(5, scores)
        best_jobs = []
        for i in range(5):
          idx = scores.index(top_scores[i])
          best_jobs.append([top_scores[i], idx])
        
        job_data = []
        for i in range(5):
          title = job["train"][best_jobs[i][1]]["position_title"]
          name = job["train"][best_jobs[i][1]]["company_name"]
          similarity = float(best_jobs[i][0]) * 100
          job_data.append((title, name, similarity))
          #print(f"#{i + 1} Job position is {title} for {name} company with {similarity:.2f}% similarity rate")
        
        return render_template('result.html', user_resume=res, job_data=job_data)

    return render_template('index.html')

@app.route("/ocr", methods=["POST"])
def ocr():
    if request.method == 'POST':
        # Get file from POST request and save it
        f = request.files['file']
        f.save(f.filename)  

        # Using PyMuPDF to convert PDF to text
        path = os.path.abspath(f'.\{f.filename}')
        text = pdf_to_text(path)
        # Remove the file after processing
        os.remove(f.filename)
        # Summarize the text
        res = summarizer(text)
        # Pass the summarized text to the model
        job_descs = []
        resumes = []
        for i in range(25):
          sum_des = summarizer(job["train"][i]["job_description"])
          job_descs.append(sum_des)
          resumes.append(res)

        #Compute embedding for both lists
        embeddings1 = model.encode(job_descs, normalize_embeddings=True)
        embeddings2 = model.encode(resumes, normalize_embeddings=True)

        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        scores = []
        for i in range(25):
          scores.append(str(cosine_scores[i][i])[7:13])
        
        top_scores = nlargest(5, scores)
        best_jobs = []
        for i in range(5):
          idx = scores.index(top_scores[i])
          best_jobs.append([top_scores[i], idx])
        
        job_data = []
        for i in range(5):
          title = job["train"][best_jobs[i][1]]["position_title"]
          name = job["train"][best_jobs[i][1]]["company_name"]
          similarity = float(best_jobs[i][0]) * 100
          job_data.append((title, name, similarity))
          
        
        return render_template('result.html', user_resume=res, job_data=job_data)


if __name__ == '__main__':
    app.run(debug=True)
