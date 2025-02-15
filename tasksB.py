import os
import requests
import sqlite3
import markdown
import subprocess
import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image

# Security Checks: Ensure access is only within /data/
def B12(filepath):
    if not filepath.startswith("/data"):
        raise PermissionError("Access outside /data is not allowed.")
    return True

# B3: Fetch Data from an API
def B3(url, save_path):
    B12(save_path)
    response = requests.get(url)
    with open(save_path, 'w') as file:
        file.write(response.text)

# B4: Clone a Git Repo and Make a Commit
def B4(repo_url, commit_message):
    repo_path = f"/data/{repo_url.split('/')[-1].replace('.git', '')}"
    subprocess.run(["git", "clone", repo_url, repo_path])
    subprocess.run(["git", "-C", repo_path, "commit", "-m", commit_message])
    return f"Repository cloned and committed in {repo_path}"

# B5: Run SQL Query
def B5(db_path, query, output_filename):
    B12(db_path)
    B12(output_filename)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    conn.close()
    with open(output_filename, 'w') as file:
        file.write(str(result))
    return result

# B6: Web Scraping
def B6(url, output_filename):
    B12(output_filename)
    result = requests.get(url).text
    with open(output_filename, 'w') as file:
        file.write(result)
    return "Web data saved."

# B7: Image Processing
def B7(image_path, output_path, resize=None):
    B12(image_path)
    B12(output_path)
    img = Image.open(image_path)
    if resize:
        img = img.resize(resize)
    img.save(output_path)
    return "Image processed and saved."

# B8: Audio Transcription
def B8(audio_path, output_path):
    B12(audio_path)
    B12(output_path)
    import openai
    with open(audio_path, 'rb') as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    with open(output_path, 'w') as file:
        file.write(transcript)
    return transcript

# B9: Markdown to HTML Conversion
def B9(md_path, output_path):
    B12(md_path)
    B12(output_path)
    with open(md_path, 'r') as file:
        html = markdown.markdown(file.read())
    with open(output_path, 'w') as file:
        file.write(html)
    return "Markdown converted to HTML."

# B10: API Endpoint for CSV Filtering
app = Flask(__name__)

@app.route('/filter_csv', methods=['POST'])
def filter_csv():
    try:
        data = request.json
        csv_path, filter_column, filter_value = data['csv_path'], data['filter_column'], data['filter_value']
        B12(csv_path)
        df = pd.read_csv(csv_path)
        if filter_column not in df.columns:
            return jsonify({"error": "Column not found"}), 400
        filtered = df[df[filter_column] == filter_value]
        return jsonify(filtered.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
