import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from PIL import Image
import tempfile
import google.generativeai as genai
import uuid
import shutil
import re

app = Flask(__name__)

# --- Dynamic CORS Configuration ---
# Reads the frontend URL from an environment variable for production.
frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
CORS(app, resources={r"/*": {"origins": frontend_url}}) # Allow all routes, not just /api

# --- Gemini API Key Configuration ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set!")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Directory Setup for Temporary Storage on Render ---
# These folders will be created on the server's temporary filesystem.
UPLOAD_FOLDER = 'uploads/temp_resumes'
SAVED_DATA_DIR = 'saved_data'
SAVED_RESUMES_DIR = os.path.join(SAVED_DATA_DIR, 'resumes')
METADATA_DB_FILE = os.path.join(SAVED_DATA_DIR, 'resumes_metadata.json')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAVED_RESUMES_DIR, exist_ok=True)

# Initialize metadata DB if it doesn't exist
if not os.path.exists(METADATA_DB_FILE):
    with open(METADATA_DB_FILE, 'w') as f:
        json.dump([], f)

# --- Helper functions for metadata ---
def load_metadata():
    if not os.path.exists(METADATA_DB_FILE) or os.stat(METADATA_DB_FILE).st_size == 0:
        return []
    with open(METADATA_DB_FILE, 'r') as f:
        return json.load(f)

def save_metadata(metadata):
    with open(METADATA_DB_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

# --- LLM Interaction Functions (No changes needed here) ---
def pdf_to_image(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    img_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_resume.png")
    pix.save(img_path)
    doc.close()
    return img_path

def parse_resume_content(pdf_file_path):
    image_path = None
    try:
        image_path = pdf_to_image(pdf_file_path)
        image = Image.open(image_path).convert("RGB")
        prompt = "You are an expert resume parser. Extract all key information from this resume image and format it as a single, valid JSON object. Include sections for personal info, education, skills, work experience, and projects."
        response = model.generate_content([prompt, image])
        raw_llm_output = response.text
        json_match = re.search(r'```json\n([\s\S]*?)\n```', raw_llm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = raw_llm_output
        parsed_json = json.loads(json_str)
        extracted_name = parsed_json.get("name", "Unknown Person")
        display_output = f"```json\n{json.dumps(parsed_json, indent=2)}\n```"
        return {
            "display_output": display_output,
            "raw_parsed_text": json.dumps(parsed_json),
            "extracted_name": extracted_name
        }
    except Exception as e:
        return {
            "display_output": f"```plain\nError during resume parsing: {e}\n```",
            "raw_parsed_text": json.dumps({"error": str(e)}),
            "extracted_name": "Error"
        }
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

def resume_check_content(resume_text):
    prompt = f"Review the following resume text and provide a comprehensive summary of red flags, areas for improvement, and overall quality. Be specific and constructive.\n\n{resume_text}"
    response = model.generate_content(prompt)
    return response.text

def jd_match_content(resume_text, jd_text):
    prompt = f"Compare the resume text with the job description. Return a detailed skill match table in Markdown format with columns: 'Skill', 'Mentioned in Resume', 'Required by JD', and 'Match Score (0-1)'.\n\nResume:\n{resume_text}\n\nJob Description:\n{jd_text}"
    response = model.generate_content(prompt)
    return response.text

def generate_questions_content(resume_text, jd_text):
    prompt = f"Based on the provided resume and job description, generate 5 technical, 5 behavioral, and 5 scenario-based interview questions. For each, provide a concise 'Best Answer' a strong candidate would give. Format as Markdown tables.\n\nResume:\n{resume_text}\n\nJob Description:\n{jd_text}"
    response = model.generate_content(prompt)
    return response.text

def fit_score_content(resume_text, jd_text):
    prompt = f"Analyze how well the resume fits the job description. The first line of your response MUST be 'Score: X.X/10'. Follow this with a detailed '### Justification:' covering skill alignment, experience relevance, and areas for improvement.\n\nResume:\n{resume_text}\n\nJob Description:\n{jd_text}"
    response = model.generate_content(prompt)
    return response.text

def convert_json_to_markdown_table_programmatic(json_string):
    try:
        data = json.loads(json_string)
        if "raw_text_fallback" in data:
            return f"Could not generate table. Raw text: {data['raw_text_fallback']}"
        table = "| Category | Details |\n|---|---|\n"
        for key, value in data.items():
            details = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
            table += f"| **{key.replace('_', ' ').title()}** | {details.replace('\n', '<br>')} |\n"
        return table
    except Exception as e:
        return f"Error generating table: {e}"

# --- JD Samples ---
JD_OPTIONS = {
    "Software Engineer": "We are seeking a skilled Software Engineer with strong problem-solving abilities and experience in data structures, algorithms, and object-oriented programming. Proficiency in Python, Java, or C++ is required. Experience with web frameworks like Django/Flask or Spring Boot, and database systems such as SQL or NoSQL is a plus. Candidates should be familiar with version control (Git) and agile development methodologies.",
    "Data Scientist": "Build ML models and extract insights from complex datasets. Requires strong statistical knowledge, proficiency in Python/R, and experience with libraries like Pandas, NumPy, Scikit-learn, and TensorFlow/PyTorch. Experience with data visualization tools (Matplotlib, Seaborn, Tableau) and big data technologies (Spark, Hadoop) is a plus. Strong communication skills for presenting findings are essential.",
    "Frontend Developer": "Looking for a frontend developer proficient in React.js, HTML5, CSS3, and JavaScript. Experience with state management libraries (Redux, Zustand) and modern build tools (Webpack, Vite) is essential. Familiarity with responsive design principles, cross-browser compatibility, and UI/UX best practices is highly valued. Knowledge of TypeScript and component libraries like Material-UI or Ant Design is a plus."
    # Add other JDs here if you want
}

# --- API Endpoints (Corrected without /api prefix) ---

@app.route('/jd_options', methods=['GET'])
def get_jd_options():
    return jsonify(list(JD_OPTIONS.keys()))

@app.route('/jd_default', methods=['GET'])
def get_jd_default():
    return jsonify(JD_OPTIONS["Software Engineer"])

@app.route('/jd_text', methods=['POST'])
def get_jd_text():
    data = request.get_json()
    role = data.get('role')
    if role == "Custom Input":
        return jsonify("")
    return jsonify(JD_OPTIONS.get(role, "No default JD found."))

@app.route('/parse_resume', methods=['POST'])
def api_parse_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        original_filename = secure_filename(file.filename)
        unique_temp_filename = f"{uuid.uuid4()}_{original_filename}"
        temp_filepath = os.path.join(UPLOAD_FOLDER, unique_temp_filename)
        file.save(temp_filepath)
        parsed_data = parse_resume_content(temp_filepath)
        return jsonify({
            "display_output": parsed_data["display_output"],
            "raw_parsed_text": parsed_data["raw_parsed_text"],
            "original_filename": original_filename,
            "temp_saved_filename": unique_temp_filename,
            "extracted_name": parsed_data["extracted_name"]
        })

@app.route('/resume_check', methods=['POST'])
def api_resume_check():
    data = request.get_json()
    result = resume_check_content(data.get('resume_text'))
    return jsonify({"output": result})

@app.route('/jd_match', methods=['POST'])
def api_jd_match():
    data = request.get_json()
    result = jd_match_content(data.get('resume_text'), data.get('jd_text'))
    return jsonify({"output": result})

@app.route('/generate_questions', methods=['POST'])
def api_generate_questions():
    data = request.get_json()
    result = generate_questions_content(data.get('resume_text'), data.get('jd_text'))
    return jsonify({"output": result})

@app.route('/fit_score', methods=['POST'])
def api_fit_score():
    data = request.get_json()
    result = fit_score_content(data.get('resume_text'), data.get('jd_text'))
    return jsonify({"output": result})

@app.route('/generate_resume_table', methods=['POST'])
def api_generate_resume_table():
    data = request.get_json()
    result = convert_json_to_markdown_table_programmatic(data.get('resume_text_cache'))
    return jsonify({"output": result})

@app.route('/confirm_document', methods=['POST'])
def confirm_document():
    data = request.json
    if not all(k in data for k in ['resume_text_cache', 'fit_score_output', 'interview_qa_output', 'selected_jd_role', 'original_file_name', 'temp_saved_filename', 'parsed_resume_name']):
        return jsonify({"error": "Missing required data"}), 400
    entry_id = str(uuid.uuid4())
    file_ext = os.path.splitext(data['original_file_name'])[1]
    saved_resume_filename = f"{entry_id}_{secure_filename(os.path.splitext(data['original_file_name'])[0])}{file_ext}"
    saved_qa_filename = f"{entry_id}_qa.md"
    temp_resume_path = os.path.join(UPLOAD_FOLDER, data['temp_saved_filename'])
    if os.path.exists(temp_resume_path):
        shutil.move(temp_resume_path, os.path.join(SAVED_RESUMES_DIR, saved_resume_filename))
    with open(os.path.join(SAVED_RESUMES_DIR, saved_qa_filename), 'w', encoding='utf-8') as f:
        f.write(data['interview_qa_output'])
    metadata = load_metadata()
    metadata.append({
        "id": entry_id,
        "person_name": data['parsed_resume_name'],
        "jd_role": data['selected_jd_role'],
        "fit_score": data['fit_score_output'],
        "resume_filename": saved_resume_filename,
        "qa_filename": saved_qa_filename,
        "timestamp": data.get('timestamp')
    })
    save_metadata(metadata)
    return jsonify({"message": "Document saved!", "id": entry_id}), 200

@app.route('/get_saved_resumes', methods=['GET'])
def get_saved_resumes():
    role_filter = request.args.get('role')
    sort_key = request.args.get('sort_key', 'timestamp')
    sort_order = request.args.get('sort_order', 'desc')
    metadata = load_metadata()
    if role_filter and role_filter != 'All Roles':
        filtered_metadata = [entry for entry in metadata if entry.get('jd_role') == role_filter]
    else:
        filtered_metadata = metadata
    
    def get_score(entry):
        try:
            return float(entry.get('fit_score', '0/10').split('/')[0].split(':')[1].strip())
        except:
            return 0.0
            
    reverse_sort = sort_order == 'desc'
    if sort_key == 'fit_score':
        filtered_metadata.sort(key=get_score, reverse=reverse_sort)
    elif sort_key == 'person_name':
        filtered_metadata.sort(key=lambda x: x.get('person_name', '').lower(), reverse=reverse_sort)
    else:
        filtered_metadata.sort(key=lambda x: x.get(sort_key, ''), reverse=reverse_sort)
        
    return jsonify(filtered_metadata)

@app.route('/download_resume/<filename>', methods=['GET'])
def download_resume(filename):
    return send_from_directory(SAVED_RESUMES_DIR, secure_filename(filename), as_attachment=True)

@app.route('/get_interview_qa/<filename>', methods=['GET'])
def get_interview_qa(filename):
    file_path = os.path.join(SAVED_RESUMES_DIR, secure_filename(filename))
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return jsonify({"qa_content": f.read()})
    return jsonify({"error": "File not found"}), 404

@app.route('/clear_all_data', methods=['POST'])
def clear_all_data():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    if os.path.exists(SAVED_DATA_DIR):
        shutil.rmtree(SAVED_DATA_DIR)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(SAVED_RESUMES_DIR, exist_ok=True)
    with open(METADATA_DB_FILE, 'w') as f:
        json.dump([], f)
    return jsonify({"message": "All data cleared successfully!"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)