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
# Falls back to localhost for local development.
frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
CORS(app, resources={r"/api/*": {"origins": frontend_url}})

# --- Gemini API Key Configuration ---
# It's good practice to ensure the key is set.
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set!")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash") # gemini-1.5-flash is a valid and recent model name

# --- Directory Setup for Persistent Storage on Render ---
# Use a Render Disk path from an environment variable.
# Default to the current directory for local development.
render_disk_path = os.getenv('RENDER_DISK_PATH', '.') 

# Define paths relative to the disk mount point
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
    """Loads resume metadata from the JSON database file."""
    if not os.path.exists(METADATA_DB_FILE) or os.stat(METADATA_DB_FILE).st_size == 0:
        return []
    with open(METADATA_DB_FILE, 'r') as f:
        return json.load(f)

def save_metadata(metadata):
    """Saves resume metadata to the JSON database file."""
    with open(METADATA_DB_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

# --- LLM Interaction Functions ---
# These functions interact with the Gemini model.
# The prompts are designed to get structured or specific outputs.

def pdf_to_image(pdf_file_path):
    """Converts the first page of a PDF to an image and returns the image path."""
    doc = fitz.open(pdf_file_path)
    first_page = doc[0]
    pix = first_page.get_pixmap(dpi=300)
    # Use tempfile to ensure unique and safe temporary image path
    img_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_resume_img.png")
    pix.save(img_path)
    doc.close()
    return img_path

def parse_resume_content(pdf_file_path):
    """
    Parses a resume PDF using the Gemini Vision model.
    Accepts the path to the PDF file.
    Returns parsed JSON string, raw text, and an extracted name.
    """
    image_path = None
    try:
        # Convert the first page of the PDF to an image
        image_path = pdf_to_image(pdf_file_path)
        image = Image.open(image_path).convert("RGB")

        # Prompt for resume parsing - requesting structured JSON output
        prompt = """
        You are an AI resume parser. Extract the following information from the resume:
        1.  **Name**: The full name of the candidate.
        2.  **Email**: The candidate's email address.
        3.  **Phone Number**: The candidate's phone number.
        4.  **Education**: A list of educational entries. For each, include:
            * Degree (e.g., "Bachelor of Science in Computer Engineering")
            * Institution (e.g., "University of California, Berkeley")
            * Years (e.g., "2018-2022")
            * Location (e.g., "Berkeley, CA")
        5.  **Skills**: A list of key technical and soft skills. Categorize if possible (e.g., "Programming Languages", "Frameworks", "Tools").
        6.  **Work Experience**: A list of work experience entries. For each, include:
            * Title (e.g., "Software Engineer")
            * Company (e.g., "Google")
            * Dates (e.g., "Jan 2022 - Present")
            * Responsibilities (a list of key responsibilities and achievements, use bullet points).
        7.  **Projects**: A list of significant projects. For each, include:
            * Name (e.g., "E-commerce Platform")
            * Technologies (a list of technologies used).
            * Outcomes (a list of key outcomes or features).

        Format your entire response as a single, valid JSON object.
        If a section is not found or is empty, use an empty string for single values or an empty list for arrays.
        Example JSON structure:
        {
          "name": "John Doe",
          "email": "john.doe@example.com",
          "phone": "+1234567890",
          "education": [
            {
              "degree": "B.S. Computer Science",
              "institution": "University A",
              "years": "2018-2022",
              "location": "City A"
            }
          ],
          "skills": {
            "Programming Languages": ["Python", "Java"],
            "Frameworks": ["React", "Spring Boot"]
          },
          "experience": [
            {
              "title": "Software Engineer",
              "company": "Tech Corp",
              "dates": "Jan 2023 - Present",
              "responsibilities": ["Developed scalable APIs", "Optimized database queries"]
            }
          ],
          "projects": [
            {
              "name": "Portfolio Website",
              "technologies": ["React", "Node.js"],
              "outcomes": ["Showcased projects", "Improved personal branding"]
            }
          ]
        }
        """
        response = model.generate_content([prompt, image])
        raw_llm_output = response.text

        # Attempt to parse the LLM's response as JSON
        parsed_json = {}
        extracted_name = "Unknown Person"
        try:
            # The LLM might wrap JSON in markdown code blocks, extract it
            json_match = re.search(r'```json\n([\s\S]*?)\n```', raw_llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = raw_llm_output # Assume it's just JSON if no markdown block

            parsed_json = json.loads(json_str)
            extracted_name = parsed_json.get("name", "Unknown Person")
            
            # Return the pretty-printed JSON as display_output
            display_output = f"```json\n{json.dumps(parsed_json, indent=2)}\n```"
            
        except json.JSONDecodeError as e:
            # If LLM didn't return valid JSON, return raw text and an error message
            display_output = f"```plain\nError parsing LLM JSON output: {e}\nRaw LLM Output:\n{raw_llm_output}\n```"
            parsed_json = {"raw_text_fallback": raw_llm_output} # Store raw text for other functions
            extracted_name = "Unknown Person (Parsing Error)"
        
        return {
            "display_output": display_output,
            "raw_parsed_text": json.dumps(parsed_json), # Store the parsed JSON (or raw text fallback) as raw_parsed_text
            "extracted_name": extracted_name
        }
    except Exception as e:
        return {
            "display_output": f"```plain\nError during resume parsing process: {e}\n```",
            "raw_parsed_text": json.dumps({"error": str(e)}),
            "extracted_name": "Error"
        }
    finally:
        # Clean up temporary image file created from PDF
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

def resume_check_content(resume_text):
    """Performs a smart check on the resume text for common issues."""
    if not resume_text:
        return "Please parse a resume first."
    prompt = """
    Review the following resume text for:
    - **Fake certifications:** Point out any certifications that seem suspicious or unverified.
    - **Outdated technologies:** Mention any technologies listed that are no longer current or widely used in the industry for the roles typically associated with this resume.
    - **Grammar and spelling issues:** Point out specific examples of grammatical errors, typos, or awkward phrasing.
    - **Missing project descriptions:** If projects are listed, but lack details on technologies used, outcomes, or your specific contributions, highlight this.
    - **General readability and conciseness:** Provide feedback on whether the resume is easy to read, well-organized, and free from unnecessary jargon or excessive detail.
    - **Quantifiable achievements:** Suggest areas where achievements could be quantified with numbers, percentages, or metrics.

    Return a comprehensive summary of red flags or areas to improve. Be specific, constructive, and provide actionable advice.
    """
    response = model.generate_content([prompt, resume_text])
    return response.text

def jd_match_content(resume_text, jd_text):
    """Compares resume skills with job description requirements and generates a match table."""
    if not resume_text or not jd_text:
        return "Please parse a resume and provide a job description."
    prompt = f"""
    Compare the following resume text with the job description.
    Resume:
    {resume_text}

    Job Description:
    {jd_text}

    Return a skill match table in Markdown format. The table should have exactly 4 columns:
    "Skill", "Mentioned in Resume", "Required by JD", "Match Score (0-1)".
    - "Skill": Identify at least 10-15 relevant key skills from both the resume and JD. Prioritize skills explicitly mentioned in the JD.
    - "Mentioned in Resume": Indicate 'Yes' or 'No' if the skill is clearly present or implied in the resume.
    - "Required by JD": Indicate 'Yes' or 'No' if the skill is explicitly or implicitly required by the JD.
    - "Match Score (0-1)": Provide a numerical score from 0 to 1 (e.g., 0.8, 0.5, 0.2) indicating the strength of the match for that specific skill. A score of 1 means a perfect match, 0 means no match.
    Ensure the output is a valid Markdown table.
    """
    response = model.generate_content([prompt, resume_text, jd_text])
    return response.text

def generate_questions_content(resume_text, jd_text):
    """Generates interview questions and best answers based on resume and JD."""
    if not resume_text or not jd_text:
        return "Please parse a resume and provide a job description."
    prompt = f"""
    Based on the provided resume and job description, generate:
    - 5 Technical Interview Questions
    - 5 Behavioral Questions
    - 5 Scenario-Based Questions

    For each question, also provide a concise "Best Answer" that a strong candidate (like the one described in the resume) would give, highlighting relevant skills or experiences from the resume if applicable.
    Format the output using Markdown tables. Each section (Technical, Behavioral, Scenario) should have its own table.
    Each table should have two columns: "Question" and "Best Answer".

    Example format:
    ## Technical Interview Questions
    | Question | Best Answer |
    |---|---|
    | What is a RESTful API? | A RESTful API (Representational State Transfer) is an architectural style for networked applications. It uses standard HTTP methods (GET, POST, PUT, DELETE) to interact with resources. Key principles include statelessness, client-server separation, cacheability, and a uniform interface. |
    | Explain the difference between SQL and NoSQL databases. | SQL databases are relational, using structured query language and predefined schemas, suitable for complex queries and ACID compliance. NoSQL databases are non-relational, schema-less, and offer flexibility and scalability for large, unstructured data, often used for big data and real-time web apps. |

    ## Behavioral Interview Questions
    | Question | Best Answer |
    |---|---|
    | Tell me about a time you faced a challenging problem and how you overcame it. | In my previous role, we encountered a critical bug in production that was causing data corruption. I immediately initiated a debugging session, collaborating with the team to isolate the issue. After identifying a race condition, I proposed a synchronization mechanism, implemented it, and thoroughly tested the fix, resolving the problem within hours and preventing further data loss. |

    ## Scenario-Based Interview Questions
    | Question | Best Answer |
    |---|---|
    | Imagine a user reports a critical bug in your deployed application. Walk me through your steps to diagnose and resolve it. | First, I'd gather details from the user (reproduction steps, error messages). Then, I'd check logs and monitoring tools for anomalies. I'd try to reproduce the bug in a development environment. Once reproduced, I'd use debugging tools to pinpoint the root cause. After fixing, I'd write unit/integration tests, deploy to a staging environment for validation, and finally push to production, communicating updates to the user throughout. |
    """
    response = model.generate_content([prompt, resume_text, jd_text])
    return response.text

def fit_score_content(resume_text, jd_text):
    """Analyzes how well the resume fits the job description and returns a score."""
    if not resume_text or not jd_text:
        return "Please parse a resume and provide a job description."
    prompt = f"""
    Analyze how well the following resume text fits the job description.
    Resume:
    {resume_text}

    Job Description:
    {jd_text}

    Provide a "Fit Score" out of 10 (e.g., 8.5/10, 6.0/10, 9/10).
    Then, provide a detailed "Justification" in markdown, covering:
    - **Skill Alignment:** How well do the resume's skills match the JD's requirements? Be specific about matches and gaps.
    - **Experience Relevance:** How relevant is the candidate's work experience to the JD? Highlight specific roles/projects.
    - **Overall Suitability:** A general assessment of how strong a candidate they are for this specific role based on all information.
    - **Areas for Improvement:** Suggest 2-3 concrete ways the resume could be improved for this JD.

    Format your response clearly. The first line MUST be "Score: X.X/10" followed by a newline, then "### Justification:", then the detailed justification.
    Example:
    Score: 7.5/10

    ### Justification:
    * **Skill Alignment:** The resume shows good alignment...
    * **Experience Relevance:** This is the weakest point...
    * **Overall Suitability:** While the candidate has the foundational programming language...
    * **Areas for Improvement:**
        * Quantify achievements in experience.
        * Add a summary tailored to the JD.
    """
    response = model.generate_content([prompt, resume_text, jd_text])
    return response.text

def convert_json_to_markdown_table_programmatic(json_string):
    """
    Parses the JSON string (expected from parse_resume_content) and programmatically
    generates a Markdown table, using HTML line breaks for multi-line content.
    """
    if not json_string:
        return "No resume data to display in table. Please parse a resume first."

    try:
        parsed_data = json.loads(json_string)
        # If it's a fallback raw text, try to use LLM to convert
        if isinstance(parsed_data, dict) and "raw_text_fallback" in parsed_data:
            return generate_table_from_raw_text(parsed_data["raw_text_fallback"])
        if not isinstance(parsed_data, dict): # If it's not a dict, it's unexpected
            return generate_table_from_raw_text(json_string) # Treat as raw text
    except json.JSONDecodeError:
        # If direct JSON parsing fails, try to send the raw string to the LLM for table generation
        return generate_table_from_raw_text(json_string)
    except Exception as e:
        return f"An unexpected error occurred during table generation setup: {e}"

    table_lines = ["| Category | Details |", "|---|---|"]

    # Define the desired order of categories
    categories_order = ["name", "email", "phone", "education", "skills", "experience", "projects"]

    for category in categories_order:
        details = parsed_data.get(category, "N/A")

        formatted_details = "N/A" # Default

        if category == "education" and isinstance(details, list) and details:
            edu_lines = []
            for edu_item in details:
                edu_str = ""
                if isinstance(edu_item, dict):
                    parts = []
                    if edu_item.get('degree'): parts.append(f"**{edu_item['degree']}**")
                    if edu_item.get('institution'): parts.append(edu_item['institution'])
                    if edu_item.get('years'): parts.append(f"({edu_item['years']})")
                    if edu_item.get('location'): parts.append(edu_item['location'])
                    edu_str = ", ".join(parts)
                else:
                    edu_str = str(edu_item)
                if edu_str:
                    edu_lines.append(f"- {edu_str}")
            formatted_details = "<br>".join(edu_lines) if edu_lines else "N/A"

        elif category == "skills" and isinstance(details, dict):
            skill_lines = []
            for skill_type, skill_list in details.items():
                if isinstance(skill_list, list) and skill_list:
                    skill_lines.append(f"**{skill_type}:** {', '.join(skill_list)}")
            formatted_details = "<br>".join(skill_lines) if skill_lines else "N/A"
        elif category == "skills" and isinstance(details, list): # Fallback if skills are just a list
            formatted_details = "<br>".join([f"- {skill}" for skill in details]) if details else "N/A"


        elif category in ["experience", "projects"] and isinstance(details, list):
            item_details = []
            for item in details:
                if isinstance(item, dict):
                    if category == "experience":
                        if 'title' in item and 'company' in item and 'dates' in item:
                            item_details.append(f"- **{item['title']}**, {item['company']} ({item['dates']}):")
                            if 'responsibilities' in item and isinstance(item['responsibilities'], list):
                                for resp in item['responsibilities']:
                                    item_details.append(f"  - {resp}")
                    elif category == "projects":
                        proj_line = f"- **{item.get('name', 'Unnamed Project')}**"
                        tech = item.get('technologies')
                        outcomes = item.get('outcomes')
                        if tech and isinstance(tech, list):
                            proj_line += f" (Technologies: {', '.join(tech)})"
                        item_details.append(f"{proj_line}:")
                        if outcomes and isinstance(outcomes, list):
                            for outcome in outcomes:
                                item_details.append(f"  - {outcome}")
                    else: # Fallback for unexpected dict format
                        item_details.append(f"- {json.dumps(item)}")
                else: # Fallback for non-dict items in list
                    item_details.append(f"- {item}")
            formatted_details = "<br>".join(item_details) if item_details else "N/A"

        else:
            formatted_details = str(details)
            if formatted_details.strip() == "" or formatted_details.strip().lower() == "n/a":
                formatted_details = "N/A"

        table_lines.append(f"| **{category.replace('_', ' ').title()}** | {formatted_details} |")

    return "\n".join(table_lines)


def generate_table_from_raw_text(raw_text):
    """
    Fallback function: If direct JSON parsing fails, send the entire raw text
    from the initial resume parsing to the LLM and ask it to format it into a table.
    This leverages the LLM's understanding to salvage messy output.
    """
    if not raw_text:
        return "Could not generate table. Raw resume output was empty."

    prompt = f"""
    I have a raw text output that attempts to parse a resume. This text might contain JSON,
    or it might be a mix of text and incomplete JSON. Your task is to extract the following
    categories and present them in a Markdown table with two columns: "Category" and "Details".

    The categories you must look for are:
    - Name
    - Email
    - Phone
    - Education
    - Skills
    - Work Experience
    - Projects

    **Formatting Rules for 'Details' column:**
    - Use HTML line breaks (`<br>`) for all new lines within a table cell.
    - For lists of items (like Skills, Education summaries, or individual responsibilities/outcomes within Experience/Projects), use Markdown bullet points (`- `) followed by `<br>`.
    - For multi-line entries within Experience or Projects, ensure clear indentation for sub-bullets (e.g., `  - ` followed by `<br>`).
    - If a category is not found or is empty, state "N/A".
    - Do NOT truncate any information. Include all details found for each category.

    Here is the raw resume output text:
    ---
    {raw_text}
    ---

    Please generate only the Markdown table.
    """
    try:
        response = model.generate_content([prompt, raw_text])
        return response.text
    except Exception as e:
        return f"Error using LLM to generate table from raw text: {e}"

# --- JD Samples (Full List Restored) ---
JD_OPTIONS = {
    "Software Engineer": "We are seeking a skilled Software Engineer with strong problem-solving abilities and experience in data structures, algorithms, and object-oriented programming. Proficiency in Python, Java, or C++ is required. Experience with web frameworks like Django/Flask or Spring Boot, and database systems such as SQL or NoSQL is a plus. Candidates should be familiar with version control (Git) and agile development methodologies.",
    "Frontend Developer": "Looking for a frontend developer proficient in React.js, HTML5, CSS3, and JavaScript. Experience with state management libraries (Redux, Zustand) and modern build tools (Webpack, Vite) is essential. Familiarity with responsive design principles, cross-browser compatibility, and UI/UX best practices is highly valued. Knowledge of TypeScript and component libraries like Material-UI or Ant Design is a plus.",
    "Backend Developer": "Experience with Node.js, Python, and RESTful APIs. Solid understanding of database design (SQL/NoSQL), authentication/authorization mechanisms, and cloud platforms (AWS, Azure, GCP). Familiarity with microservices architecture, message queues (Kafka, RabbitMQ), and containerization (Docker, Kubernetes) is preferred. Strong debugging and performance optimization skills are required.",
    "Data Scientist": "Build ML models and extract insights from complex datasets. Requires strong statistical knowledge, proficiency in Python/R, and experience with libraries like Pandas, NumPy, Scikit-learn, and TensorFlow/PyTorch. Experience with data visualization tools (Matplotlib, Seaborn, Tableau) and big data technologies (Spark, Hadoop) is a plus. Strong communication skills for presenting findings are essential.",
    "Machine Learning Engineer": "Productionize models using TensorFlow/PyTorch. Design, develop, and deploy scalable ML systems. Strong programming skills in Python, experience with MLOps practices, and cloud platforms (AWS Sagemaker, GCP AI Platform). Knowledge of model optimization, deployment strategies, and monitoring tools is crucial. Familiarity with distributed training and data pipelines is a plus.",
    "DevOps Engineer": "Handle CI/CD pipelines, Docker, Kubernetes, and cloud infrastructure (AWS, Azure, GCP). Experience with automation tools (Ansible, Terraform), monitoring systems (Prometheus, Grafana), and scripting (Bash, Python). Strong understanding of network protocols, security best practices, and system administration. Ability to troubleshoot complex production issues is key.",
    "Cybersecurity Analyst": "Monitor threats, configure firewalls, ensure security policies. Experience with SIEM tools, vulnerability assessments, penetration testing, and incident response. Knowledge of network security, application security, and data privacy regulations (GDPR, HIPAA). Certifications like CompTIA Security+, CEH, or CISSP are highly desirable.",
    "UI/UX Designer": "Skilled in Figma, Adobe XD, and user-first design principles. Create wireframes, prototypes, user flows, and high-fidelity mockups. Strong understanding of usability, accessibility, and responsive design. Experience with user research, A/B testing, and design systems. A portfolio demonstrating strong visual design and problem-solving skills is required.",
    "Cloud Architect": "Design scalable systems on AWS/Azure/GCP. Expertise in cloud services (compute, storage, networking, databases), migration strategies, and cost optimization. Strong understanding of security best practices in the cloud, disaster recovery, and high availability. Certifications (AWS Certified Solutions Architect, Azure Solutions Architect Expert) are a significant advantage.",
    "Mobile App Developer": "Flutter or React Native with iOS/Android deployment. Strong proficiency in Dart/JavaScript/TypeScript. Experience with mobile UI/UX best practices, API integration, and push notifications. Familiarity with mobile testing frameworks and app store deployment processes. Knowledge of native platform development (Swift/Kotlin) is a plus.",
    "AI Researcher": "Work on NLP, deep learning, generative models. Strong theoretical background in AI/ML, mathematics, and statistics. Proficiency in Python and deep learning frameworks (TensorFlow, PyTorch). Experience with research publications, experimental design, and large-scale data analysis. PhD or equivalent research experience in a relevant field is often required.",
    "Full Stack Developer": "MERN or MEAN stack experience required. Develop both frontend (React/Angular/Vue) and backend (Node.js/Express) components. Strong understanding of database interactions (MongoDB/SQL), RESTful APIs, and deployment processes. Familiarity with cloud platforms and version control. Ability to work across the entire software development lifecycle.",
    "System Administrator": "Manage infrastructure, troubleshoot, maintain servers (Linux/Windows). Experience with virtualization (VMware, Hyper-V), networking, and scripting (Bash, PowerShell). Knowledge of monitoring tools, backup solutions, and security patches. Ability to diagnose and combat security threats.",
    "Data Analyst": "Use SQL, Python, dashboards, and Excel to analyze data and provide actionable insights. Experience with data cleaning, transformation, and visualization. Familiarity with business intelligence tools (Tableau, Power BI) and statistical analysis. Strong communication skills to present findings to non-technical stakeholders.",
    "Blockchain Developer": "Work with Solidity, Ethereum, and smart contracts. Experience with decentralized applications (dApps), Web3.js/Ethers.js, and blockchain development frameworks (Truffle, Hardhat). Understanding of cryptographic principles, consensus mechanisms, and token standards (ERC-20, ERC-721). Familiarity with layer 2 solutions and defi concepts is a plus.",
    "QA Engineer": "Manual & automated testing with Selenium/Cypress. Design and execute test plans, write test cases, and report bugs. Experience with test management tools (Jira, TestRail) and CI/CD integration. Strong attention to detail and ability to identify edge cases. Familiarity with performance and security testing is a plus.",
    "Product Manager": "Coordinate engineering/design, write specs, define roadmap. Strong understanding of market research, user needs, and product lifecycle. Experience with agile methodologies, backlog prioritization, and stakeholder management. Excellent communication and leadership skills to drive product success.",
    "Technical Writer": "Create clear dev and user documentation. Translate complex technical concepts into easy-to-understand language. Experience with documentation tools (Markdown, Sphinx, Confluence) and version control. Strong research skills and attention to detail. Ability to collaborate with engineers and product teams.",
    "Game Developer": "Unity or Unreal Engine, prototype & build games. Strong programming skills in C#/C++. Experience with game design principles, physics engines, and graphics rendering. Familiarity with game development pipelines, asset management, and performance optimization. Ability to work in a team and contribute to all stages of game development.",
    "Network Engineer": "Design, implement, and maintain network infrastructure. Expertise in routing protocols (BGP, OSPF), switching, and firewalls. Experience with network monitoring tools, troubleshooting, and security best practices. Certifications like CCNA, CCNP, or JNCIE are highly desirable. Strong understanding of TCP/IP and network security principles."
}

# --- API Endpoints ---

@app.route('/api/jd_options', methods=['GET'])
def get_jd_options():
    """Returns a list of available job description roles."""
    return jsonify(list(JD_OPTIONS.keys()))

@app.route('/api/jd_default', methods=['GET'])
def get_jd_default():
    """Returns the default job description text."""
    return jsonify(JD_OPTIONS["Software Engineer"])

@app.route('/api/jd_text', methods=['POST'])
def get_jd_text():
    """Returns the job description text for a given role."""
    data = request.get_json()
    role = data.get('role')
    if role == "Custom Input":
        return jsonify("") # Return empty string for custom input
    return jsonify(JD_OPTIONS.get(role, "No default JD found for this role."))

@app.route('/api/parse_resume', methods=['POST'])
def api_parse_resume():
    """
    Parses an uploaded resume PDF.
    Saves the PDF temporarily and returns its parsed text and a temporary filename.
    """
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Generate a unique temporary filename to avoid overwrites and issues with special chars
        original_filename = secure_filename(file.filename)
        unique_temp_filename = f"{uuid.uuid4()}_{original_filename}"
        temp_filepath = os.path.join(UPLOAD_FOLDER, unique_temp_filename)
        
        file.save(temp_filepath) # Save the uploaded file to the temporary directory

        try:
            # Pass the path to the saved PDF to parse_resume_content
            parsed_data = parse_resume_content(temp_filepath)
            
            return jsonify({
                "display_output": parsed_data["display_output"],
                "raw_parsed_text": parsed_data["raw_parsed_text"],
                "original_filename": original_filename, # Original name for frontend display
                "temp_saved_filename": unique_temp_filename, # Unique temp filename for later reference
                "extracted_name": parsed_data["extracted_name"] # Pass extracted name
            })
        except Exception as e:
            # If parsing fails, clean up the temporary PDF file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return jsonify({
                "error": f"Error processing resume: {e}",
                "display_output": f"```plain\\nError: {e}\\n\`\`\`",
                "raw_parsed_text": json.dumps({"error": str(e)}), # Ensure raw_parsed_text is always valid JSON
                "extracted_name": "Error"
            }), 500

@app.route('/api/resume_check', methods=['POST'])
def api_resume_check():
    """Performs a resume check and returns feedback."""
    data = request.get_json()
    resume_text = data.get('resume_text')
    result = resume_check_content(resume_text)
    return jsonify({"output": result})

@app.route('/api/jd_match', methods=['POST'])
def api_jd_match():
    """Calculates JD match and returns analysis."""
    data = request.get_json()
    resume_text = data.get('resume_text')
    jd_text = data.get('jd_text')
    result = jd_match_content(resume_text, jd_text)
    return jsonify({"output": result})

@app.route('/api/generate_questions', methods=['POST'])
def api_generate_questions():
    """Generates interview questions based on resume and JD."""
    data = request.get_json()
    resume_text = data.get('resume_text')
    jd_text = data.get('jd_text')
    result = generate_questions_content(resume_text, jd_text)
    return jsonify({"output": result})

@app.route('/api/fit_score', methods=['POST'])
def api_fit_score():
    """Calculates a fit score and returns justification."""
    data = request.get_json()
    resume_text = data.get('resume_text')
    jd_text = data.get('jd_text')
    result = fit_score_content(resume_text, jd_text)
    return jsonify({"output": result})

@app.route('/api/generate_resume_table', methods=['POST'])
def api_generate_resume_table():
    """Generates a structured resume table."""
    data = request.get_json()
    resume_text_cache = data.get('resume_text_cache')
    result = convert_json_to_markdown_table_programmatic(resume_text_cache)
    return jsonify({"output": result})

# --- NEW ENDPOINTS FOR SAVING/RETRIEVING DOCUMENTS ---

@app.route('/api/confirm_document', methods=['POST'])
def confirm_document():
    """
    Confirms and permanently saves a processed resume and its associated data.
    Moves the temporary PDF file to permanent storage.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    resume_text_cache = data.get('resume_text_cache')
    jd_text = data.get('jd_text')
    fit_score_output = data.get('fit_score_output')
    interview_qa_output = data.get('interview_qa_output')
    selected_jd_role = data.get('selected_jd_role')
    original_file_name = data.get('original_file_name') # Original name from user's computer
    temp_saved_filename = data.get('temp_saved_filename') # Unique temp name from parse_resume
    parsed_resume_name = data.get('parsed_resume_name') # Extracted name

    # Validate all required data is present
    if not all([resume_text_cache, jd_text, fit_score_output, interview_qa_output,
                selected_jd_role, original_file_name, temp_saved_filename, parsed_resume_name]):
        return jsonify({"error": "Missing required data for confirmation"}), 400

    # Generate a unique ID for this saved entry
    entry_id = str(uuid.uuid4())
    
    # Define paths for saving files
    file_extension = os.path.splitext(original_file_name)[1]
    saved_resume_filename_unique = f"{entry_id}_{secure_filename(os.path.splitext(original_file_name)[0])}{file_extension}"
    saved_qa_filename = f"{entry_id}_qa.md" # QA is always a .md file

    saved_resume_path = os.path.join(SAVED_RESUMES_DIR, saved_resume_filename_unique)
    saved_qa_path = os.path.join(SAVED_RESUMES_DIR, saved_qa_filename)

    # Path to the temporary uploaded file
    temp_resume_source_path = os.path.join(UPLOAD_FOLDER, temp_saved_filename)
    
    # Move the temporary resume file to the permanent saved resumes directory
    if os.path.exists(temp_resume_source_path):
        try:
            shutil.move(temp_resume_source_path, saved_resume_path) # Use shutil.move to move (cut-paste)
        except Exception as e:
            print(f"Error moving resume file from temp to saved: {e}")
            return jsonify({"error": f"Failed to save resume file permanently: {e}"}), 500
    else:
        # This is a critical error: the temporary file was not found.
        print(f"Error: Temporary resume file not found at {temp_resume_source_path}. Cannot save PDF.")
        return jsonify({"error": "Temporary resume file not found on server. Please re-upload and try again."}), 500

    # Save interview Q&A as a markdown file
    try:
        with open(saved_qa_path, 'w', encoding='utf-8') as f:
            f.write(interview_qa_output)
    except Exception as e:
        print(f"Error saving QA file: {e}")
        saved_qa_filename = None # Don't link to a non-existent QA file

    # Load existing metadata, add new entry, and save
    metadata = load_metadata()
    metadata.append({
        "id": entry_id,
        "person_name": parsed_resume_name,
        "jd_role": selected_jd_role,
        "fit_score": fit_score_output,
        "resume_filename": saved_resume_filename_unique,
        "qa_filename": saved_qa_filename,
        "timestamp": data.get('timestamp') # Timestamp from frontend
    })
    save_metadata(metadata)

    return jsonify({"message": "Document confirmed and saved!", "id": entry_id}), 200

@app.route('/api/get_saved_resumes', methods=['GET'])
def get_saved_resumes():
    """
    Retrieves a list of saved resume metadata, optionally filtered and sorted.
    """
    role_filter = request.args.get('role')
    sort_key = request.args.get('sort_key', 'timestamp') # Default sort by timestamp
    sort_order = request.args.get('sort_order', 'desc') # Default sort order descending

    metadata = load_metadata()
    
    if role_filter and role_filter != 'All Roles' and role_filter in JD_OPTIONS:
        filtered_metadata = [entry for entry in metadata if entry['jd_role'] == role_filter]
    else:
        filtered_metadata = metadata
        
    # Apply sorting based on sort_key and sort_order
    if sort_key:
        if sort_key == 'fit_score':
            # Need a custom sort for fit_score because it's a string like "X.X/10"
            # Extract numerical value for sorting. Handle cases where score might be missing or malformed.
            def get_score_value(entry):
                score_str = entry.get('fit_score', '0/10').split('/')[0]
                try:
                    return float(score_str)
                except ValueError:
                    return 0.0 # Default to 0 if score cannot be parsed
            filtered_metadata.sort(key=get_score_value, reverse=(sort_order == 'desc'))
        elif sort_key == 'person_name':
            filtered_metadata.sort(key=lambda x: x.get('person_name', '').lower(), reverse=(sort_order == 'desc'))
        else: # Default to timestamp or any other string key
            filtered_metadata.sort(key=lambda x: x.get(sort_key, ''), reverse=(sort_order == 'desc'))
        
    return jsonify(filtered_metadata), 200

@app.route('/api/download_resume/<filename>', methods=['GET'])
def download_resume(filename):
    """Serves a saved resume PDF file for download."""
    safe_filename = secure_filename(filename)
    if os.path.exists(os.path.join(SAVED_RESUMES_DIR, safe_filename)):
        return send_from_directory(SAVED_RESUMES_DIR, safe_filename, as_attachment=True)
    return jsonify({"error": "Resume file not found"}), 404

@app.route('/api/get_interview_qa/<filename>', methods=['GET'])
def get_interview_qa(filename):
    """Retrieves the content of a saved interview Q&A markdown file."""
    safe_filename = secure_filename(filename)
    file_path = os.path.join(SAVED_RESUMES_DIR, safe_filename)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            qa_content = f.read()
        return jsonify({"qa_content": qa_content}), 200
    return jsonify({"error": "QA file not found"}), 404

@app.route('/api/clear_all_data', methods=['POST'])
def clear_all_data():
    """
    Clears all temporary files, all saved resumes (PDFs and Q&A markdown files),
    and resets the metadata database. This is a destructive operation.
    """
    try:
        # Clear temporary uploads
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Clear saved resumes and Q&A files
        for filename in os.listdir(SAVED_RESUMES_DIR):
            file_path = os.path.join(SAVED_RESUMES_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Reset metadata database
        with open(METADATA_DB_FILE, 'w') as f:
            json.dump([], f)
            
        return jsonify({"message": "All data cleared successfully!"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to clear all data: {e}"}), 500

if __name__ == '__main__':
    # Clean up any old temporary files on startup for development convenience
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up old temp file {file_path}: {e}")
    app.run(debug=True, port=5000)
