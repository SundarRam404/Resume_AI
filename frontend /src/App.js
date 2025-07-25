import React, { useState, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

import LoaderGif from './load.gif'; // Make sure this path is correct

// Using explicit URL for API calls
// Use environment variable for the API URL, with a fallback for local development
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000/api'; 

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jdOptions, setJdOptions] = useState([]); // Stores JD roles from backend
  const [selectedJDRole, setSelectedJDRole] = useState('Software Engineer'); // Currently selected JD role
  const [jdText, setJdText] = useState(''); // Text content of the JD
  
  // CRITICAL FIX: Ensure these are correctly declared with useState
  const [resumeTextCache, setResumeTextCache] = useState(''); // Raw text from parsed resume
  
  const [parsedResumeOutput, setParsedResumeOutput] = useState(''); // Formatted parsed resume output (markdown)
  const [resumeTableOutput, setResumeTableOutput] = useState('');
  const [resumeCheckOutput, setResumeCheckOutput] = useState('');
  const [jdMatchOutput, setJdMatchOutput] = useState('');
  const [interviewQaOutput, setInterviewQaOutput] = useState('');
  const [fitScoreOutput, setFitScoreOutput] = useState('');
  const [activeTab, setActiveTab] = useState('parse'); // Controls which tab content is visible
  const [loading, setLoading] = useState(false); // Global loading indicator
  const [error, setError] = useState(null); // Global error message
  const [showFullFitScoreReview, setShowFullFitScoreReview] = useState(false); // For Fit Score dropdown

  // NEW STATES FOR SAVED RESUMES FEATURE
  const [savedResumes, setSavedResumes] = useState([]); // List of saved resume metadata
  const [showInterviewQaId, setShowInterviewQaId] = useState(null); // Tracks which QA dropdown is open
  const [savedJdFilter, setSavedJdFilter] = useState('All Roles'); // Filter for the "All Resumes" section
  const [extractedResumeName, setExtractedResumeName] = useState(''); // Name extracted from the current resume
  const [tempSavedFilename, setTempSavedFilename] = useState(''); // Temporary filename from backend after parse

  // NEW STATES FOR SORTING SAVED RESUMES
  const [sortKey, setSortKey] = useState('timestamp'); // 'timestamp', 'person_name', 'fit_score'
  const [sortOrder, setSortOrder] = useState('desc'); // 'asc', 'desc'

  // Function to extract numerical score (e.g., "90%" or "7/10")
  const getDisplayScore = (markdown) => {
    if (!markdown) return null;

    // Try to extract score from lines like "Score: 5.0/10" or "Overall Score: 75%"
    let scoreMatch = markdown.match(/Score:\s*(\d+(\.\d+)?)\s*(\/\s*\d+)?(\s*%?)?/i);
    if (scoreMatch) {
      const value = scoreMatch[1]; // e.g., "5" or "75"
      const divisor = scoreMatch[3] ? scoreMatch[3].replace(/\s*\/\s*/, '') : ''; // e.g., "/10" -> "10"
      const isPercentage = scoreMatch[4] && scoreMatch[4].includes('%');

      if (isPercentage) {
        return `${parseInt(value)}/100`;
      } else if (divisor) {
        return `${value}/${divisor}`;
      } else {
        // If it's just "Score: 75", assume out of 100
        return `${parseInt(value)}/100`;
      }
    }

    // Fallback patterns if explicit "Score:" is not found at the beginning
    let match = markdown.match(/^(\d+)%/); // e.g., "90%"
    if (match) {
      return `${match[1]}/100`;
    }

    match = markdown.match(/^(\d+)\s*\/\s*(\d+)/); // e.g., "7/10"
    if (match) {
      return `${match[1]}/${match[2]}`;
    }

    match = markdown.match(/^(\d+)(?:\.\d+)?$/); // e.g., "75"
    if (match) {
        return `${parseInt(match[1])}/100`;
    }

    // Final fallback: take the first line or a default message
    const firstLine = markdown.split('\n')[0].trim();
    if (firstLine.length > 50) return "Score Available"; // Prevent long first lines
    return firstLine || "Score Available";
  };

  // NEW: Function to fetch saved resumes based on filter and sort (wrapped in useCallback)
  // Removed sortKey and sortOrder from useCallback dependencies as they are passed as arguments
  const fetchSavedResumes = useCallback(async (roleFilter, currentSortKey, currentSortOrder) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/get_saved_resumes?role=${encodeURIComponent(roleFilter)}&sort_key=${encodeURIComponent(currentSortKey)}&sort_order=${encodeURIComponent(currentSortOrder)}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setSavedResumes(data);
    } catch (err) {
      console.error("Failed to fetch saved resumes:", err);
      setError("Failed to fetch saved resumes: " + err.message);
    } finally {
      setLoading(false);
    }
  }, []); // Empty dependency array now, as arguments are external dependencies

  // Initial data fetch on component mount
  useEffect(() => {
    const fetchInitialData = async () => {
      setLoading(true);
      setError(null);
      try {
        const optionsRes = await fetch(`${API_BASE_URL}/jd_options`);
        if (!optionsRes.ok) throw new Error(`HTTP error! status: ${optionsRes.status}`);
        const optionsData = await optionsRes.json();
        // Add 'All Roles' for filtering saved resumes
        setJdOptions(['All Roles', ...optionsData, 'Custom Input']);

        const defaultJdRes = await fetch(`${API_BASE_URL}/jd_default`);
        if (!defaultJdRes.ok) throw new Error(`HTTP error! status: ${defaultJdRes.status}`);
        const defaultJdData = await defaultJdRes.json();
        setJdText(defaultJdData);

        // Fetch initial saved resumes with current sort states
        fetchSavedResumes('All Roles', sortKey, sortOrder); 

      } catch (err) {
        console.error("Failed to load initial data:", err);
        setError("Failed to load initial data. Please ensure the backend server is running.");
      } finally {
        setLoading(false);
      }
    };
    fetchInitialData();
  }, [fetchSavedResumes, sortKey, sortOrder]); // fetchSavedResumes is a dependency, sortKey and sortOrder are still needed here


  // Handler for JD role selection change
  const handleJDRoleChange = async (e) => {
    const role = e.target.value;
    setSelectedJDRole(role);
    // If not 'Custom Input' or 'All Roles', fetch JD text
    if (role !== 'Custom Input' && role !== 'All Roles') {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/jd_text`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ role: role }),
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setJdText(data);
      } catch (err) {
        console.error("Failed to fetch JD text:", err);
        setError("Failed to fetch JD text: " + err.message);
      } finally {
        setLoading(false);
      }
    } else if (role === 'Custom Input') {
      setJdText(''); // Clear JD text for custom input
    }
  };

  // Handler for file input change
  const handleResumeFileChange = (e) => {
    setResumeFile(e.target.files[0]);
    // Clear previous outputs when a new file is selected
    setParsedResumeOutput('');
    setResumeTextCache('');
    setResumeTableOutput('');
    setResumeCheckOutput('');
    setJdMatchOutput('');
    setInterviewQaOutput('');
    setFitScoreOutput('');
    setExtractedResumeName(''); // Clear extracted name
    setTempSavedFilename(''); // Clear temp filename
    setError(null);
  };

  // Handler for parsing resume
  const handleParseResume = async () => {
    if (!resumeFile) {
      setError("Please upload a PDF resume.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('resume', resumeFile); // 'resume' must match Flask's request.files['resume']

      const response = await fetch(`${API_BASE_URL}/parse_resume`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setParsedResumeOutput(data.display_output);
        setResumeTextCache(data.raw_parsed_text);
        setExtractedResumeName(data.extracted_name); // Store extracted name from backend
        setTempSavedFilename(data.temp_saved_filename); // Store temp filename from backend
        setActiveTab('parse'); // Switch to parsed resume tab
      } else {
        setError(data.error || "Failed to parse resume.");
        setParsedResumeOutput(`\`\`\`plain\\nError: ${data.error || "Failed to parse resume."}\\n\`\`\``);
      }
    } catch (err) {
      console.error("Error parsing resume:", err);
      setError("Error parsing resume: " + err.message);
      setParsedResumeOutput(`\`\`\`plain\\nError: ${err.message}\\n\`\`\``);
    } finally {
      setLoading(false);
    }
  };

  // Handlers for other functionalities (Generate Resume Table, Resume Check, JD Match, Generate Questions, Fit Score)
  // These remain largely the same, but ensure they use `resumeTextCache` and `jdText` correctly.

  const handleGenerateResumeTable = async () => {
    if (!resumeTextCache) {
      setError("Please parse a resume first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/generate_resume_table`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume_text_cache: resumeTextCache }),
      });
      const data = await response.json();
      if (response.ok) {
        setResumeTableOutput(data.output);
        setActiveTab('table'); // Switch to table tab after generation
      } else {
        setError(data.error || "Failed to generate resume table.");
      }
    } catch (err) {
      console.error("Error generating resume table:", err);
      setError("Error generating resume table: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleResumeCheck = async () => {
    if (!resumeTextCache) {
      setError("Please parse a resume first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/resume_check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume_text: resumeTextCache }),
      });
      const data = await response.json();
      if (response.ok) {
        setResumeCheckOutput(data.output);
        setActiveTab('check'); // Switch to check tab after generation
      } else {
        setError(data.error || "Failed to perform resume check.");
      }
    } catch (err) {
      console.error("Error performing resume check:", err);
      setError("Error performing resume check: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleJDMatch = async () => {
    if (!resumeTextCache || !jdText) {
      setError("Please parse a resume and provide a job description.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/jd_match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume_text: resumeTextCache, jd_text: jdText }),
      });
      const data = await response.json();
      if (response.ok) {
        setJdMatchOutput(data.output);
        setActiveTab('jd-match'); // Switch to JD match tab after generation
      } else {
        setError(data.error || "Failed to generate JD match.");
      }
    } catch (err) {
      console.error("Error generating JD match:", err);
      setError("Error generating JD match: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateQuestions = async () => {
    if (!resumeTextCache || !jdText) {
      setError("Please parse a resume and provide a job description.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/generate_questions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume_text: resumeTextCache, jd_text: jdText }),
      });
      const data = await response.json();
      if (response.ok) {
        setInterviewQaOutput(data.output);
        setActiveTab('interview-qa'); // Switch to Q&A tab after generation
      } else {
        setError(data.error || "Failed to generate questions.");
      }
    } catch (err) {
      console.error("Error generating questions:", err);
      setError("Error generating questions: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFitScore = async () => {
    if (!resumeTextCache || !jdText) {
      setError("Please parse a resume and provide a job description.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/fit_score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume_text: resumeTextCache, jd_text: jdText }),
      });
      const data = await response.json();
      if (response.ok) {
        setFitScoreOutput(data.output);
        // No tab switch here, as the score output is visible in the input section
      } else {
        setError(data.error || "Failed to calculate fit score.");
      }
    } catch (err) {
      console.error("Error calculating fit score:", err);
      setError("Error calculating fit score: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // NEW: Confirm Document Handler
  const handleConfirmDocument = async () => {
    if (!resumeTextCache || !jdText || !fitScoreOutput || !interviewQaOutput || !extractedResumeName || !tempSavedFilename) {
      setError("Please ensure a resume is parsed, fit score and interview Q&A are generated, and a name is extracted before confirming.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/confirm_document`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resume_text_cache: resumeTextCache,
          jd_text: jdText,
          fit_score_output: fitScoreOutput,
          interview_qa_output: interviewQaOutput,
          selected_jd_role: selectedJDRole,
          original_file_name: resumeFile.name, // Original name from user's file system
          temp_saved_filename: tempSavedFilename, // Unique temporary name from backend
          parsed_resume_name: extractedResumeName, // Extracted name
          timestamp: new Date().toISOString() // Current timestamp
        }),
      });
      const data = await response.json();
      if (response.ok) {
        // In a real application, replace alert with a custom modal or toast notification
        alert("Document saved successfully!"); 
        // Refresh the list of saved resumes
        await fetchSavedResumes(savedJdFilter, sortKey, sortOrder); // Pass current sort states
        // Clear current form inputs for a new entry
        handleClearOutputs(); // Use the new clear outputs function
        setActiveTab('all-resumes'); // Switch to all resumes tab after saving
      } else {
        setError(data.error || "Failed to confirm and save document.");
      }
    } catch (err) {
      console.error("Error confirming document:", err);
      setError("Error confirming document: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // NEW: Download Resume Handler
  const handleDownloadResume = (filename) => {
    window.open(`${API_BASE_URL}/download_resume/${filename}`, '_blank');
  };

  // NEW: Toggle Interview QA Display
  const toggleInterviewQa = async (id, filename) => {
    if (showInterviewQaId === id) {
      setShowInterviewQaId(null); // Hide if already open
    } else {
      setLoading(true); // Small loading for QA fetch
      setError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/get_interview_qa/${filename}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        // Temporarily store the QA content in the item for display
        setSavedResumes(prevResumes => prevResumes.map(resume =>
          resume.id === id ? { ...resume, qa_content_display: data.qa_content } : resume
        ));
        setShowInterviewQaId(id);
      } catch (err) {
        console.error("Error fetching interview QA:", err);
        setError("Error fetching interview QA: " + err.message);
      } finally {
        setLoading(false);
      }
    }
  };

  // NEW: Handler to clear all current outputs (but not saved resumes)
  const handleClearOutputs = () => {
    setResumeFile(null);
    setParsedResumeOutput('');
    setResumeTextCache('');
    setResumeTableOutput('');
    setResumeCheckOutput('');
    setJdMatchOutput('');
    setInterviewQaOutput('');
    setFitScoreOutput('');
    setExtractedResumeName('');
    setTempSavedFilename('');
    setError(null);
    setShowFullFitScoreReview(false);
    // Reset file input display
    const fileInput = document.getElementById('resume-upload');
    if (fileInput) fileInput.value = '';
    setActiveTab('parse'); // Go back to parse tab
  };

  // NEW: Handler to clear EVERYTHING (including saved data)
  const handleClearEverything = async () => {
    // Using window.confirm for simplicity, replace with a custom modal in production
    if (window.confirm("WARNING: This will delete ALL saved resumes and analyses permanently.\nNote: This will delete all saved resumes. Are you sure?")) { 
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/clear_all_data`, {
          method: 'POST',
        });
        const data = await response.json();
        if (response.ok) {
          alert("All data cleared successfully!"); // Use custom modal in production
          handleClearOutputs(); // Clear current outputs
          await fetchSavedResumes('All Roles', sortKey, sortOrder); // Refresh saved resumes (should be empty)
        } else {
          setError(data.error || "Failed to clear all data.");
        }
      } catch (err) {
        console.error("Error clearing all data:", err);
        setError("Error clearing all data: " + err.message);
      } finally {
        setLoading(false);
      }
    }
  };

  // Handler for sorting saved resumes
  const handleSortChange = (key) => {
    let newOrder = 'asc';
    if (sortKey === key && sortOrder === 'asc') {
      newOrder = 'desc';
    }
    setSortKey(key);
    setSortOrder(newOrder);
    // fetchSavedResumes is already called via useEffect when sortKey/sortOrder change
  };


  return (
    <div className="App">
      {/* Hero Section: The background for this whole area is handled by the <body> CSS */}
      <div className="hero-section">
        <div className="hero-background"></div>
        {/* hero-content now contains only text */}
        <div className="hero-content">
          {/* Changed the heading text to the new project name */}
          <h1>ResumeFlow AI</h1>
          <p>Instantly analyze your resume, optimize for job descriptions, and prepare for interviews.</p>
        </div>
      </div>

      <main className="main-content">
        <section className="input-section">
          <h2 className="section-title">Resume & JD Analyzer</h2>
          {error && <div className="error-message">{error}</div>}

          <div className="input-grid">
            <div className="input-group">
              <label htmlFor="resume-upload">Upload Your Resume (PDF)</label>
              <input type="file" id="resume-upload" accept=".pdf" onChange={handleResumeFileChange} />
              <button onClick={handleParseResume} disabled={loading || !resumeFile}>
                {loading && activeTab === 'parse' ? 'Parsing...' : 'Parse Resume'}
              </button>
            </div>

            {/* Added job-description-group class here */}
            <div className="input-group job-description-group">
              <label htmlFor="jd-select">Select Job Description Role</label>
              <select id="jd-select" value={selectedJDRole} onChange={handleJDRoleChange} disabled={loading}>
                {jdOptions.filter(opt => opt !== 'All Roles').map((option) => ( // Filter out 'All Roles' for this selector
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>

              <label htmlFor="jd-text-input">Or Paste Custom Job Description</label> {/* Added id */}
              <textarea
                id="jd-text-input" // Added id
                placeholder="Paste your job description here..."
                value={jdText}
                onChange={(e) => setJdText(e.target.value)}
                disabled={selectedJDRole !== 'Custom Input' || loading}
              ></textarea>
            </div>

            <div className="input-group score-input-group">
                {/* Removed htmlFor attribute from label as it targets a button, not an input */}
                <label>Calculate Resume Fit Score</label> 
                <button onClick={handleFitScore} disabled={loading || !resumeTextCache || !jdText}>
                    {loading && activeTab === 'fit-score' ? 'Calculating...' : 'Get Fit Score'}
                </button>
                {fitScoreOutput && (
                    <>
                        <div className="score-output-box">
                            <span className="medal-icon">🏅</span>
                            <div className="score-value">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {getDisplayScore(fitScoreOutput)}
                                </ReactMarkdown>
                            </div>
                        </div>
                        <button
                            onClick={() => setShowFullFitScoreReview(!showFullFitScoreReview)}
                            className="toggle-review-button"
                        >
                            {showFullFitScoreReview ? 'Hide Full Review' : 'Show Full Review'}
                        </button>
                        {showFullFitScoreReview && (
                            <div className="full-review-dropdown">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {fitScoreOutput}
                                </ReactMarkdown>
                            </div>
                        )}
                    </>
                )}
                {/* NEW: Confirm Document Button */}
                {(fitScoreOutput && interviewQaOutput && extractedResumeName && tempSavedFilename) && (
                    <button
                        className="confirm-document-button"
                        onClick={handleConfirmDocument}
                        disabled={loading}
                    >
                        Confirm Document
                    </button>
                )}
            </div>
          </div>

          <div className="action-buttons-group">
            <button onClick={handleClearOutputs} disabled={loading}>
                Clear All Outputs
            </button>
            <button onClick={handleClearEverything} disabled={loading} className="clear-everything-button">
                Clear EVERYTHING ⚠️ (Note: This will delete all saved resumes)
            </button>
          </div>

        </section>

        {loading && (
            <section className="loading-overlay-section">
                <div className="loading-message">
                    <img src={LoaderGif} alt="Loading..." className="loader-image" />
                    <p>Processing your request...</p>
                </div>
            </section>
        )}

        <section className="output-section">
          <div className="tabs">
            <button
              className={activeTab === 'parse' ? 'active' : ''}
              onClick={() => setActiveTab('parse')}
            >
              Parsed Resume
            </button>
            <button
              className={activeTab === 'table' ? 'active' : ''}
              onClick={() => setActiveTab('table')} // Changed to just switch tab
              disabled={!resumeTextCache}
            >
              Resume Table
            </button>
            <button
              className={activeTab === 'check' ? 'active' : ''}
              onClick={() => setActiveTab('check')} // Changed to just switch tab
              disabled={!resumeTextCache}
            >
              Resume Check
            </button>
            <button
              className={activeTab === 'jd-match' ? 'active' : ''}
              onClick={() => setActiveTab('jd-match')} // Changed to just switch tab
              disabled={!resumeTextCache || !jdText}
            >
              JD Match
            </button>
            <button
              className={activeTab === 'interview-qa' ? 'active' : ''}
              onClick={() => setActiveTab('interview-qa')} // Changed to just switch tab
              disabled={!resumeTextCache || !jdText}
            >
              Interview Q&A
            </button>
            {/* NEW: All Resumes Tab */}
            <button
              className={activeTab === 'all-resumes' ? 'active' : ''}
              onClick={() => { setActiveTab('all-resumes'); fetchSavedResumes(savedJdFilter, sortKey, sortOrder); }}
            >
              All Resumes
            </button>
          </div>

          <div className="tab-content">
            {activeTab === 'parse' && (
              <div className="tab-pane">
                <h3>Parsed Resume Text</h3>
                <div className="output-area code-block">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {parsedResumeOutput || 'Upload and parse a resume to see the extracted text here.'}
                  </ReactMarkdown>
                </div>
              </div>
            )}
            {activeTab === 'table' && (
              <div className="tab-pane">
                <h3>Generated Resume Table</h3>
                {/* Add button to trigger generation */}
                <button onClick={handleGenerateResumeTable} disabled={loading || !resumeTextCache}>
                    Generate Table
                </button>
                <div className="output-area markdown-output">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {resumeTableOutput || 'Click "Generate Table" to get structured resume data.'}
                  </ReactMarkdown>
                </div>
              </div>
            )}
            {activeTab === 'check' && (
              <div className="tab-pane">
                <h3>Resume Check & Feedback</h3>
                {/* Add button to trigger generation */}
                <button onClick={handleResumeCheck} disabled={loading || !resumeTextCache}>
                    Perform Check
                </button>
                <div className="output-area markdown-output">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {resumeCheckOutput || 'Click "Perform Check" to get feedback on your resume.'}
                  </ReactMarkdown>
                </div>
              </div>
            )}
            {activeTab === 'jd-match' && (
              <div className="tab-pane">
                <h3>Job Description Match Analysis</h3>
                {/* Add button to trigger generation */}
                <button onClick={handleJDMatch} disabled={loading || !resumeTextCache || !jdText}>
                    Analyze JD Match
                </button>
                <div className="output-area markdown-output">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {jdMatchOutput || 'Click "Analyze JD Match" to see how well your resume matches the job description.'}
                  </ReactMarkdown>
                </div>
              </div>
            )}
            {activeTab === 'interview-qa' && (
              <div className="tab-pane">
                <h3>Interview Questions & Answers</h3>
                {/* Add button to trigger generation */}
                <button onClick={handleGenerateQuestions} disabled={loading || !resumeTextCache || !jdText}>
                    Generate Q&A
                </button>
                <div className="output-area markdown-output">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {interviewQaOutput || 'Click "Generate Q&A" to generate personalized interview questions.'}
                  </ReactMarkdown>
                </div>
              </div>
            )}
            {/* All Resumes Tab Content */}
            {activeTab === 'all-resumes' && (
              <div className="tab-pane all-resumes-section">
                <h3>All Saved Resumes</h3>
                <div className="jd-filter-container">
                    <label htmlFor="saved-jd-filter">Filter by Job Role:</label>
                    <select id="saved-jd-filter" value={savedJdFilter} onChange={(e) => {
                        setSavedJdFilter(e.target.value);
                        fetchSavedResumes(e.target.value, sortKey, sortOrder);
                    }} disabled={loading}>
                        {jdOptions.map((option) => ( // Show all options including 'All Roles'
                            <option key={option} value={option}>{option}</option>
                        ))}
                    </select>
                </div>

                {/* NEW: Sorting Controls */}
                <div className="sort-controls">
                    <span>Sort by:</span>
                    <button 
                        onClick={() => handleSortChange('person_name')} 
                        className={sortKey === 'person_name' ? 'active' : ''}
                    >
                        Name {sortKey === 'person_name' && (sortOrder === 'asc' ? '▲' : '▼')}
                    </button>
                    <button 
                        onClick={() => handleSortChange('fit_score')} 
                        className={sortKey === 'fit_score' ? 'active' : ''}
                    >
                        Score {sortKey === 'fit_score' && (sortOrder === 'asc' ? '▲' : '▼')}
                    </button>
                    <button 
                        onClick={() => handleSortChange('timestamp')} 
                        className={sortKey === 'timestamp' ? 'active' : ''}
                    >
                        Date {sortKey === 'timestamp' && (sortOrder === 'asc' ? '▲' : '▼')}
                    </button>
                </div>


                {loading ? (
                    <div className="loading-message">
                        <img src={LoaderGif} alt="Loading..." className="loader-image" />
                        <p>Loading saved resumes...</p>
                    </div>
                ) : savedResumes.length === 0 ? (
                    <p className="no-resumes-message">No resumes saved yet for this role. Confirm a document to see it here!</p>
                ) : (
                    <div className="saved-resumes-list">
                        {savedResumes.map((resume) => (
                            <div key={resume.id} className="saved-resume-item">
                                <div className="resume-info">
                                    <span className="person-name">{resume.person_name}</span>
                                    <span className="jd-role-tag">{resume.jd_role}</span>
                                    <span className="fit-score-display">
                                        Score: <ReactMarkdown remarkPlugins={[remarkGfm]}>{getDisplayScore(resume.fit_score)}</ReactMarkdown>
                                    </span>
                                </div>
                                <div className="resume-actions">
                                    {resume.resume_filename && (
                                        <button
                                            className="download-button"
                                            onClick={() => handleDownloadResume(resume.resume_filename)}
                                        >
                                            Download Resume
                                        </button>
                                    )}
                                    {resume.qa_filename && (
                                        <button
                                            className="toggle-qa-button"
                                            onClick={() => toggleInterviewQa(resume.id, resume.qa_filename)}
                                        >
                                            {showInterviewQaId === resume.id ? 'Hide Q&A' : 'Show Q&A'}
                                        </button>
                                    )}
                                </div>
                                {showInterviewQaId === resume.id && resume.qa_content_display && (
                                    <div className="interview-qa-dropdown output-area markdown-output">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                            {resume.qa_content_display}
                                        </ReactMarkdown>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
