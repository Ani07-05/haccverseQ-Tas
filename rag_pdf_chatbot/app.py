import streamlit as st
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from PyPDF2 import PdfReader
import pandas as pd
import base64
import json
import re
import os
import uuid
import numpy as np
from datetime import datetime, timedelta

# Update imports for LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# For parsing resumes
import docx2txt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Load environment variables
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)
if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY not found in environment variables. Please check your .env file."
    )
    st.stop()


# Resume parsing functions
def extract_text_from_pdf(pdf_file):
    """Extract text content from PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(docx_file):
    """Extract text content from DOCX file"""
    text = docx2txt.process(docx_file)
    return text


def extract_text_from_txt(txt_file):
    """Extract text content from TXT file"""
    return txt_file.read().decode("utf-8")


def extract_resume_text(file):
    """Extract text from resume file based on file type"""
    file_ext = os.path.splitext(file.name)[1].lower()

    if file_ext == ".pdf":
        return extract_text_from_pdf(file)
    elif file_ext == ".docx":
        return extract_text_from_docx(file)
    elif file_ext == ".txt":
        return extract_text_from_txt(file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def extract_skills(text):
    """Extract skills from resume text with domain awareness"""
    # Common general skills
    common_skills = [
        "python",
        "java",
        "javascript",
        "typescript",
        "html",
        "css",
        "react",
        "angular",
        "vue",
        "node",
        "express",
        "django",
        "flask",
        "spring",
        "sql",
        "mongodb",
        "nosql",
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "ci/cd",
        "git",
        "agile",
        "scrum",
        "machine learning",
        "artificial intelligence",
        "data science",
        "nlp",
        "computer vision",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
        "scikit-learn",
        "matlab",
        "r",
        "tableau",
        "power bi",
        "excel",
        "word",
        "powerpoint",
        "photoshop",
        "illustrator",
        "figma",
        "ui/ux",
        "product management",
        "project management",
        "leadership",
        "communication",
    ]

    # Domain-specific skills by category
    domain_skills = {
        "cybersecurity": [
            "security",
            "cybersecurity",
            "pentesting",
            "penetration testing",
            "ethical hacking",
            "vulnerability assessment",
            "network security",
            "firewalls",
            "ids",
            "ips",
            "siem",
            "threat detection",
            "incident response",
            "forensics",
            "malware analysis",
            "reverse engineering",
            "cryptography",
            "encryption",
            "secure coding",
            "owasp",
            "web security",
            "application security",
            "cloud security",
            "security+",
            "cissp",
            "ceh",
            "oscp",
            "capture the flag",
            "ctf",
            "wireshark",
            "nmap",
            "metasploit",
            "kali linux",
            "burp suite",
            "snort",
            "splunk",
            "virus",
            "ransomware",
            "phishing",
            "social engineering",
            "zero-day",
            "exploit",
            "vulnerability",
            "threat hunting",
            "breach",
            "attack",
            "defense",
            "security operations",
            "soc",
            "compliance",
            "gdpr",
            "hipaa",
            "pci dss",
            "iso 27001",
            "authentication",
            "authorization",
            "access control",
            "identity management",
            "vpn",
            "dos",
            "ddos",
            "waf",
            "firewall",
            "ids/ips",
            "hashing",
            "security audit",
            "red team",
            "blue team",
            "security assessment",
            "security architecture",
            "security design",
            "security testing",
            "security analysis",
            "security monitoring",
            "security tools",
            "security protocols",
            "security frameworks",
            "security standards",
            "security policies",
            "security procedures",
            "linux",
            "networking",
            "network protocols",
            "tcp/ip",
            "dns",
            "http/https",
        ],
        "data_science": [
            "data analysis",
            "data mining",
            "data visualization",
            "statistics",
            "big data",
            "hadoop",
            "spark",
            "data modeling",
            "data warehouse",
            "data lake",
            "etl",
            "business intelligence",
            "predictive modeling",
            "predictive analytics",
            "regression",
            "classification",
            "clustering",
            "dimensionality reduction",
            "feature engineering",
            "feature selection",
            "time series analysis",
            "a/b testing",
            "hypothesis testing",
            "statistical significance",
            "anomaly detection",
            "recommender systems",
            "natural language processing",
        ],
        "web_development": [
            "frontend",
            "backend",
            "full stack",
            "web design",
            "responsive design",
            "progressive web apps",
            "single page applications",
            "restful api",
            "graphql",
            "microservices",
            "serverless",
            "jamstack",
            "webpack",
            "babel",
            "sass",
            "less",
            "bootstrap",
            "material ui",
            "tailwind css",
            "jquery",
            "ajax",
            "json",
            "xml",
            "seo",
            "web performance",
            "web accessibility",
            "web security",
            "cross-browser compatibility",
            "browser dev tools",
            "cdn",
        ],
        "mobile_development": [
            "android",
            "ios",
            "swift",
            "objective-c",
            "kotlin",
            "java",
            "react native",
            "flutter",
            "xamarin",
            "ionic",
            "cordova",
            "mobile ui",
            "mobile ux",
            "responsive design",
            "app store",
            "google play",
            "push notifications",
            "geolocation",
            "camera",
            "sensors",
            "offline storage",
            "mobile performance",
            "mobile security",
            "mobile testing",
            "mobile analytics",
            "mobile seo",
        ],
    }

    # Extract generic skills
    skills = []
    text_lower = text.lower()

    # Extract general skills
    for skill in common_skills:
        if skill in text_lower:
            skills.append(skill)

    # Extract domain-specific skills and tag them
    domain_tagged_skills = {}
    for domain, domain_skill_list in domain_skills.items():
        domain_skills_found = []
        for skill in domain_skill_list:
            if skill in text_lower and skill not in skills:
                domain_skills_found.append(skill)
                skills.append(skill)

        if domain_skills_found:
            domain_tagged_skills[domain] = domain_skills_found

    # Return both the flat list and the categorized skills
    return skills, domain_tagged_skills


def extract_education(text):
    """Extract education information from resume text"""
    education = []
    education_keywords = [
        "bachelor",
        "master",
        "phd",
        "doctorate",
        "degree",
        "university",
        "college",
        "school",
        "institute",
    ]

    lines = text.split("\n")
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in education_keywords):
            # Remove extra whitespace
            clean_line = " ".join(line.split())
            if clean_line:
                education.append(clean_line)

    return education


def extract_experience(text):
    """Enhanced experience calculation with accurate work day counting"""

    def parse_date(date_str):
        months = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

        date_str = date_str.lower().strip()
        parts = date_str.split()

        if len(parts) != 2:
            return None

        month_str = parts[0][:3]
        year_str = parts[1]

        if month_str not in months:
            return None

        try:
            year = int(year_str)
            month = months[month_str]
            return datetime(year, month, 1)  # Return datetime object
        except ValueError:
            return None

    # Store all date ranges to handle overlaps
    date_ranges = []
    dates_found = False

    # Look for date ranges with months
    date_pattern = r"([A-Za-z]+)\s+(\d{4})\s*[-–—]\s*([A-Za-z]+)\s+(\d{4})|([A-Za-z]+)\s+(\d{4})\s*[-–—]\s*(Present|Current|Now)"

    for match in re.finditer(date_pattern, text):
        dates_found = True
        start_date = None
        end_date = None

        if match.group(7):  # Present date format
            start_month = match.group(5)
            start_year = match.group(6)
            start_date = parse_date(f"{start_month} {start_year}")
            end_date = datetime.now()
        else:  # Full date range format
            start_month, start_year = match.group(1), match.group(2)
            end_month, end_year = match.group(3), match.group(4)

            start_date = parse_date(f"{start_month} {start_year}")
            end_date = parse_date(f"{end_month} {end_year}")

        if start_date and end_date and start_date < end_date:
            date_ranges.append((start_date, end_date))

    # If no proper date ranges found, return None
    if not dates_found:
        return None

    # Sort date ranges by start date
    date_ranges.sort(key=lambda x: x[0])

    # Merge overlapping ranges
    merged_ranges = []
    for start, end in date_ranges:
        if not merged_ranges or start > merged_ranges[-1][1]:
            merged_ranges.append((start, end))
        else:
            # Extend the last range if there's overlap
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))

    # Calculate total days excluding weekends
    total_days = 0
    for start, end in merged_ranges:
        delta = end - start
        weeks = delta.days // 7
        remaining_days = delta.days % 7

        # Count weekdays in complete weeks (5 days per week)
        weekdays = weeks * 5

        # Add remaining days, excluding weekends
        for i in range(remaining_days):
            day = (start + timedelta(weeks * 7 + i)).weekday()
            if day < 5:  # Monday = 0, Friday = 4
                weekdays += 1

        total_days += weekdays

    # Convert work days to years (52 weeks * 5 workdays = 260 workdays per year)
    years = round(total_days / 260, 1)
    return years


def extract_email(text):
    """Extract email from resume text"""
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    match = re.search(email_pattern, text)
    if match:
        return match.group(0)
    return None


def extract_phone(text):
    """Extract phone number from resume text"""
    phone_pattern = r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(phone_pattern, text)
    if match:
        return match.group(0)
    return None


def extract_name(text):
    """Extract name from resume text (simple heuristic)"""
    # Take first line that's not empty
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        # Assume the first line might be the name if it's short
        if len(lines[0].split()) <= 4:
            return lines[0]
    return "Unknown"


def extract_positions(text):
    """Extract job positions and their details from resume text"""
    positions = []

    # Common job title indicators
    job_indicators = [
        "developer",
        "engineer",
        "manager",
        "consultant",
        "specialist",
        "analyst",
        "designer",
        "intern",
        "internship",
        "administrator",
        "director",
        "coordinator",
        "associate",
        "assistant",
        "lead",
        "head",
        "portfolio",
    ]

    # Capture any section that looks like a position with date
    # Look for lines that contain a job title followed by or preceded by a date range
    lines = text.split("\n")

    for i, line in enumerate(lines):
        line_lower = line.lower()

        # Check if this line contains a job indicator
        if any(indicator in line_lower for indicator in job_indicators):
            # Look for date pattern in this line or the next few lines
            date_range = None
            title = line.strip()

            # Check current line for date
            date_match1 = re.search(
                r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*[-–—]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|Present|Current|Now))",
                line,
            )
            date_match2 = re.search(
                r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})",
                line,
            )

            if date_match1:
                date_range = date_match1.group(1).strip()
                # Remove the date from the title
                title = line.replace(date_range, "").strip()
            elif date_match2 and "Present" in line:
                month_year = date_match2.group(1).strip()
                date_range = f"{month_year} - Present"
                # Remove the date from the title
                title = line.replace(month_year, "").replace("- Present", "").strip()
            elif i < len(lines) - 1:  # Check if next line has a date
                next_line = lines[i + 1]
                date_match1 = re.search(
                    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*[-–—]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|Present|Current|Now))",
                    next_line,
                )

                if date_match1:
                    date_range = date_match1.group(1).strip()

            # Clean up the title
            if title:
                # Remove common patterns like ';' or ':' and anything after them
                semicolon_pos = title.find(";")
                if semicolon_pos > 0:
                    title = title[:semicolon_pos].strip()

                colon_pos = title.find(":")
                if colon_pos > 0:
                    title = title[:colon_pos].strip()

                # Add the position only if we have both title and date
                if date_range:
                    positions.append({"title": title, "date_range": date_range})

    return positions


def generate_concept_questions(
    api_key, language, difficulty, num_questions, resume_data
):
    """
    Generate concept-based interview questions for a specific programming language.

    Args:
        api_key (str): Google AI API key
        language (str): Programming language
        difficulty (str): Difficulty level (Basic, Intermediate, Advanced)
        num_questions (int): Number of questions to generate
        resume_data (DataFrame): Shortlisted candidate data

    Returns:
        list: List of question dictionaries with question, answer, and follow-up
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key
    )

    # Extract skills from resumes to contextualize questions
    skills = []
    if "Skills" in resume_data.columns:
        for skill_list in resume_data["Skills"].dropna():
            if isinstance(skill_list, str):
                skills.extend([s.strip().lower() for s in skill_list.split(",")])

    unique_skills = list(set(skills))
    relevant_skills = [
        s
        for s in unique_skills
        if s != language
        and s
        in [
            "aws",
            "docker",
            "kubernetes",
            "cloud",
            "database",
            "sql",
            "nosql",
            "frontend",
            "backend",
            "web",
            "api",
            "rest",
            "microservices",
        ]
    ]

    # Create prompt with context
    prompt = f"""
    Generate {num_questions} {difficulty.lower()} {language} programming concept questions.
    
    Requirements:
    1. Each question must be 30 words or less
    2. Provide answers in bullet points (max 5 points)
    3. Each point should be clear and concise
    4. Include one follow-up question (15 words max)
    
    Consider that candidates also have these skills: {', '.join(relevant_skills[:5])}
    
    Return as JSON array with 'question', 'answer', and 'follow_up' keys.
    """

    response = model.invoke(prompt)
    response_text = response.content

    # Extract JSON from response
    json_start = response_text.find("[")
    json_end = response_text.rfind("]") + 1

    if json_start >= 0 and json_end > json_start:
        json_str = response_text[json_start:json_end]
        try:
            questions = json.loads(json_str)
            return questions
        except:
            # Fallback if JSON parsing fails
            return [
                {
                    "question": f"Question about {language} {difficulty.lower()} concepts",
                    "answer": "This is a placeholder answer due to parsing issues.",
                }
            ]
    else:
        # Fallback for no JSON found
        return [
            {
                "question": f"Question about {language} {difficulty.lower()} concepts",
                "answer": "This is a placeholder answer.",
            }
        ]


def generate_coding_questions(api_key, difficulty, topics, num_questions):
    """
    Generate coding/LeetCode style interview questions.

    Args:
        api_key (str): Google AI API key
        difficulty (str): Difficulty level (Easy, Medium, Hard)
        topics (list): List of topics to focus on
        num_questions (int): Number of questions to generate

    Returns:
        list: List of question dictionaries with details
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key
    )

    topics_str = ", ".join(topics) if topics else "various topics"

    prompt = f"""
    Generate {num_questions} {difficulty.lower()} LeetCode-style coding interview questions focused on {topics_str}.
    
    For each question, provide:
    1. A title for the problem
    2. The difficulty level (Easy, Medium, Hard)
    3. A detailed problem description with examples
    4. The platform (e.g., LeetCode, HackerRank)
    5. A fictional link to the problem
    6. 2-3 hints that progressively guide toward the solution
    
    Return the results as a JSON array where each item has 'title', 'difficulty', 'description', 'platform', 'link', and 'hints' keys.
    """

    response = model.invoke(prompt)
    response_text = response.content

    # Extract JSON from response
    json_start = response_text.find("[")
    json_end = response_text.rfind("]") + 1

    if json_start >= 0 and json_end > json_start:
        json_str = response_text[json_start:json_end]
        try:
            questions = json.loads(json_str)
            return questions
        except:
            # Fallback if JSON parsing fails
            return [
                {
                    "title": f"{difficulty} Coding Problem",
                    "difficulty": difficulty,
                    "description": "This is a placeholder description due to parsing issues.",
                    "platform": "LeetCode",
                    "link": "https://leetcode.com/problems/sample",
                }
            ]
    else:
        # Fallback for no JSON found
        return [
            {
                "title": f"{difficulty} Coding Problem",
                "difficulty": difficulty,
                "description": "This is a placeholder description.",
                "platform": "LeetCode",
                "link": "https://leetcode.com/problems/sample",
            }
        ]


def generate_project_questions(api_key, resume_data, questions_per_project):
    """
    Generate project-based interview questions based on projects in resumes.

    Args:
        api_key (str): Google AI API key
        resume_data (DataFrame): Shortlisted candidate data
        questions_per_project (int): Number of questions per project

    Returns:
        dict: Dictionary mapping project names to lists of question dictionaries
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key
    )

    # Extract project information from resume data
    projects = []
    if "Skills" in resume_data.columns:
        skills = []
        for skill_list in resume_data["Skills"].dropna():
            if isinstance(skill_list, str):
                skills.extend([s.strip() for s in skill_list.split(",")])

        # Use skills to infer potential projects
        web_dev_skills = [
            s
            for s in skills
            if s.lower()
            in ["react", "angular", "vue", "javascript", "html", "css", "nodejs"]
        ]
        ml_skills = [
            s
            for s in skills
            if s.lower()
            in ["machine learning", "tensorflow", "pytorch", "nlp", "computer vision"]
        ]
        mobile_skills = [
            s
            for s in skills
            if s.lower() in ["android", "ios", "react native", "flutter", "mobile"]
        ]

        if web_dev_skills:
            projects.append("Web Application Development")
        if ml_skills:
            projects.append("Machine Learning Model Development")
        if mobile_skills:
            projects.append("Mobile App Development")

    # Ensure we have at least one project
    if not projects:
        projects = ["Generic Software Development"]

    # Generate questions for each project
    result = {}

    for project in projects:
        prompt = f"""
        Generate {questions_per_project} in-depth interview questions about a {project} project.
        
        For each question:
        1. Create a detailed question that probes technical decisions, challenges, and solutions
        2. Provide an ideal response that demonstrates expertise
        3. Include a follow-up question to dig deeper
        
        Return the results as a JSON array where each item has 'question', 'ideal_response', and 'follow_up' keys.
        """

        response = model.invoke(prompt)
        response_text = response.content

        # Extract JSON from response
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                questions = json.loads(json_str)
                result[project] = questions
            except:
                # Fallback if JSON parsing fails
                result[project] = [
                    {
                        "question": f"Tell me about your role in the {project} project",
                        "ideal_response": "This is a placeholder response due to parsing issues.",
                    }
                ]
        else:
            # Fallback for no JSON found
            result[project] = [
                {
                    "question": f"Tell me about your role in the {project} project",
                    "ideal_response": "This is a placeholder response.",
                }
            ]

    return result


def convert_questions_to_csv(questions, question_type):
    """
    Convert interview questions to CSV format.

    Args:
        questions (list/dict): The questions to convert
        question_type (str): The type of questions

    Returns:
        str: CSV data as a string
    """
    csv_rows = []

    if question_type == "Concept Questions":
        # Header row
        csv_rows.append("Question,Answer,Follow-up")

        # Data rows
        for q in questions:
            question = q.get("question", "").replace(",", ";").replace("\n", " ")
            answer = q.get("answer", "").replace(",", ";").replace("\n", " ")
            follow_up = q.get("follow_up", "").replace(",", ";").replace("\n", " ")
            csv_rows.append(f'"{question}","{answer}","{follow_up}"')

    elif question_type == "Coding/LeetCode Questions":
        # Header row
        csv_rows.append("Title,Difficulty,Platform,Link,Description,Hints")

        # Data rows
        for q in questions:
            title = q.get("title", "").replace(",", ";").replace("\n", " ")
            difficulty = q.get("difficulty", "").replace(",", ";")
            platform = q.get("platform", "").replace(",", ";")
            link = q.get("link", "").replace(",", ";")
            description = q.get("description", "").replace(",", ";").replace("\n", " ")
            hints = ";".join(
                [h.replace(",", ";").replace("\n", " ") for h in q.get("hints", [])]
            )
            csv_rows.append(
                f'"{title}","{difficulty}","{platform}","{link}","{description}","{hints}"'
            )

    elif question_type == "Project-Based Questions":
        # Header row
        csv_rows.append("Project,Question,Ideal Response,Follow-up")

        # Data rows
        for project, project_questions in questions.items():
            for q in project_questions:
                project_name = project.replace(",", ";").replace("\n", " ")
                question = q.get("question", "").replace(",", ";").replace("\n", " ")
                ideal_response = (
                    q.get("ideal_response", "").replace(",", ";").replace("\n", " ")
                )
                follow_up = q.get("follow_up", "").replace(",", ";").replace("\n", " ")
                csv_rows.append(
                    f'"{project_name}","{question}","{ideal_response}","{follow_up}"'
                )

    # Join rows with newlines
    return "\n".join(csv_rows)


def extract_domain_specific_experience(content, domain):
    """
    Extract and score domain-specific experience from resume content.

    Args:
        content (str): The resume text content
        domain (str): The domain to score for ('cybersecurity', 'data_science', etc.)

    Returns:
        float: A score from 0.0 to 1.0 representing domain relevance
    """
    # Lower-case content for case-insensitive matching
    content_lower = content.lower()

    # Define domain indicators by domain
    domain_indicators = {
        "cybersecurity": [
            "security",
            "cybersecurity",
            "encryption",
            "firewall",
            "vulnerability",
            "penetration testing",
            "ethical hacking",
            "threat",
            "malware",
            "security+",
            "cissp",
            "ceh",
            "oscp",
            "incident response",
            "forensics",
            "security clearance",
        ],
        "data_science": [
            "data science",
            "machine learning",
            "artificial intelligence",
            "data analysis",
            "statistics",
            "data mining",
            "data visualization",
            "big data",
            "predictive modeling",
            "tensor",
            "pytorch",
            "scikit-learn",
            "data warehouse",
            "business intelligence",
        ],
        "web_development": [
            "web development",
            "frontend",
            "backend",
            "full stack",
            "react",
            "angular",
            "vue",
            "javascript",
            "html",
            "css",
            "nodejs",
            "web application",
            "responsive design",
            "web services",
            "rest api",
            "graphql",
            "web framework",
        ],
        "mobile_development": [
            "mobile development",
            "android",
            "ios",
            "swift",
            "kotlin",
            "react native",
            "flutter",
            "mobile app",
            "app development",
            "mobile application",
            "mobile ui",
            "responsive design",
            "app store",
            "google play",
        ],
    }

    # Check if domain is in our dictionary
    if domain not in domain_indicators:
        return 0.0

    # Count occurrences of domain indicators
    indicators = domain_indicators[domain]
    indicator_count = sum(1 for indicator in indicators if indicator in content_lower)

    # Check for common domain terms like "experience in {domain}" or "{domain} developer"
    domain_key_terms = [
        f"{domain}",
        f"{domain} experience",
        f"{domain} developer",
        f"{domain} engineer",
        f"{domain} specialist",
        f"{domain} analyst",
        f"experience in {domain}",
    ]

    # Replace underscores with spaces for domain name in key terms
    domain_key_terms = [term.replace("_", " ") for term in domain_key_terms]

    # Count key term occurrences
    key_term_count = sum(1 for term in domain_key_terms if term in content_lower)

    # Calculate base score from indicator matching
    max_indicators = min(len(indicators), 10)  # Cap at 10 for scoring
    base_score = min(indicator_count / max_indicators, 1.0)

    # Add bonus for key term matches
    key_term_bonus = min(key_term_count * 0.1, 0.3)  # Up to 0.3 bonus

    # Calculate final score, capped at 1.0
    final_score = min(base_score + key_term_bonus, 1.0)

    return final_score


def validate_experience_format(text):
    """Validate experience format and return validation results"""
    is_valid = True
    errors = []

    # Required format pattern
    date_pattern = r"([A-Za-z]+)\s+(\d{4})\s*[-–—]\s*([A-Za-z]+)\s+(\d{4})|([A-Za-z]+)\s+(\d{4})\s*[-–—]\s*(Present|Current|Now)"

    # Find all experience date ranges
    matches = re.finditer(date_pattern, text)
    dates = []

    for match in matches:
        if match.group(7):  # Present date format
            month = match.group(5)
            year = match.group(6)
            if not month or not year:
                is_valid = False
                errors.append("Missing month or year in experience dates")
                continue
            dates.append((month, int(year), "Present"))
        else:
            start_month = match.group(1)
            start_year = match.group(2)
            end_month = match.group(3)
            end_year = match.group(4)

            if not all([start_month, start_year, end_month, end_year]):
                is_valid = False
                errors.append("Missing month or year in experience dates")
                continue

            dates.append((start_month, int(start_year), end_month, int(end_year)))

    if not dates:
        is_valid = False
        errors.append("No properly formatted experience dates found")
        return is_valid, errors

    # Check chronological order
    for i in range(len(dates) - 1):
        curr_end_year = dates[i][3] if len(dates[i]) == 4 else datetime.now().year
        next_start_year = dates[i + 1][1]

        if curr_end_year > next_start_year:
            is_valid = False
            errors.append("Experience dates not in chronological order")
            break

    # Check for valid months
    valid_months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]

    for date in dates:
        if date[0].lower() not in valid_months:
            is_valid = False
            errors.append(f"Invalid month format: {date[0]}")
            break

        if len(date) == 4 and date[2].lower() not in valid_months:
            is_valid = False
            errors.append(f"Invalid month format: {date[2]}")
            break

    return is_valid, errors


def parse_resume(file, content):
    """Parse resume content into structured data with enhanced domain awareness"""
    # Extract standard information
    name = extract_name(content)
    email = extract_email(content)
    phone = extract_phone(content)
    skills, domain_skills = extract_skills(content)
    education = extract_education(content)
    experience_years = extract_experience(content)

    # Extract job positions and their date ranges
    positions = extract_positions(content)

    # Calculate domain-specific relevance scores
    domains = ["cybersecurity", "data_science", "web_development", "mobile_development"]
    domain_scores = {}
    domain_indicators = {}

    for domain in domains:
        # Get domain score
        score = extract_domain_specific_experience(content, domain)
        domain_scores[domain] = score

        # Identify specific indicators that contributed to the score
        if score > 0:
            # Extract key phrases for this domain (simplified version)
            indicators = []

            # Check for domain in position titles
            for position in positions:
                position_title = position["title"].lower()
                if domain.replace("_", " ") in position_title:
                    indicators.append(f"Position: {position['title']}")

            # Check for domain-specific projects
            projects_pattern = (
                r"(?:Project|Projects)(?:.*?)(?:"
                + domain.replace("_", " ")
                + r"|"
                + "|".join(domain_skills.get(domain, []))
                + r")(?:.*?)(?:\n|$)"
            )
            projects = re.findall(projects_pattern, content, re.IGNORECASE | re.DOTALL)
            if projects:
                for p in projects[:2]:  # Limit to first 2 matches
                    indicators.append(f"Project: {p.strip()[:50]}...")

            # Add domain-specific skills
            if domain in domain_skills and domain_skills[domain]:
                indicators.append(f"Skills: {', '.join(domain_skills[domain][:5])}")

            domain_indicators[domain] = indicators

    # Enhanced contextual extraction
    context = {
        "has_recent_experience": any(
            "2023" in pos["date_range"] or "2024" in pos["date_range"]
            for pos in positions
        ),
        "position_count": len(positions),
        "experience_details": positions,
        "skill_count": len(skills),
        "domain_relevance": domain_scores,
        "domain_indicators": domain_indicators,
        "education_level": "Unknown",
    }

    # Determine highest education level
    if any("phd" in edu.lower() or "doctorate" in edu.lower() for edu in education):
        context["education_level"] = "PhD"
    elif any("master" in edu.lower() for edu in education):
        context["education_level"] = "Master"
    elif any(
        "bachelor" in edu.lower() or "b.tech" in edu.lower() or "b.e." in edu.lower()
        for edu in education
    ):
        context["education_level"] = "Bachelor"

    # Validate experience format
    is_valid_format, format_errors = validate_experience_format(content)

    parsed_data = {
        "filename": file.name,
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "domain_skills": domain_skills,
        "education": education,
        "experience_years": experience_years,
        "positions": positions,
        "context": context,
        "domain_scores": domain_scores,
        "is_valid_format": is_valid_format,
        "format_errors": format_errors,
        "content": (
            content[:1000] + "..." if len(content) > 1000 else content
        ),  # Truncate content for display
    }

    return parsed_data


def get_text_chunks(text):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, api_key):
    """Create vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("resume_index")
    return vector_store


def convert_numpy_types(obj):
    """Convert NumPy types to standard Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj


def get_domain_specific_prompt(query, domain):
    """
    Generate domain-specific prompts based on the query and detected domain.

    Args:
        query (str): The user's search query
        domain (str): The detected domain ('cybersecurity', 'data_science', etc.)

    Returns:
        str: A domain-specific prompt to enhance the main prompt
    """
    # Default empty prompt
    domain_prompt = ""

    # Domain-specific prompts
    if domain == "cybersecurity":
        domain_prompt = """
        For cybersecurity candidates, pay special attention to:
        - Security certifications (CISSP, CEH, Security+, OSCP, etc.)
        - Experience with security tools (Wireshark, Nmap, Burp Suite, etc.)
        - Knowledge of security concepts (encryption, network security, threat detection)
        - Hands-on experience with penetration testing, forensics, or incident response
        - Understanding of security frameworks and compliance (NIST, ISO 27001, GDPR, etc.)
        """
    elif domain == "data_science":
        domain_prompt = """
        For data science candidates, focus on:
        - Experience with ML frameworks and libraries (TensorFlow, PyTorch, scikit-learn)
        - Statistical knowledge and mathematical foundations
        - Data manipulation and visualization skills (Pandas, NumPy, Matplotlib)
        - Project experience with predictive modeling, classification, clustering
        - Domain expertise in relevant industries (finance, healthcare, retail, etc.)
        """
    elif domain == "web_development":
        domain_prompt = """
        For web development candidates, evaluate:
        - Frontend skills (HTML, CSS, JavaScript, React, Angular, Vue)
        - Backend experience (Node.js, Django, Flask, Spring, etc.)
        - Database knowledge (SQL, MongoDB, PostgreSQL)
        - API development and integration experience
        - Web performance optimization and security practices
        """
    elif domain == "mobile_development":
        domain_prompt = """
        For mobile development candidates, consider:
        - Platform-specific experience (iOS/Swift, Android/Kotlin/Java)
        - Cross-platform framework knowledge (React Native, Flutter)
        - App lifecycle management and deployment experience
        - UI/UX design principles for mobile
        - Integration with device features (camera, location, notifications)
        """

    return domain_prompt


def extract_cgpa(text):
    """Extract CGPA/GPA from resume text"""
    # Pattern for CGPA/GPA with various formats
    cgpa_patterns = [
        r"(?:CGPA|GPA)[\s:]+(\d+\.?\d*)",
        r"(?:CGPA|GPA)[\s:-]+(\d+\.?\d*)/\d+\.?\d*",
        r"(?:with|scored|secured|obtained)[\s]+(?:a\s+)?(?:CGPA|GPA)[\s:]+(\d+\.?\d*)",
    ]

    for pattern in cgpa_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                # Convert to float and ensure it's in 0-10 range
                cgpa = float(matches[0])
                if cgpa > 10:  # Convert from 100-point scale
                    cgpa = cgpa / 10
                return cgpa
            except ValueError:
                continue
    return None


def parse_query_conditions(query):
    """Enhanced query parsing with improved XOR detection"""
    conditions = {
        "negative": [],
        "and_conditions": [],
        "or_conditions": [],
        "xor_conditions": [],
        "experience_ranges": [],
        "cgpa_condition": None,  # Initialize with None
        "percentile_threshold": 75,
    }

    # Normalize query by handling "no experience" cases
    query_lower = query.lower()
    query_lower = query_lower.replace("no experience in", "without")
    query_lower = query_lower.replace("no experience with", "without")
    query_lower = query_lower.replace("no", "without")

    # Enhanced XOR patterns to catch more variations
    xor_patterns = [
        r"(?:either\s+)?(\w+)\s+or\s+(\w+)\s+(?:but\s+)?not\s+both",
        r"(\w+)\s+xor\s+(\w+)",
        r"(?:either\s+)?(\w+)\s+or\s+(\w+)\s+exclusively",
        r"(?:only\s+)?(?:one\s+of\s+)?(\w+)\s+or\s+(\w+)",
    ]

    query_lower = query.lower()
    # Remove noise words that might interfere with pattern matching
    query_lower = query_lower.replace("experience with", "")
    query_lower = query_lower.replace("experience in", "")
    query_lower = query_lower.replace("experienced in", "")

    for pattern in xor_patterns:
        matches = re.findall(pattern, query_lower)
        if matches:
            terms = matches[0]
            # Clean up the terms
            terms = [term.strip() for term in terms]
            conditions["xor_conditions"].extend(terms)
            break  # Use only the first matching XOR condition

    # Extract experience ranges
    range_patterns = [
        r"(\d+)(?:-|\s+to\s+)(\d+)\s*(?:years?|yrs?)",
        r"between\s+(\d+)\s+and\s+(\d+)\s*(?:years?|yrs?)",
    ]

    for pattern in range_patterns:
        matches = re.findall(pattern, query_lower)
        for start, end in matches:
            conditions["experience_ranges"].append((float(start), float(end)))

    # Handle excluding experience ranges
    exclude_patterns = [
        r"excluding\s+(\d+)(?:-|\s+to\s+)(\d+)\s*(?:years?|yrs?)",
        r"except\s+(\d+)(?:-|\s+to\s+)(\d+)\s*(?:years?|yrs?)",
    ]

    for pattern in exclude_patterns:
        matches = re.findall(pattern, query_lower)
        for start, end in matches:
            conditions["experience_ranges"].append(
                (-float(end), -float(start))
            )  # Negative ranges indicate exclusion

    # Check for negative conditions
    no_patterns = [
        r"no\s+(\w+)\s+experience",
        r"without\s+(\w+)",
        r"not\s+having\s+(\w+)",
    ]
    for pattern in no_patterns:
        matches = re.findall(pattern, query_lower)
        conditions["negative"].extend(matches)

    # Check for AND conditions
    if "and" in query_lower or "both" in query_lower:
        and_pattern = r"(?:both\s+)?(\w+)\s+and\s+(\w+)"
        matches = re.findall(and_pattern, query_lower)
        for match in matches:
            conditions["and_conditions"].extend(match)

    # Check for OR conditions
    if "or" in query_lower:
        or_pattern = r"(\w+)\s+or\s+(\w+)"
        matches = re.findall(or_pattern, query_lower)
        for match in matches:
            conditions["or_conditions"].extend(match)

    # Check for CGPA conditions
    cgpa_pattern = r"(?:cgpa|gpa)\s*(>=|>|<=|<|=)\s*(\d+\.?\d*)"
    cgpa_matches = re.findall(cgpa_pattern, query_lower)
    if cgpa_matches:
        operator, value = cgpa_matches[0]
        conditions["cgpa_condition"] = (operator, float(value))

    # Check for percentile threshold
    threshold_pattern = r"top\s+(\d+)(?:%|\s*percent)"
    threshold_matches = re.findall(threshold_pattern, query_lower)
    if threshold_matches:
        conditions["percentile_threshold"] = int(threshold_matches[0])

    return conditions


def filter_resumes_by_conditions(parsed_resumes, conditions):
    """Enhanced resume filtering with strict XOR handling"""
    filtered_resumes = []

    for resume in parsed_resumes:
        matches_conditions = True
        content_lower = resume["content"].lower()
        skills_lower = [s.lower() for s in resume.get("skills", [])]

        # Handle XOR conditions with strict checking
        if conditions["xor_conditions"]:
            term1, term2 = conditions["xor_conditions"]
            has_term1 = (
                any(term1.lower() in skill for skill in skills_lower)
                or term1.lower() in content_lower
            )
            has_term2 = (
                any(term2.lower() in skill for skill in skills_lower)
                or term2.lower() in content_lower
            )

            # True XOR: must have exactly one but not both
            if not (has_term1 != has_term2):  # XOR operation
                matches_conditions = False
                continue  # Skip this resume if it doesn't meet XOR condition

        # Handle experience ranges
        if conditions["experience_ranges"] and resume["experience_years"]:
            exp_years = resume["experience_years"]
            in_range = False

            for start, end in conditions["experience_ranges"]:
                if start < 0:  # Exclusion range
                    if -end <= exp_years <= -start:
                        matches_conditions = False
                        break
                else:  # Inclusion range
                    if start <= exp_years <= end:
                        in_range = True
                        break

            if conditions["experience_ranges"] and not in_range:
                matches_conditions = False

        # Check negative conditions
        for neg_term in conditions["negative"]:
            if neg_term.lower() in content_lower or neg_term.lower() in skills_lower:
                matches_conditions = False
                break

        # Check AND conditions
        if conditions["and_conditions"]:
            matches_and = all(
                term.lower() in content_lower or term.lower() in skills_lower
                for term in conditions["and_conditions"]
            )
            matches_conditions = matches_conditions and matches_and

        # Check OR conditions
        if conditions["or_conditions"]:
            matches_or = any(
                term.lower() in content_lower or term.lower() in skills_lower
                for term in conditions["or_conditions"]
            )
            matches_conditions = matches_conditions and matches_or

        # Check CGPA condition - only if it exists
        if conditions.get("cgpa_condition"):  # Use .get() to safely access
            cgpa = extract_cgpa(resume["content"])
            if cgpa:
                operator, value = conditions["cgpa_condition"]
                matches_cgpa = {
                    ">=": cgpa >= value,
                    ">": cgpa > value,
                    "<=": cgpa <= value,
                    "<": cgpa < value,
                    "=": cgpa == value,
                }[operator]
                matches_conditions = matches_conditions and matches_cgpa
            else:
                matches_conditions = False

        if matches_conditions:
            filtered_resumes.append(resume)

    return filtered_resumes


def apply_percentile_threshold(results, threshold):
    """Apply percentile threshold to results"""
    if not results:
        return results

    # Calculate scores
    scores = [r["match_score"] for r in results]
    threshold_score = np.percentile(scores, threshold)

    # Filter results above threshold
    filtered_results = [r for r in results if r["match_score"] >= threshold_score]

    return filtered_results


def get_gemini_response(api_key, query, parsed_resumes, num_results=5, filters=None):
    """Get response with handling of badly formatted resumes"""
    # Separate valid and invalid resumes
    valid_resumes = []
    invalid_resumes = []

    for resume in parsed_resumes:
        if resume.get("is_valid_format", True):
            valid_resumes.append(resume)
        else:
            invalid_resumes.append(
                {
                    "name": resume.get("name", "Unknown"),
                    "filename": resume.get("filename", ""),
                    "format_errors": resume.get(
                        "format_errors", ["Unknown formatting error"]
                    ),
                    "content": resume.get("content", ""),
                }
            )

    # Process only valid resumes for search
    filtered_resumes = filter_resumes_by_conditions(
        valid_resumes, parse_query_conditions(query)
    )

    # Detect if query contains domain-specific keywords
    detected_domain = None
    domain_keywords = {
        "cybersecurity": ["security", "cyber", "hacking", "firewall", "vulnerability"],
        "data_science": [
            "data science",
            "machine learning",
            "ai",
            "analytics",
            "statistics",
        ],
        "web_development": ["web", "frontend", "backend", "full stack", "javascript"],
        "mobile_development": ["mobile", "android", "ios", "app", "flutter"],
    }

    # Check if any domain keywords are in the query
    query_lower = query.lower()
    for domain, keywords in domain_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_domain = domain
            break

    # Apply logical filters first
    filtered_resumes = filter_resumes_by_conditions(
        parsed_resumes, parse_query_conditions(query)
    )

    # If no resumes match the logical conditions, return empty results
    if not filtered_resumes:
        return {
            "summary": "No candidates match the specified criteria.",
            "results": [],
            "query": query,
        }

    # Continue with existing similarity search on filtered resumes
    # Create embeddings for the query
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )

    # Prepare query for each resume
    all_chunks = []
    resume_ids = []

    for i, resume in enumerate(filtered_resumes):
        chunks = get_text_chunks(resume["content"])
        all_chunks.extend(chunks)
        resume_ids.extend([i] * len(chunks))

    # If no resumes match filters, return empty results
    if not all_chunks:
        return {"results": [], "query": query}

    # Create vector store with all chunks
    vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)

    # Search for relevant chunks
    search_results = vector_store.similarity_search_with_score(
        query, k=min(10, len(all_chunks))
    )

    # Group results by resume
    resume_scores = {}
    for doc, score in search_results:
        doc_index = all_chunks.index(doc.page_content)
        resume_index = resume_ids[doc_index]

        if resume_index not in resume_scores:
            resume_scores[resume_index] = {"score": score, "highlights": []}

        # Add highlight (truncate to 200 chars around match)
        context = doc.page_content
        if len(context) > 200:
            start = max(0, len(context) // 2 - 100)
            context = "..." + context[start : start + 200] + "..."

        resume_scores[resume_index]["highlights"].append(context)

    # Sort resumes by score and take top N
    sorted_resumes = sorted(resume_scores.items(), key=lambda x: x[1]["score"])
    top_resumes = sorted_resumes[:num_results]

    # Prepare results
    results = []
    for idx, score_data in top_resumes:
        resume = filtered_resumes[idx]

        # Calculate different scoring components
        # 1. Similarity score (from vector search) - normalized to 0-100
        similarity_score = 100 - min(100, score_data["score"] * 100)

        # 2. Skills match score
        query_terms = set(query.lower().split())
        skills = set(s.lower() for s in resume["skills"])
        skills_match = (
            len(query_terms.intersection(skills)) / len(query_terms) * 100
            if query_terms
            else 0
        )

        # 3. Experience relevance score
        exp_score = (
            min(100, resume["experience_years"] * 10)
            if resume["experience_years"]
            else 0
        )

        # 4. Education score
        education_score = 0
        education_texts = [edu.lower() for edu in resume["education"]]
        if any("phd" in edu or "doctorate" in edu for edu in education_texts):
            education_score = 100
        elif any("master" in edu for edu in education_texts):
            education_score = 85
        elif any(
            "bachelor" in edu or "b.tech" in edu or "b.e." in edu
            for edu in education_texts
        ):
            education_score = 70

        # 5. Position/role match score
        positions = " ".join([p["title"].lower() for p in resume.get("positions", [])])
        position_match = any(term in positions for term in query_terms) * 100

        # Calculate weighted final score
        weights = {
            "similarity": 0.3,
            "skills": 0.25,
            "experience": 0.2,
            "education": 0.15,
            "position": 0.1,
        }

        adjusted_score = (
            similarity_score * weights["similarity"]
            + skills_match * weights["skills"]
            + exp_score * weights["experience"]
            + education_score * weights["education"]
            + position_match * weights["position"]
        )

        # Ensure score is between 0 and 100
        final_score = max(0, min(100, adjusted_score))

        result = {
            "id": str(uuid.uuid4()),
            "name": resume["name"],
            "filename": resume["filename"],
            "email": resume["email"],
            "phone": resume["phone"],
            "skills": resume["skills"],
            "education": resume["education"],
            "experience_years": resume["experience_years"],
            "match_score": final_score,
            "score_components": {
                "similarity": similarity_score,
                "skills_match": skills_match,
                "experience": exp_score,
                "education": education_score,
                "position_match": position_match,
            },
        }
        results.append(result)

    # Re-sort based on adjusted score
    results.sort(key=lambda x: x["match_score"], reverse=True)

    # Get domain-specific prompt if available
    domain_prompt = ""
    if detected_domain:
        domain_prompt = get_domain_specific_prompt(query, detected_domain)

    # Prepare a more contextual prompt based on detected domain
    prompt_template = """
    I need you to analyze these candidate profiles based on the query: "{query}"
    
    {domain_prompt}
    
    Here are the top candidates I found:
    {candidates}
    
    Please provide a detailed, insightful analysis of why these candidates match or don't match the query. 
    Be straight-forward and honest in your assessment - if a candidate has clear qualifications for the role, highlight them.
    If they lack qualifications, state this directly rather than trying to find positives.
    
    Consider the following aspects in your analysis:
    1. The relevance of their skills and experience to the query
    2. Their experience duration and recency
    3. The quality and depth of their relevant experience
    4. Domain-specific projects, certifications, and accomplishments
    5. Overall suitability for the role specified in the query
    
    Be natural and conversational in your explanation, as if you're a senior technical recruiter discussing these candidates.
    
    Return your analysis in the following JSON format:
    {{
        "summary": "Your detailed explanation of the candidates' suitability",
        "key_insights": [
            "A key insight about the candidate pool",
            "Another important observation"
        ],
        "results": {results_json},
        "query": "{query}"
    }}
    
    Make sure your output is valid JSON.
    """

    # Create embeddings for the query
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )

    # Prepare query for each resume
    all_chunks = []
    resume_ids = []

    for i, resume in enumerate(filtered_resumes):
        chunks = get_text_chunks(resume["content"])
        all_chunks.extend(chunks)
        resume_ids.extend([i] * len(chunks))

    # If no resumes match filters, return empty results
    if not all_chunks:
        return {"results": [], "query": query}

    # Create vector store with all chunks
    vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)

    # Search for relevant chunks
    search_results = vector_store.similarity_search_with_score(
        query, k=min(10, len(all_chunks))
    )

    # Group results by resume
    resume_scores = {}
    for doc, score in search_results:
        doc_index = all_chunks.index(doc.page_content)
        resume_index = resume_ids[doc_index]

        if resume_index not in resume_scores:
            resume_scores[resume_index] = {"score": score, "highlights": []}

        # Add highlight (truncate to 200 chars around match)
        context = doc.page_content
        if len(context) > 200:
            start = max(0, len(context) // 2 - 100)
            context = "..." + context[start : start + 200] + "..."

        resume_scores[resume_index]["highlights"].append(context)

    # Sort resumes by score and take top N
    sorted_resumes = sorted(resume_scores.items(), key=lambda x: x[1]["score"])
    top_resumes = sorted_resumes[:num_results]

    # Prepare results
    results = []
    for idx, score_data in top_resumes:
        resume = filtered_resumes[idx]
        result = {
            "id": str(uuid.uuid4()),  # Generate a unique ID
            "name": resume["name"],
            "filename": resume["filename"],
            "email": resume["email"],
            "phone": resume["phone"],
            "skills": resume["skills"],
            "education": resume["education"],
            "experience_years": resume["experience_years"],
            "match_score": float(
                (1 - score_data["score"]) * 100
            ),  # Convert to percentage (higher is better)
            "highlights": score_data["highlights"],
        }
        results.append(result)

    # Use Gemini to generate a summary
    prompt_template = """
    I need you to analyze these candidate profiles based on the query: "{query}"
    
    Here are the top candidates I found:
    {candidates}
    
    Please provide a detailed, insightful analysis of why these candidates match the query. 
    Consider the following aspects in your analysis:
    1. The relevance of their skills to the query
    2. Their experience duration and recency
    3. The quality and depth of their experience
    4. Their education background
    5. Any special accomplishments or projects that stand out
    
    Be natural and conversational in your explanation, as if you're a senior recruiter discussing these candidates.
    
    Return your analysis in the following JSON format:
    {{
        "summary": "Your detailed explanation of why these candidates are good matches",
        "key_insights": [
            "A key insight about the candidate pool",
            "Another important observation"
        ],
        "results": {results_json},
        "query": "{query}"
    }}
    
    Make sure your output is valid JSON.
    """

    candidates_text = "\n\n".join(
        [
            f"Candidate: {r['name']}\n"
            f"Skills: {', '.join(r['skills'])}\n"
            f"Domain-Specific Skills: {', '.join([skill for domain_skills in r.get('domain_skills', {}).values() for skill in domain_skills])}\n"
            f"Experience: {r['experience_years']} years\n"
            f"Positions: {'; '.join([p['title'] + ' (' + p['date_range'] + ')' for p in r.get('positions', [])])}\n"
            f"Education: {'; '.join(r['education'])}\n"
            f"Match Score: {r['match_score']:.2f}%\n"
            f"Domain Relevance: {r.get('domain_relevance', 0):.2f}%\n"
            f"Key Domain Indicators: {'; '.join([indicator for domain, indicators in r.get('context', {}).get('domain_indicators', {}).items() for indicator in indicators[:3]])}\n"
            f"Highlights from Resume:\n"
            + "\n".join([f"- {h}" for h in r["highlights"][:3]])
            for r in results
        ]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key
    )

    # Convert NumPy types to standard Python types for JSON serialization
    converted_results = convert_numpy_types(results)

    prompt = prompt_template.format(
        query=query,
        candidates=candidates_text,
        results_json=json.dumps(converted_results),
    )

    response = model.invoke(prompt)

    try:
        # Try to parse the response as JSON
        response_text = response.content
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result_json = json.loads(json_str)
            return result_json
        else:
            # Fallback if the model didn't return valid JSON
            return {
                "summary": "Analyzed candidates based on your query.",
                "results": results,
                "query": query,
            }
    except Exception as e:
        # Fallback on parsing errors
        return {
            "summary": "Found matching candidates based on your query.",
            "results": results,
            "query": query,
            "badly_formatted_resumes": {
                "count": len(invalid_resumes),
                "resumes": invalid_resumes,
            },
        }


import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pygame import mixer
import threading
import queue
import time
import tempfile
import os
import sounddevice as sd
import soundfile as sf

import base64
from PIL import Image, ImageDraw, ImageFont
import io


def encode_image(image_path):
    """Convert image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_resume_with_vision(resume_content, candidate_info):
    """Analyze resume using Gemini Vision"""
    # Convert text content to image
    image = Image.new("RGB", (800, 1000), color="white")
    d = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = None
    d.text((20, 20), resume_content, fill="black", font=font)

    # Save temp image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Create vision prompt
    vision_prompt = f"""Analyze this candidate's resume:
    Name: {candidate_info['name']}
    Match Score: {candidate_info['match_score']:.2f}%
    
    Provide a detailed analysis with the following sections:
    1. Executive Summary
    2. Technical Skills Assessment
    3. Experience Analysis
    4. Education & Certifications
    5. Interview Focus Areas
    6. Final Recommendation
    """

    try:
        # Use Gemini Vision
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([vision_prompt, img_byte_arr])
        return response.text
    except Exception as e:
        return f"Error analyzing resume: {str(e)}"


def play_audio_response(text, max_duration=30):
    """Convert text to speech and play with duration limit"""
    try:
        # Split long text into chunks of ~500 characters
        chunks = [text[i : i + 500] for i in range(0, len(text), 500)]

        for chunk in chunks:
            # Create temp audio file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                audio_file = temp_file.name
                tts = gTTS(text=chunk, lang="en")
                tts.save(audio_file)

                # Initialize pygame mixer if needed
                if not mixer.get_init():
                    mixer.init()

                # Load and play audio
                mixer.music.load(audio_file)
                mixer.music.play()

                # Wait for playback with timeout
                start_time = time.time()
                while mixer.music.get_busy():
                    if time.time() - start_time > max_duration:
                        mixer.music.stop()
                        break
                    time.sleep(0.1)

                # Cleanup
                mixer.music.unload()
                try:
                    os.remove(audio_file)
                except Exception:
                    pass

        mixer.quit()

    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")


def voice_chat_interface():
    st.title("Voice Resume Chat")

    # Resume selector
    if "parsed_resumes" in st.session_state and st.session_state.parsed_resumes:
        resume_options = [
            f"{r['name']} - {r['filename']}" for r in st.session_state.parsed_resumes
        ]
        resume_options.insert(0, "General Chat (No Resume)")

        col1, col2 = st.columns([1, 2])

        with col1:
            selected_idx = st.selectbox(
                "Select Resume:",
                range(len(resume_options)),
                format_func=lambda x: resume_options[x],
            )

            if st.button("🎤 Ask Question"):
                with st.spinner("Listening..."):
                    try:
                        # Record and transcribe
                        _ = record_audio(duration=7)
                        user_text = speech_to_text()

                        if user_text:
                            st.text(f"You asked: {user_text}")

                            # Get resume context if selected
                            if selected_idx > 0:
                                resume = st.session_state.parsed_resumes[
                                    selected_idx - 1
                                ]
                                context = f"""
                                Analyzing resume for: {resume['name']}
                                Experience: {resume['experience_years']} years
                                Skills: {', '.join(resume['skills'])}
                                Education: {'; '.join(resume['education'])}
                                Query: {user_text}
                                """
                            else:
                                context = user_text

                            # Generate response
                            model = genai.GenerativeModel("gemini-1.5-pro")
                            response = model.generate_content(context)
                            ai_response = response.text

                            # Store response and play audio
                            st.session_state.last_response = ai_response
                            play_audio_response(ai_response)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        with col2:
            if selected_idx > 0:
                resume = st.session_state.parsed_resumes[selected_idx - 1]
                st.subheader("Resume Preview")
                st.write(f"**Name:** {resume['name']}")
                st.write(f"**Experience:** {resume['experience_years']} years")
                st.write("**Skills:**", ", ".join(resume["skills"]))
                st.write("**Education:**")
                for edu in resume["education"]:
                    st.write(f"- {edu}")

                with st.expander("View Full Resume"):
                    st.text(resume["content"])

    else:
        st.warning("Please upload and process resumes in the Resume Search tab first.")

    # Display last response
    if "last_response" in st.session_state:
        st.markdown("---")
        st.subheader("AI Response:")
        st.write(st.session_state.last_response)


def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    recording = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, recording, sample_rate)
        return f.name


def transcribe_audio(audio_file):
    """Transcribe audio using Groq Whisper"""
    with open(audio_file, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file, f.read()),
            model="llama-3.3-70b-versatile",
            response_format="text",
            language="en",
        )

    # Handle different possible response formats
    if isinstance(transcription, str):
        return transcription
    elif hasattr(transcription, "text"):
        return transcription.text
    else:
        return str(transcription)  # Fallback to string representation


def voice_assistant_analysis(api_key, result_data):
    """Analyze candidate with voice assistant"""
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config=genai.GenerationConfig(
            candidate_count=1,
            top_p=0.7,
            top_k=4,
            max_output_tokens=2000,
            temperature=0.7,
        ),
    )

    initial_prompt = f"""
    You are an AI HR assistant helping evaluate candidates. Please analyze this candidate profile:
    
    Name: {result_data['name']}
    Match Score: {result_data['match_score']:.2f}%
    Experience: {result_data['experience_years']} years
    Skills: {', '.join(result_data['skills'])}
    Education: {'; '.join(result_data['education'])}
    
    Provide insights in a natural conversational way. Focus on:
    1. Overall match for the role
    2. Key strengths and weaknesses
    3. Areas to probe during interview
    4. Hiring recommendation
    """

    chat = model.start_chat(history=[])
    return chat, initial_prompt


def text_to_speech(text, lang="en"):
    """Convert text to speech"""
    mp3file = BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(mp3file)
    mp3file.seek(0)

    # Save to temp file since Streamlit's audio needs a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(mp3file.read())
        return f.name


def speech_to_text():
    """Convert speech to text using Google Speech Recognition"""
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Please speak your question.")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=5)
            st.success("Got it! Processing your question...")

            text = r.recognize_google(audio)
            return text
    except sr.WaitTimeoutError:
        st.error("No speech detected. Please try again.")
    except sr.UnknownValueError:
        st.error("Could not understand the audio. Please try again.")
    except sr.RequestError:
        st.error(
            "Could not reach the speech recognition service. Please check your internet connection."
        )
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return None


def get_voice_response(text, context=None, max_tokens=250):
    """Get response from Gemini model with length limit"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Limit response length
        prompt = f"Please provide a brief, helpful response in under {max_tokens} words: {text}"
        if context:
            prompt = f"""
            Context: {context}
            Question: {text}
            Please provide a brief answer considering the context in under {max_tokens} words.
            Stay focused on relevant, factual information.
            """

        # Generate response with basic error handling
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Handle specific API errors
            error_msg = str(e).lower()
            if "unsafe" in error_msg or "dangerous" in error_msg:
                return "I cannot provide a response to that query. Please try asking something else."
            else:
                return f"Error generating response: {str(e)}"

    except Exception as e:
        return "Sorry, I encountered an error. Please try again."


def voice_chat_interface():
    """Updated voice chat interface with improved response handling"""
    st.title("Voice Chat Interface")

    # Initialize session state for conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Resume selector for context
    context = None
    if "parsed_resumes" in st.session_state and st.session_state.parsed_resumes:
        resume_options = [
            f"{r['name']} - {r['filename']}" for r in st.session_state.parsed_resumes
        ]
        resume_options.insert(0, "General Chat (No Resume)")

        col1, col2 = st.columns([1, 2])

        with col1:
            selected_idx = st.selectbox(
                "Select Resume for Context:",
                range(len(resume_options)),
                format_func=lambda x: resume_options[x],
            )

            if selected_idx > 0:
                resume = st.session_state.parsed_resumes[selected_idx - 1]
                context = f"""
                Name: {resume['name']}
                Experience: {resume['experience_years']} years
                Skills: {', '.join(resume['skills'])}
                Education: {'; '.join(resume['education'])}
                """

                with st.expander("Selected Resume Context"):
                    st.info(context)

            # Voice input button
            if st.button("🎤 Start Voice Input"):
                user_text = speech_to_text()

                if user_text:
                    st.write("You said:", user_text)

                    with st.spinner("Getting response..."):
                        try:
                            # Get AI response with timeout
                            response = get_voice_response(user_text, context)

                            # Add to conversation history
                            st.session_state.conversation_history.append(
                                {"user": user_text, "assistant": response}
                            )

                            # Display text response immediately
                            st.markdown("**AI Response:**")
                            st.write(response)

                            # Play audio in background thread
                            threading.Thread(
                                target=play_audio_response,
                                args=(response,),
                                daemon=True,
                            ).start()

                        except Exception as e:
                            st.error(f"Error processing response: {str(e)}")

        with col2:
            # Display conversation history
            st.subheader("Conversation History")
            for exchange in st.session_state.conversation_history:
                st.write("👤 You:", exchange["user"])
                st.write("🤖 Assistant:", exchange["assistant"])
                st.markdown("---")

            # Clear history button
            if st.button("Clear History"):
                st.session_state.conversation_history = []
                st.experimental_rerun()

    else:
        st.warning(
            "Please upload and process resumes in the Resume Search tab first to enable resume context."
        )


def get_available_models():
    """Get list of available models for vision analysis"""
    return {
        "Gemini Pro Vision": "gemini-pro-vision",
        "LLaMA-2 Vision": "llama-3.2-90b-vision-preview",
        "GPT-4 Vision": "gpt-4-vision-preview",
    }


def analyze_resume_with_vision(resume_content, candidate_info, selected_model):
    """Analyze resume using selected vision model"""
    # Convert text content to image
    image = Image.new("RGB", (800, 1000), color="white")
    d = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = None
    d.text((20, 20), resume_content, fill="black", font=font)

    # Save temp image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    base64_image = base64.b64encode(img_byte_arr).decode("utf-8")

    # Create vision prompt
    vision_prompt = f"""Analyze this candidate's resume:
    Name: {candidate_info['name']}
    Match Score: {candidate_info['match_score']:.2f}%
    
    Please provide a detailed analysis focusing on:
    1. Key qualifications and skills
    2. Experience relevance
    3. Education background
    4. Areas to probe in interview
    5. Hiring recommendation
    6. Visual layout and presentation
    """

    try:
        if "gemini" in selected_model:
            # Use Gemini
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(selected_model)
            response = model.generate_content([vision_prompt, img_byte_arr])
            return response.text
        else:
            # Use Groq
            completion = groq_client.chat.completions.create(
                model=selected_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            return completion.choices[0].message.content
    except Exception as e:
        return f"Error analyzing resume: {str(e)}"


def main():
    st.set_page_config(page_title="Resume Search System", page_icon="📄", layout="wide")

    # Initialize all session state variables
    if "api_key" not in st.session_state:
        st.session_state.api_key = GEMINI_API_KEY
    if "parsed_resumes" not in st.session_state:
        st.session_state.parsed_resumes = []
    if "shortlisted" not in st.session_state:
        st.session_state.shortlisted = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "shortlisted_data" not in st.session_state:
        st.session_state.shortlisted_data = None
    if "interview_questions" not in st.session_state:
        st.session_state.interview_questions = None

    # Create tabs for different functionalities
    tabs = st.tabs(["Resume Search", "Mock Interview Questions", "Voice Chat"])

    with tabs[0]:  # Resume Search Tab
        st.title("Resume Search System with RAG")

        # Initialize session state for storing resumes
        if "parsed_resumes" not in st.session_state:
            st.session_state.parsed_resumes = []

        # Sidebar with resume upload (API key input removed)
        with st.sidebar:
            st.header("Upload Resumes")
            resume_files = st.file_uploader(
                "Upload resume files (PDF, DOCX, or TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
            )

            # Process uploaded resumes
            if st.button("Process Resumes"):
                if not resume_files:
                    st.error("Please upload at least one resume file")
                else:
                    with st.spinner("Processing resumes..."):
                        # Clear previous resumes
                        st.session_state.parsed_resumes = []

                        # Process each resume
                        for file in resume_files:
                            try:
                                # Extract text from resume
                                content = extract_resume_text(file)

                                # Parse resume
                                parsed_data = parse_resume(file, content)

                                # Add to session state
                                st.session_state.parsed_resumes.append(parsed_data)
                            except Exception as e:
                                st.error(f"Error processing {file.name}: {str(e)}")

                        st.success(
                            f"Successfully processed {len(st.session_state.parsed_resumes)} resumes"
                        )

            # Show total processed resumes
            st.write(f"Total processed resumes: {len(st.session_state.parsed_resumes)}")

            # Clear all button
            if st.button("Clear All Data"):
                st.session_state.parsed_resumes = []
                st.session_state.shortlisted = []
                st.session_state.search_results = None
                st.success("All data cleared")

        # Main area with search interface
        st.header("Search Resumes")

        # Search query input
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., candidates experienced in Python with machine learning skills",
        )

        # Search parameters
        col1, col2, col3 = st.columns(3)

        with col1:
            skills_filter = st.text_input(
                "Required Skills (comma-separated):",
                placeholder="e.g., python, react, aws",
            )

            min_experience = st.number_input(
                "Minimum Years of Experience:", min_value=0, value=0, step=1
            )

            education_filter = st.selectbox(
                "Education Level:", options=["Any", "Bachelor", "Master", "PhD"]
            )

        # Additional contextual filters
        with st.expander("Advanced Filters"):
            recent_experience = st.checkbox(
                "Only show candidates with recent experience (2023-2024)"
            )
            st.markdown("---")
            st.write("**Position Keywords** (match candidates with specific positions)")
            position_keywords = st.text_input(
                "Position contains:", placeholder="e.g., developer, intern, manager"
            )
            st.markdown("---")
            st.write("**Project Keywords** (match candidates with specific projects)")
            project_keywords = st.text_input(
                "Project keywords:",
                placeholder="e.g., machine learning, web app, mobile",
            )

        # Number of results to show
        num_results = st.slider(
            "Number of results to show:", min_value=1, max_value=20, value=5
        )

        # Search button
        if st.button("Search Resumes"):
            if not st.session_state.parsed_resumes:
                st.error("No resumes found. Please upload and process resumes first.")
            elif not query:
                st.error("Please enter a search query")
            else:
                with st.spinner("Searching resumes..."):
                    # Prepare filters with enhanced context awareness
                    filters = {}

                    if skills_filter:
                        filters["skills"] = [
                            s.strip() for s in skills_filter.split(",")
                        ]

                    if min_experience > 0:
                        filters["min_experience"] = min_experience

                    if education_filter != "Any":
                        filters["education"] = education_filter

                    # Add contextual filters
                    if recent_experience:
                        filters["recent_experience"] = True

                    if position_keywords:
                        position_terms = [
                            term.strip().lower()
                            for term in position_keywords.split(",")
                        ]

                        # Apply position filter directly in the search query
                        if position_terms:
                            # Enhance the query with position terms
                            position_terms_text = " AND ".join(position_terms)
                            query = f"{query} with positions like {position_terms_text}"

                    if project_keywords:
                        project_terms = [
                            term.strip().lower() for term in project_keywords.split(",")
                        ]

                        # Apply project filter directly in the search query
                        if project_terms:
                            # Enhance the query with project terms
                            project_terms_text = " AND ".join(project_terms)
                            query = (
                                f"{query} with projects related to {project_terms_text}"
                            )

                    # Get results from Gemini
                    results = get_gemini_response(
                        api_key=st.session_state.api_key,
                        query=query,
                        parsed_resumes=st.session_state.parsed_resumes,
                        num_results=num_results,
                        filters=filters,
                    )

                    # Store results in session state
                    st.session_state.search_results = results

        # Display search results
        if st.session_state.search_results:
            st.header("Search Results")
            results = st.session_state.search_results

            # Display overall summary if available
            if "summary" in results:
                with st.expander("Analysis Summary", expanded=True):
                    st.markdown(results["summary"])
                    if "key_insights" in results:
                        st.subheader("Key Insights")
                        for insight in results["key_insights"]:
                            st.markdown(f"• {insight}")

            # Create dropdown for resume selection
            resume_options = [
                f"{r['name']} - {r['filename']} (Match: {r['match_score']:.2f}%)"
                for r in results["results"]
            ]
            selected_resume = st.selectbox(
                "Select a resume to analyze:", resume_options
            )

            # Get selected resume data
            selected_idx = resume_options.index(selected_resume)
            result = results["results"][selected_idx]

            # Create three columns for better layout
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Basic Information")
                st.write(f"**Name:** {result['name']}")
                st.write(f"**Email:** {result['email']}")
                st.write(f"**Phone:** {result['phone']}")
                st.write(f"**Experience:** {result['experience_years']} years")

                # Display skills with categories
                st.write("**Skills:**")
                if "domain_skills" in result:
                    for domain, skills in result["domain_skills"].items():
                        with st.expander(f"{domain.replace('_', ' ').title()} Skills"):
                            st.write(", ".join(skills))
                st.write("**General Skills:**")
                st.write(", ".join(result["skills"]))

                st.write("**Education:**")
                for edu in result["education"]:
                    st.write(f"- {edu}")

            with col2:
                st.subheader("Analysis & Highlights")

                # Display match score with color
                score = result["match_score"]
                score_color = (
                    "green" if score >= 80 else "orange" if score >= 60 else "red"
                )
                st.markdown(
                    f"**Match Score:** <span style='color:{score_color}'>{score:.1f}%</span>",
                    unsafe_allow_html=True,
                )

                # Display highlights
                if "highlights" in result:
                    st.write("**Key Highlights:**")
                    for highlight in result["highlights"]:
                        st.markdown(f"- {highlight}")

                # Display positions if available
                if "positions" in result and result["positions"]:
                    st.write("**Work History:**")
                    for pos in result["positions"]:
                        st.markdown(f"- **{pos['title']}** ({pos['date_range']})")

                # Add detailed analysis button
                if st.button(
                    "Generate Detailed Analysis", key=f"analyze_{result['id']}"
                ):
                    with st.spinner("Analyzing resume..."):
                        # Get comprehensive analysis
                        analysis = analyze_resume_with_vision(
                            result.get("content", ""), result  # Use get() with fallback
                        )

                        if analysis:
                            with st.expander("Detailed Analysis", expanded=True):
                                sections = analysis.split("\n\n")
                                for section in sections:
                                    if section.strip():
                                        title = (
                                            section.split("\n")[0]
                                            if ":" in section
                                            else "Analysis"
                                        )
                                        content = (
                                            "\n".join(section.split("\n")[1:])
                                            if ":" in section
                                            else section
                                        )
                                        st.markdown(f"**{title}**")
                                        st.markdown(content)
                                        st.markdown("---")

            # Add to shortlist button
            if st.button("Add to Shortlist", key=f"shortlist_{result['id']}"):
                if result not in st.session_state.shortlisted:
                    st.session_state.shortlisted.append(result)
                    st.success(f"Added {result['name']} to shortlist!")
                else:
                    st.warning("This candidate is already shortlisted!")

        # Display shortlisted candidates
        if st.session_state.shortlisted:
            st.header("Shortlisted Candidates")

            # Display count
            st.write(f"Total shortlisted: {len(st.session_state.shortlisted)}")

            # Display each shortlisted candidate
            for candidate in st.session_state.shortlisted:
                st.write(f"**{candidate['name']}** - {candidate['filename']}")
                st.write(f"Email: {candidate['email']} | Phone: {candidate['phone']}")

            # Export as CSV
            if st.button("Export Shortlist as CSV"):
                # Create DataFrame
                df = pd.DataFrame(
                    [
                        {
                            "Name": c["name"],
                            "Email": c["email"],
                            "Phone": c["phone"],
                            "Experience": c["experience_years"],
                            "Skills": ", ".join(c["skills"]),
                            "Education": "; ".join(c["education"]),
                            "Match Score": f"{c['match_score']:.2f}%",
                            "Filename": c["filename"],
                        }
                        for c in st.session_state.shortlisted
                    ]
                )

                # Convert to CSV
                csv = df.to_csv(index=False)

                # Create download link
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="shortlisted.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

    with tabs[1]:  # Mock Interview Questions Tab
        st.title("Generate Mock Interview Questions")
        st.write(
            "Upload the shortlisted candidates CSV file to generate targeted interview questions."
        )

        # Initialize session state for interview tab
        if "shortlisted_data" not in st.session_state:
            st.session_state.shortlisted_data = None

        if "interview_questions" not in st.session_state:
            st.session_state.interview_questions = None

        # File uploader for shortlisted.csv
        uploaded_file = st.file_uploader(
            "Upload shortlisted.csv", type=["csv"], key="interview_uploader"
        )

        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                st.session_state.shortlisted_data = df
                st.success(f"Successfully loaded {len(df)} shortlisted candidates")

                # Display the uploaded data
                st.subheader("Shortlisted Candidates")
                st.dataframe(df)

                # Question type selection
                st.subheader("Generate Interview Questions")
                question_type = st.selectbox(
                    "Select question type:",
                    [
                        "Concept Questions",
                        "Coding/LeetCode Questions",
                        "Project-Based Questions",
                    ],
                )

                # Options based on question type
                if question_type == "Concept Questions":
                    # Extract languages from skills in CSV
                    all_skills = []
                    if "Skills" in df.columns:
                        for skills in df["Skills"].dropna():
                            if isinstance(skills, str):
                                all_skills.extend(
                                    [s.strip().lower() for s in skills.split(",")]
                                )

                    # Filter programming languages
                    programming_languages = [
                        lang
                        for lang in [
                            "python",
                            "java",
                            "javascript",
                            "c++",
                            "c#",
                            "typescript",
                            "ruby",
                            "php",
                            "go",
                            "swift",
                            "kotlin",
                            "rust",
                            "scala",
                            "dart",
                            "r",
                        ]
                        if lang in set(all_skills)
                    ]

                    # Add "Other" option
                    programming_languages.append("Other")

                    # Language selection
                    language = st.selectbox(
                        "Select programming language:", programming_languages
                    )

                    if language == "Other":
                        language = st.text_input("Specify language:")

                    # Difficulty selection
                    difficulty = st.select_slider(
                        "Select difficulty level:",
                        options=["Basic", "Intermediate", "Advanced"],
                    )

                    # Number of questions
                    num_questions = st.slider(
                        "Number of questions:", min_value=1, max_value=5, value=3
                    )

                elif question_type == "Coding/LeetCode Questions":
                    # Difficulty selection
                    difficulty = st.select_slider(
                        "Select difficulty level:", options=["Easy", "Medium", "Hard"]
                    )

                    # Topic selection
                    topics = st.multiselect(
                        "Select topics (optional):",
                        [
                            "Arrays",
                            "Strings",
                            "Linked Lists",
                            "Trees",
                            "Graphs",
                            "Dynamic Programming",
                            "Recursion",
                            "Sorting",
                            "Searching",
                            "Hash Tables",
                            "Stacks",
                            "Queues",
                        ],
                    )

                    # Number of questions
                    num_questions = st.slider(
                        "Number of questions:", min_value=1, max_value=5, value=3
                    )

                elif question_type == "Project-Based Questions":
                    # Project focus
                    st.info(
                        "Questions will be generated based on projects mentioned in the shortlisted resumes."
                    )

                    # Number of questions per project
                    questions_per_project = st.slider(
                        "Questions per project:", min_value=1, max_value=5, value=2
                    )

                # Generate button
                if st.button("Generate Questions"):
                    with st.spinner("Generating interview questions..."):
                        if (
                            "api_key" not in st.session_state
                            or not st.session_state.api_key
                        ):
                            st.error(
                                "Please enter your Google AI API key in the Resume Search tab first."
                            )
                        else:
                            api_key = st.session_state.api_key

                            # Generate questions based on type
                            if question_type == "Concept Questions":
                                questions = generate_concept_questions(
                                    api_key,
                                    language,
                                    difficulty,
                                    num_questions,
                                    st.session_state.shortlisted_data,
                                )
                            elif question_type == "Coding/LeetCode Questions":
                                questions = generate_coding_questions(
                                    api_key, difficulty, topics, num_questions
                                )
                            elif question_type == "Project-Based Questions":
                                questions = generate_project_questions(
                                    api_key,
                                    st.session_state.shortlisted_data,
                                    questions_per_project,
                                )

                            st.session_state.interview_questions = questions

                # Display generated questions
                if st.session_state.interview_questions:
                    st.subheader("Generated Interview Questions")

                    if question_type == "Concept Questions":
                        for i, q in enumerate(st.session_state.interview_questions):
                            with st.expander(f"Question {i+1}: {q['question']}..."):
                                st.markdown(f"**{q['question']}**")
                                st.markdown("#### Expected Answer:")
                                st.markdown(q["answer"])
                                if "follow_up" in q:
                                    st.markdown("#### Follow-up Question:")
                                    st.markdown(q["follow_up"])

                    elif question_type == "Coding/LeetCode Questions":
                        for i, q in enumerate(st.session_state.interview_questions):
                            with st.expander(f"Question {i+1}: {q['title']}"):
                                st.markdown(f"**Problem Title:** {q['title']}")
                                st.markdown(f"**Difficulty:** {q['difficulty']}")
                                st.markdown(f"**Link:** [{q['platform']}]({q['link']})")
                                st.markdown("#### Problem Description:")
                                st.markdown(q["description"])
                                if "hints" in q:
                                    st.markdown("#### Hints:")
                                    for hint in q["hints"]:
                                        st.markdown(f"- {hint}")

                    elif question_type == "Project-Based Questions":
                        for (
                            project,
                            questions,
                        ) in st.session_state.interview_questions.items():
                            with st.expander(f"Project: {project}"):
                                for i, q in enumerate(questions):
                                    st.markdown(f"**Q{i+1}: {q['question']}**")
                                    st.markdown("#### Ideal Response:")
                                    st.markdown(q["ideal_response"])
                                    if "follow_up" in q:
                                        st.markdown("#### Follow-up Question:")
                                        st.markdown(q["follow_up"])

                    # Export questions button
                    if st.button("Export Questions to CSV"):
                        csv_data = convert_questions_to_csv(
                            st.session_state.interview_questions, question_type
                        )
                        st.download_button(
                            "Download Questions CSV",
                            csv_data,
                            "interview_questions.csv",
                            "text/csv",
                            key="download-questions-csv",
                        )

            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")

    with tabs[2]:  # Voice Chat Tab
        voice_chat_interface()


if __name__ == "__main__":
    main()
