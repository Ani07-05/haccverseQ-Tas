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
from datetime import datetime

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

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
    """Extract years of experience from resume text considering date ranges with deduplication"""
    from datetime import datetime
    import calendar

    # First check for explicit mentions of experience
    experience_pattern = (
        r"(\d+)[\+]?\s*(?:years|year|yr|yrs)(?:\s+of\s+experience|\s+experience)?"
    )
    explicit_matches = re.findall(experience_pattern, text, re.IGNORECASE)

    total_months = 0
    if explicit_matches:
        # Sum all found experience years and convert to months
        total_months = sum(int(year) * 12 for year in explicit_matches)

    # Now look for date ranges and track them to avoid double-counting
    date_ranges = []

    # Pattern for "Month Year - Month Year"
    date_range_pattern1 = r"([A-Za-z]+)\s+(\d{4})\s*[-–—]\s*([A-Za-z]+)\s+(\d{4})"
    # Pattern for "Month Year - Present"
    date_range_pattern2 = r"([A-Za-z]+)\s+(\d{4})\s*[-–—]\s*(Present|Current|Now)"
    # Pattern for single month mentions like "Mar 2025"
    date_single_pattern = r"(?:^|\s|,|\()([A-Za-z]+)\s+(\d{4})(?:\s|$|,|\))"

    # Helper function to convert month name to number
    def month_to_num(month_name):
        month_name = month_name.strip().capitalize()
        if len(month_name) >= 3:
            # Try full month name
            try:
                return list(calendar.month_name).index(month_name)
            except ValueError:
                # Try abbreviated month name
                try:
                    for i, month in enumerate(calendar.month_name):
                        if month and month.startswith(month_name[:3]):
                            return i
                    # Direct abbreviation match
                    return list(calendar.month_abbr).index(month_name[:3])
                except ValueError:
                    return 1  # Default to January if we can't parse
        return 1  # Default to January for short month names

    # Get the current date for "Present" calculations
    current_date = datetime.now()

    # Process "Month Year - Month Year" date ranges
    for match in re.finditer(date_range_pattern1, text):
        start_month, start_year, end_month, end_year = match.groups()
        try:
            start_date = datetime(int(start_year), month_to_num(start_month), 1)
            end_date = datetime(int(end_year), month_to_num(end_month), 28)

            # Store as tuple of (start_date, end_date)
            date_ranges.append((start_date, end_date))
        except (ValueError, TypeError):
            continue

    # Process "Month Year - Present" date ranges
    for match in re.finditer(date_range_pattern2, text):
        start_month, start_year, _ = match.groups()
        try:
            start_date = datetime(int(start_year), month_to_num(start_month), 1)

            # Store as tuple of (start_date, current_date)
            date_ranges.append((start_date, current_date))
        except (ValueError, TypeError):
            continue

    # Process single month mentions (assume 1 month duration)
    for match in re.finditer(date_single_pattern, text):
        month, year = match.groups()
        try:
            date = datetime(int(year), month_to_num(month), 15)  # Middle of month

            # Store as tuple of (date, date + 1 month)
            end_date = datetime(
                date.year + (date.month // 12), ((date.month % 12) + 1) or 12, 15
            )
            date_ranges.append((date, end_date))
        except (ValueError, TypeError):
            continue

    # If we found date ranges, calculate non-overlapping experience
    if date_ranges:
        # Sort by start date
        date_ranges.sort(key=lambda x: x[0])

        # Merge overlapping date ranges
        merged_ranges = []
        for current_range in date_ranges:
            if not merged_ranges:
                merged_ranges.append(current_range)
                continue

            last_start, last_end = merged_ranges[-1]
            current_start, current_end = current_range

            # Check if current range overlaps with the last merged range
            if current_start <= last_end:
                # Merge by extending the end date if needed
                merged_ranges[-1] = (last_start, max(last_end, current_end))
            else:
                # No overlap, add as new range
                merged_ranges.append(current_range)

        # Calculate total months from merged ranges
        for start_date, end_date in merged_ranges:
            months_diff = (
                (end_date.year - start_date.year) * 12
                + end_date.month
                - start_date.month
            )
            total_months += max(1, months_diff)  # Ensure at least 1 month per entry

    # Convert total months to years with 1 decimal place
    total_years = round(total_months / 12, 1)

    return total_years


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
    Generate {num_questions} {difficulty.lower()} {language} programming concept interview questions.
    
    Consider that candidates also have these skills: {', '.join(relevant_skills[:5])}
    
    For each question, provide:
    1. A detailed question that tests deep understanding, not just facts
    2. An expert-level answer that demonstrates mastery
    3. A thoughtful follow-up question to probe deeper
    
    Return the results as a JSON array where each item has 'question', 'answer', and 'follow_up' keys.
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
    """Parse query to identify logical conditions"""
    conditions = {
        "negative": [],
        "and_conditions": [],
        "or_conditions": [],
        "cgpa_condition": None,
        "percentile_threshold": 75  # Default percentile threshold
    }
    
    # Convert query to lowercase for easier matching
    query_lower = query.lower()
    
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
    """Filter resumes based on parsed conditions"""
    filtered_resumes = []
    
    for resume in parsed_resumes:
        matches_conditions = True
        content_lower = resume["content"].lower()
        skills_lower = [s.lower() for s in resume.get("skills", [])]
        
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
        
        # Check CGPA condition
        if conditions["cgpa_condition"]:
            cgpa = extract_cgpa(resume["content"])
            if cgpa:
                operator, value = conditions["cgpa_condition"]
                matches_cgpa = {
                    ">=": cgpa >= value,
                    ">": cgpa > value,
                    "<=": cgpa <= value,
                    "<": cgpa < value,
                    "=": cgpa == value
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
    filtered_results = [
        r for r in results 
        if r["match_score"] >= threshold_score
    ]
    
    return filtered_results

def get_gemini_response(api_key, query, parsed_resumes, num_results=5, filters=None):
    """Get response from Gemini model with enhanced logical conditions"""
    # Parse query conditions
    conditions = parse_query_conditions(query)
    
    # Detect if query contains domain-specific keywords
    detected_domain = None
    domain_keywords = {
        "cybersecurity": ["security", "cyber", "hacking", "firewall", "vulnerability"],
        "data_science": ["data science", "machine learning", "ai", "analytics", "statistics"],
        "web_development": ["web", "frontend", "backend", "full stack", "javascript"],
        "mobile_development": ["mobile", "android", "ios", "app", "flutter"]
    }
    
    # Check if any domain keywords are in the query
    query_lower = query.lower()
    for domain, keywords in domain_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_domain = domain
            break
    
    # Apply logical filters first
    filtered_resumes = filter_resumes_by_conditions(parsed_resumes, conditions)
    
    # If no resumes match the logical conditions, return empty results
    if not filtered_resumes:
        return {
            "summary": "No candidates match the specified criteria.",
            "results": [],
            "query": query
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

        # Calculate an adjusted match score that combines similarity and domain relevance
        similarity_score = (
            1 - score_data["score"]
        ) * 100  # Convert to percentage (higher is better)
        domain_score = resume.get("domain_relevance", 0) * 100 if detected_domain else 0

        # Weighted combination (70% similarity, 30% domain relevance)
        adjusted_score = (0.7 * similarity_score) + (0.3 * domain_score)

        result = {
            "id": str(uuid.uuid4()),  # Generate a unique ID
            "name": resume["name"],
            "filename": resume["filename"],
            "email": resume["email"],
            "phone": resume["phone"],
            "skills": resume["skills"],
            "domain_skills": resume.get("domain_skills", {}),
            "education": resume["education"],
            "experience_years": resume["experience_years"],
            "positions": resume.get("positions", []),
            "match_score": float(adjusted_score),
            "domain_relevance": float(domain_score),
            "highlights": score_data["highlights"],
            "context": resume.get("context", {}),
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
        }


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
    tabs = st.tabs(["Resume Search", "Mock Interview Questions"])

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

        with col2:
            min_experience = st.number_input(
                "Minimum Experience (years):", min_value=0, value=0
            )

        with col3:
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

            # Display summary and key insights if available
            if "summary" in results:
                st.subheader("Summary Analysis")
                st.write(results["summary"])

            if (
                "key_insights" in results
                and results["key_insights"]
                and results["key_insights"]
            ):
                st.subheader("Key Insights")
                for insight in results["key_insights"]:
                    st.markdown(f"• {insight}")

            st.subheader(f"Found {len(results['results'])} matching candidates")

            # Shortlist interface
            st.write("Select candidates to shortlist:")

            # Create checkboxes for each result
            selected_ids = []

            for result in results["results"]:
                col1, col2 = st.columns([1, 10])

                with col1:
                    if st.checkbox("", key=f"select_{result['id']}"):
                        selected_ids.append(result["id"])

                with col2:
                    with st.expander(
                        f"{result['name']} - {result['filename']} - Match: {result['match_score']:.2f}%"
                    ):
                        st.write(f"**Email:** {result['email']}")
                        st.write(f"**Phone:** {result['phone']}")
                        st.write(f"**Experience:** {result['experience_years']} years")

                        # Display positions if available
                        if "positions" in result and result["positions"]:
                            st.write("**Positions:**")
                            for position in result["positions"]:
                                st.write(
                                    f"- {position['title']} ({position['date_range']})"
                                )

                        st.write("**Skills:**")
                        st.write(", ".join(result["skills"]))

                        st.write("**Education:**")
                        for edu in result["education"]:
                            st.write(f"- {edu}")

                        # Show additional context if available
                        if "context" in result:
                            st.write("**Additional Insights:**")
                            context = result["context"]
                            if context.get("has_recent_experience"):
                                st.write("- ✅ Has recent experience (2023-2024)")
                            if context.get("education_level") != "Unknown":
                                st.write(
                                    f"- 🎓 Highest education: {context.get('education_level')}"
                                )
                            if context.get("position_count", 0) > 1:
                                st.write(
                                    f"- 👔 Has {context.get('position_count')} different positions"
                                )

                        st.write("**Highlights:**")
                        for highlight in result["highlights"]:
                            st.markdown(f"- _{highlight}_")

            # Shortlist button
            if st.button("Shortlist Selected Candidates"):
                if not selected_ids:
                    st.error("Please select at least one candidate to shortlist")
                else:
                    # Get selected candidates
                    shortlisted = []

                    for result in results["results"]:
                        if result["id"] in selected_ids:
                            shortlisted.append(result)

                    # Store in session state
                    st.session_state.shortlisted = shortlisted

                    st.success(f"Shortlisted {len(shortlisted)} candidates")

            # Display JSON output button
            if st.button("Export Results as JSON"):
                # Create JSON output
                json_output = json.dumps(results, indent=2)

                # Create download link
                b64 = base64.b64encode(json_output.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="search_results.json">Download JSON</a>'
                st.markdown(href, unsafe_allow_html=True)

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
                            with st.expander(
                                f"Question {i+1}: {q['question'][:100]}..."
                            ):
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


if __name__ == "__main__":
    main()
