import io
from pypdf import PdfReader
from typing import Dict, Any, List

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts raw text from PDF bytes using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        extracted_pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_pages.append(text)
        return "\n".join(extracted_pages)
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def extract_skills_and_summary(resume_text: str, client=None) -> Dict[str, Any]:
    """
    Analyzes resume text using Gemini API or rule-based heuristics.
    Returns extracted skills, target role estimate, past projects/experience, and suggested interview questions.
    """
    if client:
        try:
            prompt = f"""
            Analyze the following resume text and extract key structured information:
            1. Identified technical/core skills (list)
            2. Past job titles & approximate experience level (Junior, Mid, Senior, Staff/Lead)
            3. Key projects & achievements
            4. 4 Tailored interview questions (mixture of technical depth and project-specific questions)

            Resume Text:
            \"\"\"
            {resume_text[:4000]}
            \"\"\"

            Respond in clean JSON format:
            {{
               "skills": ["Skill1", "Skill2", ...],
               "experience_level": "Senior",
               "past_roles": ["Role 1", "Role 2"],
               "key_projects": ["Project 1 summary", "Project 2 summary"],
               "tailored_questions": [
                   "Question 1",
                   "Question 2",
                   "Question 3",
                   "Question 4"
               ]
            }}
            Return ONLY the JSON string.
            """
            from google.genai import types
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            import json
            data = json.loads(response.text)
            return data
        except Exception:
            pass

    # Heuristic fallback if API key isn't active or fails
    skills_keywords = [
        "Python", "JavaScript", "TypeScript", "React", "Node.js", "Java", "C++", "SQL",
        "Docker", "Kubernetes", "AWS", "GCP", "Machine Learning", "PyTorch", "TensorFlow",
        "System Design", "GraphQL", "REST API", "Microservices", "Git", "CI/CD", "Pandas"
    ]
    found_skills = [skill for skill in skills_keywords if skill.lower() in resume_text.lower()]
    if not found_skills:
        found_skills = ["Software Engineering", "Problem Solving", "System Architecture", "API Design"]

    # Basic experience detection
    experience_level = "Mid-Level"
    if "senior" in resume_text.lower() or "lead" in resume_text.lower() or "principal" in resume_text.lower():
        experience_level = "Senior / Lead"
    elif "intern" in resume_text.lower() or "junior" in resume_text.lower():
        experience_level = "Junior / Entry-Level"

    tailored_questions = [
        f"I noticed experience with {found_skills[0]}. Can you describe how you used it to solve a complex architectural challenge?",
        f"Walk me through the design of one of your recent major projects mentioned on your resume.",
        f"How do you handle performance bottlenecks when scaling applications using {found_skills[min(1, len(found_skills)-1)]}?",
        "Tell me about a time when you had to trade off code quality for speed of delivery."
    ]

    return {
        "skills": found_skills,
        "experience_level": experience_level,
        "past_roles": ["Software Engineer / Developer"],
        "key_projects": ["Engineered scalable backend/frontend applications", "Optimized database queries and API pipelines"],
        "tailored_questions": tailored_questions
    }
