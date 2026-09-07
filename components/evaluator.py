import json
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any

try:
    from components.interview_engine import _is_low_effort_answer
except Exception:
    def _is_low_effort_answer(text: str) -> bool:
        return not text or len(text.strip().split()) < 6

def generate_performance_eval(chat_history: List[Dict[str, str]], role: str, difficulty: str, client=None) -> Dict[str, Any]:
    """
    Evaluates the completed interview history and generates a comprehensive scorecard report.
    """
    if client:
        try:
            from google.genai import types
            transcript = ""
            for msg in chat_history:
                r = "Interviewer" if msg["role"] == "interviewer" else "Candidate"
                transcript += f"{r}: {msg['text']}\n"
                
            prompt = f"""
            You are a strict, honest hiring bar raiser evaluating an interview candidate for a {difficulty} {role} position.
            Do not inflate scores out of politeness. Empty, filler, greeting-only, or off-topic responses
            (e.g. "hello", "hey hey hello hi", or a response that never actually addresses the question)
            must be scored low on every relevant dimension and must NOT be treated as if they were real answers.
            If most or all of the candidate's responses were filler/empty, the overall recommendation must be
            "No Hire" regardless of how the interviewer's follow-up questions read.
            
            Interview Transcript:
            \"\"\"
            {transcript[:6000]}
            \"\"\"
            
            Evaluate the candidate on a 0-100 scale across 5 criteria:
            1. Technical Depth & Accuracy
            2. Problem Solving & Logic
            3. Communication & Clarity (STAR framework)
            4. Efficiency & Edge Cases
            5. Confidence & Structure
            
            Also generate:
            - Overall Score (0-100)
            - Key Strengths (3 bullet points)
            - Primary Areas for Improvement (3 bullet points)
            - Detailed Recommendation (Hire / Strong Hire / Weak Pass / No Hire)
            - Question-by-Question breakdown with Model Answer advice.
            
            Return STRICT JSON:
            {{
              "overall_score": 85,
              "recommendation": "Strong Hire",
              "scores": {{
                "Technical Depth": 85,
                "Problem Solving": 90,
                "Communication": 80,
                "Efficiency": 82,
                "Confidence & Structure": 88
              }},
              "strengths": [
                "Strong grasp of time complexity and space trade-offs",
                "Articulate explanation of edge cases",
                "Quick adaptation to follow-up hints"
              ],
              "improvements": [
                "Structure answers more explicitly using the STAR method for behavioral questions",
                "Elaborate more on error handling and fault tolerance",
                "Quantify past project impact with specific metrics"
              ],
              "detailed_summary": "The candidate demonstrated solid engineering capability and strong analytical skills throughout..."
            }}
            """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception:
            pass

    # Heuristic fallback evaluation (used when no Gemini client is active,
    # or the live API call fails). This actually looks at chat_history
    # instead of returning a fixed score, so a candidate who gave filler or
    # empty answers cannot receive an inflated "Hire" result.
    candidate_msgs = [m["text"] for m in chat_history if m.get("role") == "candidate"]
    total_answers = len(candidate_msgs)
    low_effort_count = sum(1 for t in candidate_msgs if _is_low_effort_answer(t))
    substantive_count = total_answers - low_effort_count
    substantive_ratio = (substantive_count / total_answers) if total_answers else 0.0

    # Scale 20 (nothing substantive) up to 90 (everything substantive).
    # This is still a heuristic, not real grading of correctness — it can
    # only tell effort/relevance apart from filler, not verify technical
    # accuracy. That limitation is called out in the summary below.
    base_score = int(20 + substantive_ratio * 70)
    scores = {
        "Technical Depth": base_score,
        "Problem Solving": base_score,
        "Communication": base_score,
        "Efficiency": base_score,
        "Confidence & Structure": base_score,
    }

    if total_answers == 0:
        recommendation = "No Hire"
    elif base_score >= 80:
        recommendation = "Strong Hire"
    elif base_score >= 65:
        recommendation = "Hire"
    elif base_score >= 45:
        recommendation = "Weak Pass"
    else:
        recommendation = "No Hire"

    strengths = []
    improvements = []
    if substantive_count > 0:
        strengths.append(f"Provided {substantive_count} substantive response(s) engaging with the questions asked.")
    else:
        strengths.append("No substantive responses were recorded in this session.")

    if low_effort_count > 0:
        improvements.append(
            f"{low_effort_count} of {total_answers} response(s) were empty, filler, or too short to evaluate "
            "(e.g. greetings, single words, or unfilled placeholder text) — these need real technical content."
        )
    improvements.append("Note: no live AI grading was available for this session (simulation mode), so technical correctness was not verified — only response effort/relevance could be assessed.")

    summary = (
        f"Simulation-mode evaluation for a {difficulty} {role} session: {substantive_count} of {total_answers} "
        f"candidate response(s) contained substantive content; {low_effort_count} were empty, filler, or "
        "unanswered. This score reflects effort and relevance only — connect a Gemini API key for real "
        "technical accuracy grading."
    )

    return {
        "overall_score": base_score,
        "recommendation": recommendation,
        "scores": scores,
        "strengths": strengths,
        "improvements": improvements,
        "detailed_summary": summary,
    }

def create_radar_chart(scores: Dict[str, float]) -> go.Figure:
    """Creates an interactive radar chart of candidate performance scores."""
    categories = list(scores.keys())
    values = list(scores.values())
    
    # Close the radar loop
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='Candidate Score',
        fillcolor='rgba(137, 180, 250, 0.3)',
        line=dict(color='#89b4fa', width=3)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color='#cdd6f4',
                gridcolor='#45475a'
            ),
            angularaxis=dict(
                color='#cdd6f4',
                gridcolor='#45475a'
            ),
            bgcolor='#181825'
        ),
        paper_bgcolor='#11111b',
        font=dict(color='#cdd6f4', size=12),
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )
    return fig