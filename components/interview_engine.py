import json
import re
import time
from typing import Dict, List, Any, Optional

# Words that carry no technical/behavioral content on their own. Used to catch
# filler-only answers ("hey hey hello hi") that shouldn't be graded as if
# they were substantive responses.
_FILLER_WORDS = {
    "hey", "hi", "hello", "um", "uh", "yeah", "yep", "yes", "no", "ok", "okay",
    "test", "testing", "idk", "dunno", "hmm", "well", "so", "like", "sure",
    "the", "a", "an", "is", "it", "and", "or", "but"
}


def _is_low_effort_answer(text: str) -> bool:
    """Heuristically flags empty, filler-only, or too-short answers.

    This is a deterministic guard that runs regardless of whether the Gemini
    client is configured, so a greeting or empty submission can never be
    scored as a genuine attempt.
    """
    if not text or not text.strip():
        return True
    if "TRANSCRIPTION_UNAVAILABLE" in text or "Recorded audio received" in text:
        return True
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    words = cleaned.split()
    if len(words) < 6:
        return True
    non_filler = set(words) - _FILLER_WORDS
    if len(non_filler) < 3:
        return True
    return False

PERSONA_PROMPTS = {
    "Software Engineering": """You are a Principal Software Engineer conducting a technical interview. 
Evaluate algorithmic thinking, data structures, code clarity, edge cases, and Big-O efficiency.
Be professional, encouraging, but probe deeply into technical details.""",
    
    "System Design Architect": """You are a Staff Infrastructure Architect conducting a System Design interview.
Evaluate scalability, API design, database choices, caching, load balancing, and fault tolerance.""",

    "Data Science & ML": """You are a Lead AI Research Scientist evaluating machine learning fundamentals, feature engineering, model evaluation, and deployment trade-offs.""",

    "Product Manager": """You are a Senior Director of Product Management evaluating product intuition, user empathy, prioritization frameworks (RICE/MOSCOW), and execution metrics.""",

    "Behavioral (STAR Method)": """You are a Senior Engineering Manager assessing leadership, teamwork, conflict resolution, and handling failures via the STAR method (Situation, Task, Action, Result)."""
}

STARTER_QUESTIONS = {
    "Software Engineering": [
        "Welcome! Let's start with a core problem: How would you design an algorithm to find the median of two sorted arrays in O(log(min(n, m))) time?",
        "Hello! To begin, how would you detect and break a cycle in a singly linked list? Compare Floyd's Cycle-Finding Algorithm with a Hash Set approach.",
        "Welcome! Can you explain how you would construct a Trie (Prefix Tree) and write methods for insert, search, and startsWith?"
    ],
    "System Design Architect": [
        "Welcome! Let's dive into system architecture: How would you design a real-time collaborative document editor like Google Docs?",
        "Hello! How would you architect a global notification system capable of sending millions of push notifications per minute with high reliability?",
        "Welcome! Walk me through the design of a rate limiter service. What algorithms and data stores would you choose?"
    ],
    "Data Science & ML": [
        "Welcome! How would you handle severe class imbalance in a fraud detection dataset? Compare resampling, focal loss, and evaluation metrics.",
        "Hello! Explain how Transformer self-attention mechanisms work compared to traditional RNN/LSTM architectures.",
        "Welcome! How would you design an A/B testing framework to evaluate a new recommendation algorithm in a high-traffic app?"
    ],
    "Product Manager": [
        "Welcome! How would you design a feature to improve user retention for a music streaming platform?",
        "Hello! Imagine you are the PM for a ride-sharing app. How would you handle a sudden 15% drop in rider bookings in a major city?",
        "Welcome! How do you prioritize feature requests when engineering capacity is limited and stakeholders have conflicting priorities?"
    ],
    "Behavioral (STAR Method)": [
        "Welcome! Tell me about a time when you were leading a project and encountered an unexpected technical roadblock or tight deadline change.",
        "Hello! Describe a situation where you had to persuade stakeholders or team members to adopt an unpopular technical decision.",
        "Welcome! Give me an example of a mistake or failure you experienced in a past role. What did you learn and how did you adjust your process?"
    ]
}

def _parse_json_response(raw_text: str) -> dict:
    """Parses a JSON response from the model, tolerating markdown code fences
    (```json ... ```) that some models add even when response_mime_type is
    set to application/json."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

class InterviewSession:
    def __init__(self, role: str = "Software Engineering", difficulty: str = "Mid-Level", client=None):
        self.role = role
        self.difficulty = difficulty
        self.client = client
        self.chat_history: List[Dict[str, str]] = []
        self.question_count = 0
        self.max_questions = 4
        self.is_completed = False
        self.start_timestamp = time.time()
        self.question_start_time = time.time()
        self.response_times: List[float] = []
        self.last_error: Optional[str] = None
        
    def start_interview(self, custom_starter: Optional[str] = None) -> str:
        """Initializes the interview and returns the first question."""
        if custom_starter:
            first_q = custom_starter
        else:
            starters = STARTER_QUESTIONS.get(self.role, STARTER_QUESTIONS["Software Engineering"])
            import random
            first_q = random.choice(starters)
        
        greeting = f"**Interviewer**: {first_q}\n\n*(Level: {self.difficulty} | Persona: {self.role})*"
        self.chat_history.append({"role": "interviewer", "text": greeting})
        self.question_count += 1
        self.question_start_time = time.time()
        return greeting

    def get_elapsed_minutes(self) -> float:
        """Returns total elapsed session duration in minutes."""
        return round((time.time() - self.start_timestamp) / 60.0, 1)

    def process_candidate_answer(self, candidate_answer: str, action_override: Optional[str] = None) -> Dict[str, Any]:
        """Processes candidate response or quick action request and generates interviewer reply."""
        elapsed_ans = round(time.time() - self.question_start_time, 1)
        self.response_times.append(elapsed_ans)
        
        user_text = candidate_answer
        if action_override:
            user_text = f"[{action_override}] {candidate_answer}"
            
        self.chat_history.append({"role": "candidate", "text": user_text})

        # Deterministic guard: catch filler/empty/non-answers BEFORE any grading
        # path runs, so they can never be scored as a genuine attempt (by the
        # AI or by the simulated fallback).
        if not action_override and _is_low_effort_answer(candidate_answer):
            last_question = self._last_interviewer_text()
            reply_text = (
                "**Interviewer**: That doesn't actually answer the question — I need a real response "
                "covering your approach, not a greeting or filler text. Let's try again:\n\n"
                f"{last_question}"
            )
            self.chat_history.append({"role": "interviewer", "text": reply_text})
            return {"reply": reply_text, "is_completed": self.is_completed}

        if self.client:
            try:
                from google.genai import types
                system_instruction = PERSONA_PROMPTS.get(self.role, PERSONA_PROMPTS["Software Engineering"])
                
                prompt = f"""
                {system_instruction}
                
                Difficulty Level: {self.difficulty}
                Current Question Number: {self.question_count} of {self.max_questions}
                Action Requested by Candidate: {action_override if action_override else 'Normal Answer Submission'}
                
                Conversation History so far:
                {self._formatted_history()}
                
                Candidate's latest response:
                "{candidate_answer}"
                
                Task:
                1. First, judge whether the answer is actually correct and relevant to the question asked.
                   Be an honest, rigorous evaluator, not an agreeable one — do not praise an answer unless
                   it genuinely demonstrates correct understanding. Do not soften or hide inaccuracies.
                2. If the answer is wrong, vague, off-topic, or missing key points, say so explicitly and
                   name the specific gap or error. Do not respond with generic encouragement ("Solid response!",
                   "Great job!", etc.) unless the content actually earns it.
                3. If candidate asked to "Challenge My Complexity", push them hard on Big-O optimization.
                4. If candidate asked for "Edge Cases", present a tricky edge case scenario.
                5. Otherwise, transition to the next question or follow-up — but only after giving honest
                   feedback on the current one per points 1-2.
                6. If question limit reached ({self.max_questions}), conclude interview nicely.
                
                Respond in valid JSON format:
                {{
                  "inline_feedback": "Honest, specific feedback — call out errors or gaps by name if present...",
                  "interviewer_reply": "Interviewer statement and next question or conclusion...",
                  "answer_was_correct": true,
                  "is_final": false
                }}
                """
                
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                data = _parse_json_response(response.text)
                
                reply_text = data.get("interviewer_reply", "Thank you. Let's move on.")
                inline_fb = data.get("inline_feedback", "")
                
                full_interviewer_response = f"{inline_fb}\n\n**Interviewer**: {reply_text}" if inline_fb else reply_text
                self.chat_history.append({"role": "interviewer", "text": full_interviewer_response})
                self.question_count += 1
                self.question_start_time = time.time()
                self.last_error = None
                
                if self.question_count > self.max_questions or data.get("is_final", False):
                    self.is_completed = True
                    
                return {
                    "reply": full_interviewer_response,
                    "is_completed": self.is_completed
                }
            except Exception as e:
                # Surface the real reason instead of silently falling back —
                # check session.last_error in the UI to see what actually
                # went wrong (bad model name, invalid key, quota, malformed
                # response, etc).
                self.last_error = f"{type(e).__name__}: {e}"

        # Fallback simulated response
        self.question_count += 1
        self.question_start_time = time.time()
        
        if action_override == "Challenge My Complexity":
            reply_text = f"**Interviewer**: Good initiative! Your current solution is functional, but can you optimize it to reduction from O(n^2) to O(n) or O(log n) using extra memory or a two-pointer technique?"
        elif action_override == "Ask Edge Case Scenario":
            reply_text = f"**Interviewer**: Let's test boundary conditions: How does your code behave if the input array is empty, contains duplicate elements, or exceeds 10^7 elements in memory?"
        elif self.question_count <= self.max_questions:
            reply_text = (
                f"**Interviewer**: Noted — I don't have live AI grading active right now (simulation mode), "
                f"so I can't verify the technical accuracy of that answer. Moving to question {self.question_count}: "
                f"How do you approach error handling and scale trade-offs in this design?"
            )
        else:
            reply_text = "**Interviewer**: Excellent session throughout! That completes our mock interview. Click 'Generate Scorecard' to view your performance analytics."
            self.is_completed = True
            
        self.chat_history.append({"role": "interviewer", "text": reply_text})
        return {
            "reply": reply_text,
            "is_completed": self.is_completed
        }

    def generate_hint(self, current_question: str) -> str:
        """Generates an AI hint for the current active question."""
        if self.client:
            try:
                from google.genai import types
                prompt = f"Provide a subtle, guiding hint (without giving away full code) for this interview question: '{current_question}'"
                res = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                self.last_error = None
                return res.text
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
        return "💡 **Hint**: Break the problem down into smaller sub-problems. Consider using a Hash Map for O(1) lookups, Two Pointers, or Sliding Window."

    def _last_interviewer_text(self) -> str:
        """Returns the most recent interviewer message, for re-asking after a low-effort answer."""
        for item in reversed(self.chat_history):
            if item["role"] == "interviewer":
                return item["text"]
        return "Could you please share your answer to the current question?"

    def _formatted_history(self) -> str:
        formatted = []
        for item in self.chat_history:
            role = "Interviewer" if item["role"] == "interviewer" else "Candidate"
            formatted.append(f"{role}: {item['text']}")
        return "\n".join(formatted)