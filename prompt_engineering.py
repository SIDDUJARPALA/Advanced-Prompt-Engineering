"""Interactive demonstrations of nine prompt-engineering techniques with Mistral."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from mistralai import Mistral
except ImportError:  # fallback for older mistralai versions
    from mistralai.client import Mistral


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "outputs" / "results.json"
load_dotenv(BASE_DIR / ".env")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip() or "mistral-small-latest"


def call_mistral(prompt: str) -> str:
    """Send one user prompt to Mistral; return a readable error instead of crashing."""
    if not MISTRAL_API_KEY or MISTRAL_API_KEY == "your_api_key_here":
        return (
            "ERROR: MISTRAL_API_KEY is not configured. Add your key to the local "
            ".env file as MISTRAL_API_KEY=your_api_key_here, replacing the placeholder."
        )

    candidate_models = [MISTRAL_MODEL, "mistral-small-latest", "mistral-small", "open-mistral-7b"]
    seen_models: set[str] = set()

    try:
        for model_name in candidate_models:
            if not model_name or model_name in seen_models:
                continue
            seen_models.add(model_name)
            try:
                client = Mistral(api_key=MISTRAL_API_KEY)
                response = client.chat.complete(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                if not response.choices:
                    return "ERROR: Mistral returned no response choices."
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    return "ERROR: Mistral returned an empty response."
                return content.strip()
            except Exception as exc:
                last_error = exc
                error_name = type(exc).__name__.lower()
                if "model" in str(exc).lower() and "not found" in str(exc).lower():
                    continue
                if "unauthorized" in error_name or "401" in str(exc):
                    raise  # invalid API key; do not hide it
                if "not found" in str(exc).lower() or "unknown model" in str(exc).lower():
                    continue
                # Try one more format with the generic SDK call if the first attempt fails for a non-model reason.
                try:
                    client = Mistral(api_key=MISTRAL_API_KEY)
                    response = client.chat.complete(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    if not response.choices:
                        return "ERROR: Mistral returned no response choices."
                    content = response.choices[0].message.content
                    if not isinstance(content, str) or not content.strip():
                        return "ERROR: Mistral returned an empty response."
                    return content.strip()
                except Exception:
                    pass
                if model_name == candidate_models[-1]:
                    raise last_error

        return "ERROR: No valid Mistral model was available for this account."
    except Exception as exc:
        message = str(exc).replace(MISTRAL_API_KEY, "[REDACTED]")
        exception_name = type(exc).__name__.lower()
        status_code = getattr(exc, "status_code", None)

        if status_code == 401 or "unauthorized" in exception_name or "unauthorized" in message.lower():
            detail = "The Mistral API key was rejected. Check that it is valid and active."
        elif any(word in exception_name for word in ("connection", "timeout", "network")) or any(word in message.lower() for word in ("connection", "timeout", "network")):
            detail = "Could not connect to the Mistral API. Check your internet connection and retry."
        elif "rate limit" in message.lower():
            detail = "The Mistral API rate limit was reached. Please wait and retry."
        else:
            detail = f"Mistral API request failed ({type(exc).__name__})."

        if message and message != str(exc):
            detail += f" Details: {message}"
        return f"ERROR: {detail}"


def _record(
    technique: str,
    domain: str,
    prompt: str,
    response: str,
    analysis: str,
    **extra: Any,
) -> dict[str, Any]:
    record = {
        "technique": technique,
        "domain": domain,
        "prompt": prompt,
        "response": response,
        "analysis": analysis,
        **extra,
    }
    _display_record(record)
    return record


def _display_record(record: dict[str, Any]) -> None:
    print(f"\nTechnique: {record['technique']}")
    print(f"Domain: {record['domain']}")
    if "prompt" in record:
        print(f"Prompt:\n{record['prompt']}")
    if "examples" in record:
        print(f"Examples:\n{record['examples']}")
    if "final_input" in record:
        print(f"Final input: {record['final_input']}")
    if "stages" in record:
        for stage in record["stages"]:
            print(f"\n{stage['name']} prompt:\n{stage['prompt']}")
            print(f"{stage['name']} response:\n{stage['response']}")
    if "rounds" in record:
        for round_data in record["rounds"]:
            print(f"\n{round_data['name']}:\n{round_data['response']}")
            if round_data.get("feedback"):
                print(f"Feedback: {round_data['feedback']}")
    if "valid_json" in record:
        print(f"Valid JSON: {'Yes' if record['valid_json'] else 'No'}")
        if record.get("parsed_json") is not None:
            print("Parsed JSON:")
            print(json.dumps(record["parsed_json"], indent=4, ensure_ascii=False))
    elif "response" in record:
        print(f"Response:\n{record['response']}")
    print(f"Analysis: {record['analysis']}")


def _save_results() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as result_file:
        json.dump(RESULTS, result_file, indent=4, ensure_ascii=False)
    print(f"\nResults saved to: {OUTPUT_FILE.relative_to(BASE_DIR)}")


def zero_shot_prompting() -> list[dict[str, Any]]:
    """Demonstrate zero-shot prompting across three domains."""
    cases = [
        (
            "Education",
            "Explain recursion in Python to a beginner using a simple example.",
        ),
        (
            "Software Development",
            "Explain the difference between a list and tuple in Python.",
        ),
        (
            "Career Guidance",
            "Give five practical tips for preparing for a technical interview.",
        ),
    ]
    records = []
    for domain, prompt in cases:
        response = call_mistral(prompt)
        records.append(
            _record(
                "Zero-Shot Prompting",
                domain,
                prompt,
                response,
                "The model receives a task directly, without examples to imitate.",
            )
        )
    RESULTS["zero_shot"] = records
    return records


def few_shot_prompting() -> list[dict[str, Any]]:
    """Show how input-output examples establish a classification pattern."""
    examples = (
        "Input: I love this course.\nOutput: Positive\n\n"
        "Input: The explanation was confusing.\nOutput: Negative"
    )
    final_input = "The instructor explained the topic clearly."
    prompt = (
        "Classify each statement as Positive or Negative. Follow the examples.\n\n"
        f"{examples}\n\nInput: {final_input}\nOutput:"
    )
    response = call_mistral(prompt)
    record = _record(
        "Few-Shot Prompting",
        "Education",
        prompt,
        response,
        "The examples demonstrate the requested labels and output format before the new input.",
        examples=examples,
        final_input=final_input,
    )
    RESULTS["few_shot"] = [record]
    return [record]


def structured_reasoning_prompting() -> list[dict[str, Any]]:
    """Compare a direct instruction with a concise, structured solution request."""
    problem = "A Python program needs to find the second-largest number in a list."
    simple_prompt = "Write Python code to find the second-largest number in a list."
    structured_prompt = (
        f"Solve this programming problem: {problem}\n\n"
        "Provide:\n"
        "1. Approach\n"
        "2. Important considerations\n"
        "3. Python solution\n"
        "4. Final result explanation\n\n"
        "Keep the solution concise. Provide a brief solution outline, not private "
        "chain-of-thought."
    )
    simple_response = call_mistral(simple_prompt)
    structured_response = call_mistral(structured_prompt)
    analysis = (
        "The structured prompt requests labeled components and edge-case considerations, "
        "so its answer can be easier to follow and evaluate than the simple prompt. "
        "It requests a concise solution outline, not hidden chain-of-thought."
    )
    records = [
        _record(
            "Structured Reasoning (Simple Prompt)",
            "Software Development",
            simple_prompt,
            simple_response,
            "Baseline answer with no requested sections.",
        ),
        _record(
            "Structured Reasoning (Organized Prompt)",
            "Software Development",
            structured_prompt,
            structured_response,
            analysis,
        ),
    ]
    RESULTS["structured_reasoning"] = records
    return records


def role_based_prompting() -> list[dict[str, Any]]:
    """Demonstrate role-specific emphasis over education and career domains."""
    cases = [
        (
            "Software Development",
            "You are an experienced Python developer. Explain how to improve this "
            "function for readability and correctness:\n"
            "def average(values):\n    return sum(values) / len(values)",
        ),
        (
            "Software Development",
            "You are a professional data analyst. Analyze this small dataset and state "
            "one useful observation: weekly study hours = [2, 3, 3, 5, 7].",
        ),
        (
            "Education",
            "You are a technical interviewer. Create five Python interview questions "
            "for a beginner.",
        ),
        (
            "Career Guidance",
            "You are a professional career counsellor. Suggest a learning roadmap for "
            "a student who wants to become a Python developer.",
        ),
    ]
    records = []
    for domain, prompt in cases:
        records.append(
            _record(
                "Role-Based Prompting",
                domain,
                prompt,
                call_mistral(prompt),
                "The role directs tone, priorities, and the expertise the response should emphasize.",
            )
        )
    RESULTS["role_based"] = records
    return records


def iterative_prompting() -> list[dict[str, Any]]:
    """Refine a response over successive rounds, carrying each answer forward."""
    initial_prompt = "Write a short professional summary for a Python developer resume."
    initial = call_mistral(initial_prompt)
    feedback_1 = (
        "The summary is too generic. Make it more specific, concise and achievement-oriented."
    )
    prompt_2 = f"{feedback_1}\n\nPrevious summary:\n{initial}"
    improved = call_mistral(prompt_2)
    feedback_2 = (
        "Make the summary suitable for a fresher and avoid inventing work experience. "
        "Keep it concise."
    )
    prompt_3 = f"{feedback_2}\n\nPrevious summary:\n{improved}"
    final = call_mistral(prompt_3)
    records = [
        {
            "name": "Initial Response",
            "prompt": initial_prompt,
            "response": initial,
        },
        {
            "name": "Improved Response",
            "feedback": feedback_1,
            "prompt": prompt_2,
            "response": improved,
        },
        {
            "name": "Final Response",
            "feedback": feedback_2,
            "prompt": prompt_3,
            "response": final,
        },
    ]
    record = {
        "technique": "Iterative Prompting",
        "domain": "Career Guidance",
        "prompt": initial_prompt,
        "rounds": records,
        "analysis": (
            "Round 1 creates a baseline. Feedback 1 asks for specificity and impact; "
            "Feedback 2 adds fresher-appropriate, truthful constraints."
        ),
    }
    _display_record(record)
    RESULTS["iterative"] = [record]
    return [record]


def prompt_chaining() -> list[dict[str, Any]]:
    """Pass each stage's generated content as the next stage's input."""
    job_description = (
        "Junior AI developer. Required skills: Python, SQL, Git, machine learning, "
        "clear written communication, and teamwork."
    )
    stage_1_prompt = (
        "Extract the important skills from this job description as a concise list. "
        "Do not add skills that are not stated.\n\n"
        f"Job description:\n{job_description}"
    )
    stage_1 = call_mistral(stage_1_prompt)
    stage_2_prompt = (
        "Categorize only these extracted skills under Programming, Data, Tools, "
        f"and Soft Skills. Preserve the provided information:\n\n{stage_1}"
    )
    stage_2 = call_mistral(stage_2_prompt)
    stage_3_prompt = (
        "Create a practical one-week study plan based only on these categorized skills. "
        f"Use one focus per day and include a small practice activity:\n\n{stage_2}"
    )
    stage_3 = call_mistral(stage_3_prompt)
    record = {
        "technique": "Prompt Chaining",
        "domain": "Career Guidance",
        "prompt": stage_1_prompt,
        "stages": [
            {"name": "Stage 1 - Extract", "prompt": stage_1_prompt, "response": stage_1},
            {"name": "Stage 2 - Analyze", "prompt": stage_2_prompt, "response": stage_2},
            {"name": "Stage 3 - Generate", "prompt": stage_3_prompt, "response": stage_3},
        ],
        "analysis": (
            "Each prompt consumes the preceding stage's response: extracted skills are "
            "categorized, then the categories ground the study plan."
        ),
    }
    _display_record(record)
    RESULTS["prompt_chaining"] = [record]
    return [record]


def negative_prompting() -> list[dict[str, Any]]:
    """Use explicit restrictions to reduce unwanted or unsupported content."""
    prompt = (
        "Create a beginner-friendly Python learning plan.\n\n"
        "Rules:\n"
        "- Do not invent certifications.\n"
        "- Do not claim the learner has professional experience.\n"
        "- Do not include irrelevant technologies.\n"
        "- Do not use unnecessary motivational statements.\n"
        "- Keep the answer concise.\n"
        "- Use bullet points."
    )
    record = _record(
        "Negative Prompting",
        "Education",
        prompt,
        call_mistral(prompt),
        "Restrictions target fabricated credentials or experience, irrelevant content, "
        "unwanted motivational filler, and formatting drift.",
    )
    RESULTS["negative_prompting"] = [record]
    return [record]


TEMPLATE = """Explain {topic}.

Difficulty level: {difficulty}
Target audience: {audience}
Output format: {output_format}

Do not assume knowledge beyond the specified difficulty level."""


def generate_from_template(
    topic: str, difficulty: str, audience: str, output_format: str
) -> dict[str, Any]:
    """Fill a reusable prompt template and return its prompt and model response."""
    prompt = TEMPLATE.format(
        topic=topic,
        difficulty=difficulty,
        audience=audience,
        output_format=output_format,
    )
    return {"prompt": prompt, "response": call_mistral(prompt)}


def reusable_prompt_template() -> list[dict[str, Any]]:
    """Run a dynamic template with varied topics, audiences, and domains."""
    cases = [
        ("Python decorators", "Intermediate", "Computer science students", "Bullet points with code example", "Software Development"),
        ("Photosynthesis", "Beginner", "Middle school students", "Three short bullet points", "Education"),
        ("Preparing for a first technical interview", "Beginner", "Recent graduates", "A concise numbered checklist", "Career Guidance"),
    ]
    records = []
    for topic, difficulty, audience, output_format, domain in cases:
        generated = generate_from_template(topic, difficulty, audience, output_format)
        records.append(
            _record(
                "Reusable Prompt Templates",
                domain,
                generated["prompt"],
                generated["response"],
                "The same template is customized with placeholders for topic, level, audience, and format.",
            )
        )
    RESULTS["reusable_template"] = records
    return records


def json_output_prompting() -> list[dict[str, Any]]:
    """Request JSON, remove optional code fences, and validate with json.loads."""
    prompt = """Analyze the following candidate profile.

Return ONLY valid JSON using exactly this structure:
{
    "skills": [],
    "education": [],
    "experience_level": "",
    "strengths": [],
    "areas_to_improve": []
}

Do not include Markdown.
Do not include explanations outside the JSON.
Do not invent information. If information is not provided, use an empty list
or an empty string as appropriate.

Candidate profile:
Python developer fresher with knowledge of Python, SQL and Git."""
    response = call_mistral(prompt)
    cleaned_response = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$", "", response, flags=re.IGNORECASE
    ).strip()
    parsed: dict[str, Any] | list[Any] | None = None
    try:
        loaded = json.loads(cleaned_response)
        if not isinstance(loaded, (dict, list)):
            raise json.JSONDecodeError("Expected a JSON object or array", cleaned_response, 0)
        parsed = loaded
        validity_note = "The response was successfully parsed with Python's json module."
    except json.JSONDecodeError as exc:
        validity_note = f"JSON parsing failed: {exc}"
    record = {
        "technique": "JSON Output Prompting",
        "domain": "Career Guidance",
        "prompt": prompt,
        "response": response,
        "valid_json": parsed is not None,
        "parsed_json": parsed,
        "analysis": (
            "A defined schema encourages machine-readable output. "
            f"{validity_note}"
        ),
    }
    _display_record(record)
    RESULTS["json_output"] = [record]
    return [record]


def build_document_prompt(
    document_text: str,
    task: str,
    audience: str = "general user",
    output_format: str = "clear and structured response",
    constraints: list[str] | None = None,
) -> str:
    """Create a prompt grounded in a supplied document and a clear user task."""
    document_excerpt = (document_text or "").strip()
    if not document_excerpt:
        document_excerpt = (
            "Advanced Prompt Engineering: Techniques, Implementation and Analysis. "
            "The document explains nine prompt engineering techniques: zero-shot, "
            "few-shot, structured reasoning, role-based prompting, iterative prompting, "
            "prompt chaining, negative prompting, reusable templates, and JSON output prompting."
        )

    normalized_constraints = constraints or [
        "Use only information supported by the document.",
        "Be clear, useful, and accurate.",
        "Keep the answer relevant to the request.",
    ]
    constraint_block = "\n".join(f"- {item}" for item in normalized_constraints)

    return (
        "You are an expert AI assistant. Use the document below as the primary source of truth.\n\n"
        f"Document context:\n{document_excerpt[:3000]}\n\n"
        f"Task: {task}\n"
        f"Audience: {audience}\n"
        f"Output format: {output_format}\n\n"
        "Rules:\n"
        f"{constraint_block}\n\n"
        "Provide the final answer only in the requested output format."
    )


def build_general_prompt(
    task: str,
    context: str = "",
    audience: str = "general user",
    output_format: str = "clear and concise response",
    constraints: list[str] | None = None,
) -> str:
    """Create a reusable prompt for general-purpose AI tasks without special document context."""
    context_block = f"Context:\n{context.strip()}\n\n" if context.strip() else ""
    normalized_constraints = constraints or [
        "Answer directly and confidently.",
        "Avoid assumptions unless clearly labeled as such.",
        "Keep the response relevant to the request.",
    ]
    constraint_block = "\n".join(f"- {item}" for item in normalized_constraints)

    return (
        "You are a highly capable AI assistant.\n\n"
        f"{context_block}"
        f"Task: {task}\n"
        f"Audience: {audience}\n"
        f"Output format: {output_format}\n\n"
        "Instructions:\n"
        f"{constraint_block}\n\n"
        "Return only the requested output format and do not add filler text."
    )


def document_prompt_generator() -> list[dict[str, Any]]:
    """Generate a prompt grounded in the assignment document context."""
    document_text = (
        "Advanced Prompt Engineering: Techniques, Implementation and Analysis. "
        "This project demonstrates how prompt design can change accuracy, consistency, "
        "relevance, structure, and reliability of outputs from a language model. "
        "It covers nine techniques: zero-shot, few-shot, structured reasoning, role-based "
        "prompting, iterative prompting, prompt chaining, negative prompting, reusable "
        "templates, and JSON output prompting."
    )
    task = (
        "Explain the purpose of prompt engineering in a simple but professional way and "
        "summarize the nine techniques with practical use cases for students."
    )
    audience = "college students and beginners"
    output_format = "Markdown with sections, bullet points, and a short conclusion"
    constraints = [
        "Use only information from the document context.",
        "Keep the tone educational and approachable.",
        "Mention each prompting technique briefly but clearly.",
        "Add a practical takeaway section.",
    ]
    generated_prompt = build_document_prompt(document_text, task, audience, output_format, constraints)
    response = call_mistral(generated_prompt)
    record = _record(
        "Document Prompt Generator",
        "Education",
        generated_prompt,
        response,
        "This prompt turns the document context into a guided AI task, giving the model a clear audience, format, and constraints.",
    )
    RESULTS["document_prompt_generator"] = [record]
    return [record]


def general_prompt_generator() -> list[dict[str, Any]]:
    """Generate a general-purpose prompt template for any real-world task."""
    task = "Create a practical study plan for a beginner learning Python for job readiness."
    context = (
        "The learner is a fresher with basic computer skills, wants to improve Python, "
        "and has 30 minutes per day for study."
    )
    audience = "student"
    output_format = "clear weekly plan with daily activities and learning goals"
    constraints = [
        "Keep it realistic for a beginner.",
        "Do not invent certifications or unrelated technologies.",
        "Include a simple practice activity each day.",
        "Keep it concise but actionable.",
    ]
    generated_prompt = build_general_prompt(task, context, audience, output_format, constraints)
    response = call_mistral(generated_prompt)
    record = _record(
        "General Prompt Generator",
        "Career Guidance",
        generated_prompt,
        response,
        "This general template works for many tasks by defining the goal, context, audience, output format, and rules without tying it to one document.",
    )
    RESULTS["general_prompt_generator"] = [record]
    return [record]


def display_comparison() -> None:
    """Print a concise comparison and practical prompt-design takeaway."""
    rows = [
        ("Zero-shot", "No examples", "Direct response"),
        ("Few-shot", "Uses examples", "Better consistency"),
        ("Structured reasoning", "Organized solution", "Easier to follow"),
        ("Role-based", "Assigns a perspective", "Focused response"),
        ("Iterative", "Multiple revisions", "Refined response"),
        ("Prompt chaining", "Dependent stages", "Complex task handling"),
        ("Negative prompting", "Explicit restrictions", "Less unwanted content"),
        ("Templates", "Reusable placeholders", "Dynamic and reusable"),
        ("JSON prompting", "Defined schema", "Machine-readable output"),
    ]
    print("\nTechnique                 Main Purpose              Expected Benefit")
    print("-" * 76)
    for technique, purpose, benefit in rows:
        print(f"{technique:<26} {purpose:<26} {benefit}")
    print(
        "\nAnalysis: Prompt design influences the response's accuracy, consistency, "
        "relevance, structure, and format. Examples, context, constraints, and "
        "staged tasks help in different ways; no technique is universally best. "
        "Choose based on the task and validate important results."
    )


MENU: dict[str, tuple[str, Any]] = {
    "1": ("Zero-Shot Prompting", zero_shot_prompting),
    "2": ("Few-Shot Prompting", few_shot_prompting),
    "3": ("Structured Reasoning", structured_reasoning_prompting),
    "4": ("Role-Based Prompting", role_based_prompting),
    "5": ("Iterative Prompting", iterative_prompting),
    "6": ("Prompt Chaining", prompt_chaining),
    "7": ("Negative Prompting", negative_prompting),
    "8": ("Reusable Prompt Templates", reusable_prompt_template),
    "9": ("JSON Output Prompting", json_output_prompting),
    "D": ("Document Prompt Generator", document_prompt_generator),
    "G": ("General Prompt Generator", general_prompt_generator),
}


def main() -> None:
    """Display the menu and run selected demonstrations until the user exits."""
    print("=" * 52)
    print("     ADVANCED PROMPT ENGINEERING DEMONSTRATION")
    print("=" * 52)
    if not MISTRAL_API_KEY or MISTRAL_API_KEY == "your_api_key_here":
        print(
            "Configuration note: MISTRAL_API_KEY is missing or still the example "
            "placeholder. Add a valid key to .env before running API demonstrations."
        )

    while True:
        print(
            "\n1. Zero-Shot Prompting\n"
            "2. Few-Shot Prompting\n"
            "3. Structured Reasoning\n"
            "4. Role-Based Prompting\n"
            "5. Iterative Prompting\n"
            "6. Prompt Chaining\n"
            "7. Negative Prompting\n"
            "8. Reusable Prompt Templates\n"
            "9. JSON Output Prompting\n"
            "D. Document Prompt Generator\n"
            "G. General Prompt Generator\n"
            "10. Run All Techniques\n"
            "0. Exit"
        )
        selection = input("\nSelect an option: ").strip()
        if selection == "0":
            break
        if selection == "10":
            for _, function in MENU.values():
                function()
                _save_results()
            display_comparison()
            continue
        if selection in MENU:
            _, function = MENU[selection]
            function()
            _save_results()
            if selection == "3":
                print(
                    "\nComparison: the simple prompt asks only for code; the structured "
                    "prompt additionally requests an approach, considerations, and a "
                    "result explanation."
                )
            continue
        print("Invalid selection. Choose a number from 0 to 10.")

    _save_results()
    print("Goodbye.")


if __name__ == "__main__":
    main()
