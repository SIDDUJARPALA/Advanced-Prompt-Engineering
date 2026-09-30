
# Advanced Prompt Engineering: Techniques, Implementation and Analysis

## Objective

This project demonstrates how prompt design can change the accuracy, consistency,
relevance, structure, and reliability of responses from a Large Language Model. It
uses the Mistral API and provides an interactive Python console menu for a college
demonstration.

## Technologies

- Python 3.11
- Mistral API (`mistralai`)
- `python-dotenv` for local environment configuration
- Python's built-in `json` module for result storage and JSON validation

## Nine Techniques

1. **Zero-shot prompting** gives the model a task without examples; three prompts cover education, software development, and career guidance.
2. **Few-shot prompting** supplies input-output examples to guide sentiment classification and response format.
3. **Structured reasoning** compares a direct coding request with a concise, sectioned solution request. It does not request hidden chain-of-thought.
4. **Role-based prompting** assigns Python developer, data analyst, interviewer, and career counsellor roles.
5. **Iterative prompting** refines a resume summary through two feedback rounds.
6. **Prompt chaining** passes extracted job skills into categorization and then into a study-plan prompt.
7. **Negative prompting** adds explicit restrictions to avoid unsupported or irrelevant resume and learning-plan details.
8. **Reusable prompt templates** fill dynamic topic, difficulty, audience, and output-format placeholders for three examples.
9. **JSON output prompting** requests a fixed candidate-profile schema and validates the response using `json.loads`.

Examples span **Education**, **Software Development**, and **Career Guidance**. Every displayed model response comes from a Mistral API request; API errors are clearly labeled rather than replaced with fabricated output.

## Installation

From this project directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 is recommended for the assignment. The `.venv/` directory is excluded
from Git.

## Environment Configuration

Copy `.env.example` to `.env` if needed, then replace the placeholder with your
Mistral API key:

```text
MISTRAL_API_KEY=your_api_key_here
```

Keep the real key private. `.env` is ignored by Git; `.env.example` contains only a
placeholder. The application displays a setup message when the key is missing or
still set to the placeholder.

## Running the Project

```powershell
python prompt_engineering.py
```

Select one technique (1-9), run all techniques with option 10, or exit with 0.
Each technique prints its domain, prompt, response, and analysis. API request errors
are shown clearly and do not stop the complete menu program.

## Output

Results are saved as UTF-8 JSON in:

```text
outputs/results.json
```

The file contains a top-level entry for each of the nine techniques. Selective runs
populate the selected entry; **Run All** fills all entries. The JSON demonstration
also stores whether the model response parsed successfully and the parsed value
when valid.

## Screenshots

Capture screenshots of:

- The application menu and a selected technique, including its prompt and response.
- The structured reasoning comparison, showing the direct and organized outputs.
- The iterative prompt's initial, improved, and final rounds.
- All three prompt-chaining stages.
- JSON output validation and the parsed result.
- `outputs/results.json` after using **Run All Techniques**.

Save assignment screenshots in the `screenshots/` directory.

## Comparison of Prompting Techniques

| Dimension | How prompt design can affect it |
|---|---|
| Accuracy | Clear context and constraints can reduce ambiguity, but do not guarantee factual correctness. |
| Consistency | Few-shot examples and explicit output instructions can make responses more uniform. |
| Relevance | A role, audience, and task context can focus the answer on the intended need. |
| Output control | Requested sections, formats, and negative constraints guide response shape and content. |
| Reusability | Templates separate stable instructions from variable inputs. |
| Structured output | JSON prompting can support downstream software when the response validates against the expected schema. |
| Complexity | Prompt chaining divides a multi-part task into dependent, inspectable stages, at the cost of multiple requests. |
| Practical use cases | Direct questions suit simple tasks; examples suit repeated formats; roles suit perspective-focused tasks; iterative prompts suit refinement; and JSON suits data exchange. |

No technique is universally superior. Effectiveness depends on the task, the
information supplied, the model, and how the response will be used. Important
outputs should be reviewed rather than assumed correct.

## Conclusion

The project shows that instructions, examples, roles, constraints, templates, and
multi-stage workflows shape how an LLM responds. Comparing techniques helps select
a prompt design that fits the required task and output.

