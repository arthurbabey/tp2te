# tp2te v0 — build spec

Nov 11, 2026 · @arthur

## 0. For the build session

This doc is the spec for a Claude Code session that builds `tp2te` v0.

- Build in the milestone order of section 14. Finish and test one milestone before starting the next.
- The layout in section 4 and the models in section 5 are the contract. Change them only when a real notebook forces it, and record the change in the spec.
- `core/` and `llm/` never import Streamlit. The CLI must be able to run the whole pipeline.
- The prompts in section 7 are drafts. Put them in `prompts/` as written, then iterate against the real notebooks.
- Do not guess how the template marks tasks and questions. Read the real template first; the files Arthur provides are listed in section 15.
- Roster names never go into a prompt. Student notebooks are never executed.
- Tests use `FakeClient` and make no network calls.
- Stack: Python 3.11+, nbformat, Pydantic v2, rapidfuzz, Jinja2, Streamlit, pytest.

## 1. Settled decisions

These are fixed for v0. The build session should not reopen them.

### From the original design

- **Format:** a written quiz, in person, one paper sheet per student.
- **Numbers:** a pool of 6 questions per group and 3 questions per student. Both are config values.
- **All questions are personalised** to the group's notebook. Students in a group get different subsets. Different students answering different questions is accepted, as in an oral exam.
- **The teacher grades by hand,** one click per answer. The grade levels are a config list, defaulting to Perfect / Very good / Insufficient / Bad / Not done. The LLM never grades.
- **Mistakes become questions.** Completeness earns the base points.
- **Workflow timing is the teacher's business.** Deadlines, review time and quiz length are not modelled in the app.
- **No AI detection.**

### Added after review

- **Grounding validation.** Every issue and question carries a cell id and a verbatim quote, and code checks the quote exists in that cell. Section 7 has the rule.
- **Snippet printed with each question.** The student has no notebook in front of them, so the question must carry the code or output it refers to.
- **Neutral wording for spot-the-issue questions,** and at least one of them on correct code, so the question type does not give away the answer. Types are not printed on the student sheet.
- **Roster input** (group, student). Names stay local and appear only on the printed sheet.
- **Fallback pool.** Generic questions generated once from the template, used only for late groups, absent students, or a group with too few approved questions.
- **Disk persistence.** Grading happens after class and Streamlit session state does not survive a refresh.
- **Roster-driven scrubber** for names, emails, home-directory paths in outputs, and notebook metadata.
- **Text outputs only.** Plots are not sent to the LLM in v0; cells with an image are marked.
- **Audit log** of every LLM call and of each question as printed.
- **Quiz language** is a prompt parameter.

### Paper or on screen

v0 stays on paper. An on-screen quiz needs a student-facing page, hosting and logins, which are all out of scope, and its only protection against AI use is supervision. Walking behind the screens with a "zero if caught" rule is a deterrent, not a check; for a low-stakes quiz that may be enough, and it is your call. The stronger option later is a lockdown browser such as Safe Exam Browser, which works with Moodle, at the cost of importing a question set per student. To keep the door open, rendering is a separate layer: an on-screen mode would be a second renderer over the same assignments.

## 2. Revised v0 scope

v0 is the smallest thing that gets correct, printed, per-student sheets into the room and grades out as CSV. Everything else waits.

| Change | What | Why |
| --- | --- | --- |
| Add | Roster file (group, student) as an input | Needed for per-student sheets; names never leave the laptop |
| Add | Lab-level brief and generic question pool, generated once from the template | Context for the per-group calls; fallback questions for late groups and absent students |
| Add | Grounding check on every issue and question | Only automatic defence against hallucinated questions |
| Add | Code or output snippet printed with each question | Paper quiz, no notebook in front of the student |
| Add | Run file on disk, saved after every edit | Grading happens hours later; session state will not survive |
| Add | Scrubber driven by the roster (names, emails, paths, metadata) | "Strip names" is not reliable otherwise |
| Add | CLI that runs the pipeline without the UI | Needed to test on past submissions and iterate on prompts |
| Add | Log of every LLM call (prompt version, tokens, response) | Audit trail and cost tracking |
| Change | Completeness judgment merged into the analysis call | One call fewer per group |
| Change | Grade levels read from config | Grading is manual, so the scale is a setting |
| Change | Pool size and questions per student read from config (default 6 and 3) | The number of questions is the setting that matters most |
| Change | Question-type mix in a config file, not a settings screen | Saves a screen; the review page still shows the distribution |
| Change | Printable sheets as HTML with print CSS, printed from the browser | WeasyPrint needs system libraries that are painful on some laptops; add it later |
| Cut | Class summary from the critical path | Useful, but build it last; the lab can run without it |
| Cut | Sending plots to the LLM | Text outputs only in v0 |
| Keep out | Everything on your out-of-scope list | Agreed, all of it |

## 3. Architecture

A linear pipeline writes everything into one run file on disk, and the Streamlit screens only read and edit that file.

```text
Ingest (zip, roster, parse, scrub) -> Section matching -> Hard checks
                                                              |
L0 lab brief (once, from template) ----------+                v
                                             +--> L1 analysis (section status, issues with evidence)
                                             |        |
                                             +--> L2 questions (pool per group, expected points)
                                                      |
                                                      v
                                          Ground + assign (verify quotes, subsets per student)
                                                      |
                                                      v
                Run store: run.json + llm_log.jsonl on disk, saved after every step and edit
                     ^                    ^                     ^
                     v                    v                     v
               Review screen         Print screen          Grade screen
          (edit, approve, redo,    (HTML sheet per       (teacher grades,
           class summary L4)      student, teacher key)     CSV export)

All L-calls go through one LLM client (hosted API, Swiss/EU host, local model, fake for tests).
```

The top two rows are `core/` and `llm/` and run from the CLI without Streamlit; the bottom row is `app/`. L3 (regenerate) is called from the Review screen and goes through the same grounding check as L2.

## 4. Repo and module layout

`core/` and `llm/` import nothing from Streamlit, so they carry over to FastAPI unchanged.

```text
tp2te/
  pyproject.toml
  config.example.toml          # provider, model, question mix, languages, limits
  src/tp2te/
    core/
      models.py                # all Pydantic models
      ingest.py                # zip + roster -> groups, students
      notebook.py              # nbformat -> NotebookDoc, output truncation, prompt serialisation
      scrub.py                 # names, emails, paths, metadata
      matching.py              # template sections, cell alignment
      checks.py                # hard completeness checks
      brief.py                 # L0: lab brief + generic pool
      analysis.py              # L1: completeness + issues
      questions.py             # L2, L3 + grounding check
      assign.py                # question subsets per student
      summary.py               # L4: class summary
      render.py                # Jinja2 -> HTML sheets and teacher key
      scoring.py               # grades -> points, CSV export
      store.py                 # run directory, load/save, LLM log
      pipeline.py              # orchestration, concurrency, cache
    llm/
      base.py                  # LLMClient protocol, LLMResult
      openai_compat.py         # any OpenAI-compatible endpoint, hosted or local
      anthropic.py
      fake.py                  # replays recorded responses for tests
    prompts/                   # one Jinja file per call, versioned
    templates/                 # sheet.html.j2, key.html.j2, sheet.css
    app/
      Home.py                  # pick or create a run
      pages/1_Upload.py  2_Review.py  3_Print.py  4_Grade.py
      state.py                 # thin session-state helpers
    cli.py                     # tp2te brief | run | eval | render
  tests/
    fixtures/                  # template, perturbed notebooks, recorded LLM responses
    test_matching.py  test_checks.py  test_grounding.py  test_assign.py  test_scrub.py
  runs/                        # gitignored: one directory per lab run
```

## 5. Data models

One `LabRun` object holds everything and is the file on disk. Models marked *LLM output* are the schemas passed to the provider; the rest are filled by code.

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

# ---------- notebook ----------
class Cell(BaseModel):
    cell_id: str                      # "c017": stable within a run, used in prompts
    idx: int                          # position in the submitted notebook
    kind: Literal["code", "markdown"]
    source: str
    outputs_text: str = ""            # stdout + text/plain, truncated
    has_image: bool = False
    error_name: str | None = None     # e.g. "ValueError"
    execution_count: int | None = None
    origin: Literal["unchanged", "modified", "added"] = "added"   # vs template
    template_cell_id: str | None = None

class NotebookDoc(BaseModel):
    group_id: str                     # "G07"
    cells: list[Cell]
    content_hash: str                 # cache key

# ---------- template and matching ----------
class TemplateSection(BaseModel):
    section_id: str                   # "S3"
    title: str
    level: int
    cell_ids: list[str]               # template cells in this section
    task_cell_ids: list[str]          # code the student must write
    question_cell_ids: list[str]      # markdown questions the student must answer

class Template(BaseModel):
    lab_id: str
    cells: list[Cell]
    sections: list[TemplateSection]

class SectionMatch(BaseModel):
    section_id: str
    cell_ids: list[str]               # student cells attributed to this section
    anchors_found: int
    anchors_total: int
    confidence: float                 # anchors_found / anchors_total
    method: Literal["anchor", "fuzzy", "position", "unmatched"]

# ---------- completeness ----------
SectionStatus = Literal["done", "partial", "missing"]

class HardFinding(BaseModel):
    kind: Literal["empty_cell", "todo_left", "not_executed", "error_output",
                  "unanswered_question", "missing_section", "out_of_order"]
    section_id: str | None
    cell_id: str | None
    detail: str

class SectionJudgment(BaseModel):      # LLM output
    section_id: str
    status: SectionStatus
    reason: str                        # one sentence

class SectionCompleteness(BaseModel):
    section_id: str
    hard_findings: list[HardFinding]
    llm: SectionJudgment
    final_status: SectionStatus        # defaults to llm.status; teacher can override

class CompletenessResult(BaseModel):
    sections: list[SectionCompleteness]
    score: float                       # done = 1, partial = 0.5, missing = 0, averaged

# ---------- issues ----------
class Evidence(BaseModel):             # LLM output
    cell_id: str
    quote: str                         # verbatim from that cell's source or output

class Issue(BaseModel):                # LLM output, except issue_id and grounded
    issue_id: str = ""
    section_id: str
    category: Literal["bug", "methodology", "suspicious_result",
                      "wrong_interpretation", "questionable_choice"]
    severity: Literal["minor", "major"]
    concept: str                       # short tag, e.g. "data leakage"
    title: str
    explanation: str
    evidence: list[Evidence]
    grounded: bool = False             # set by code

class Choice(BaseModel):               # LLM output: a defensible decision worth a "why"
    section_id: str
    description: str
    evidence: list[Evidence]

class GroupAnalysis(BaseModel):        # LLM output of L1
    judgments: list[SectionJudgment]
    issues: list[Issue]
    choices: list[Choice]

# ---------- questions ----------
QuestionType = Literal["why", "spot_issue", "what_if", "concept"]

class QuestionDraft(BaseModel):        # LLM output of L0, L2, L3
    type: QuestionType
    section_id: str
    concept: str
    difficulty: Literal[1, 2, 3]
    prompt: str                        # as printed, in the quiz language
    snippet: str | None                # verbatim code or output, max 15 lines
    evidence: list[Evidence]
    expected_points: list[str]         # 2 to 4 points the grader looks for
    source_issue_index: int | None

class Question(QuestionDraft):
    question_id: str                   # "G07-Q3" or "LAB-Q2"
    scope: Literal["group", "lab"]
    status: Literal["draft", "approved", "rejected"] = "draft"
    grounded: bool = False
    edited: bool = False
    version: int = 1                   # bumped on regenerate

class QuestionPool(BaseModel):
    owner: str                         # group_id or "LAB"
    questions: list[Question]

# ---------- students, assignment, grades ----------
class Student(BaseModel):
    student_id: str                    # "G07-B"
    group_id: str
    display_name: str                  # local only, never in a prompt

class StudentAssignment(BaseModel):
    student_id: str
    question_ids: list[str]            # in print order
    sheet_code: str                    # "L3-G07-B"
    present: bool = True

GradeLevel = str                       # one of config.grading.levels

class Grade(BaseModel):
    student_id: str
    question_id: str
    level: GradeLevel
    note: str = ""

# ---------- lab brief, summary, run ----------
class SectionBrief(BaseModel):         # LLM output of L0
    section_id: str
    concepts: list[str]
    expected_work: str
    common_pitfalls: list[str]

class LabBrief(BaseModel):             # LLM output of L0
    sections: list[SectionBrief]
    generic_questions: list[QuestionDraft]

class SummaryItem(BaseModel):          # LLM output of L4
    concept: str
    statement: str                     # "7 of 10 groups fit the scaler before the split"
    group_ids: list[str]
    debrief_hint: str

class GroupRun(BaseModel):
    group_id: str
    notebook: NotebookDoc
    matches: list[SectionMatch]
    completeness: CompletenessResult | None = None
    issues: list[Issue] = []
    choices: list[Choice] = []
    pool: QuestionPool | None = None
    error: str | None = None           # pipeline failure for this group

class LabRun(BaseModel):
    lab_id: str
    created_at: datetime
    config: dict                       # snapshot: provider, model, mix, prompt versions
    template: Template
    brief: LabBrief | None = None
    lab_pool: QuestionPool | None = None
    students: list[Student]
    groups: dict[str, GroupRun]
    assignments: list[StudentAssignment] = []
    assignments_locked: bool = False   # set when sheets are printed
    grades: list[Grade] = []
    summary: list[SummaryItem] = []
```

## 6. Section matching

Match on the template's own text, not on position: instruction cells and headings are the anchors, because students rarely edit them. No embeddings and no LLM are needed for this step.

1. **Build template sections.** Split the template at markdown headings down to a configurable level. Mark task cells and question cells using configurable markers (TODO patterns, answer placeholders). I need the real template to set these; I am not assuming how yours marks them.
2. **Fingerprint cells.** Normalise each cell (lowercase, collapse whitespace). Exact hash match to a template cell gives `unchanged`.
3. **Fuzzy match the rest.** For each remaining student cell, score it against template cells of the same kind with `rapidfuzz`. Use two scores: full similarity (edited cell) and containment (share of the student cell's lines found in the template cell, which catches split cells). Above the threshold gives `modified`; otherwise `added`.
4. **Keep an order-consistent set of anchors.** Take the matched instruction and heading cells, and keep the longest increasing subsequence in template order. This drops spurious matches. A cell that matches at 0.9 or more but sits out of order is kept as a moved anchor.
5. **Attribute every student cell to a section.** Matched cells go to their template cell's section. Added cells go to the section of the nearest preceding anchor, or the following one if there is none.
6. **Handle merges.** A student cell that contains several template cells is attributed to the first one's section and flagged.
7. **Detect answers by subtraction.** For a question cell, remove the template text from the student's text (and look at the next added markdown cell). If fewer than a set number of characters remain, it is unanswered. For a task cell, remove template lines, comments, TODO markers, `pass` and `...`; nothing left means not done.
8. **Score confidence per section** as anchors found over anchors expected. Below 0.5 the section is shown as "low-confidence match" in review.

Matching feeds the hard checks and gives the LLM section tags as hints. The LLM always receives the whole notebook, so a bad match degrades the hard checks for one section but does not hide student work from the analysis.

Hard checks that run on the matched cells: empty task cell, TODO marker left, code cell with no `execution_count`, error in outputs, unanswered question cell, section with no student cells, and execution counts that decrease down the notebook (cells run out of order, so outputs may not correspond to the code shown).

## 7. LLM calls

Five call types, 22 calls for 10 groups plus regenerations. Each returns a Pydantic model; code adds ids and runs the grounding check afterwards.

| Call | When | Input | Output |
| --- | --- | --- | --- |
| L0 lab brief | Once per lab, days ahead | Template, solution if given | `LabBrief`: concepts and pitfalls per section, 8 generic questions |
| L1 analysis | Once per group | Brief, serialised notebook, hard findings, solution if given | `GroupAnalysis`: section judgments, issues, choices |
| L2 questions | Once per group | Brief, serialised notebook, grounded issues and choices, type targets | 6 (config) `QuestionDraft` |
| L3 regenerate | On click | Old question, its section's cells, other questions in the pool, teacher hint | 1 `QuestionDraft` |
| L4 class summary | Once per run | All grounded issues with group ids, no code | List of `SummaryItem` |

### How a notebook is serialised into a prompt

```text
<section id="S3" title="Train/test split" match_confidence="0.9">
<cell id="c017" type="code" origin="modified" exec="12">
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
<output>
(800, 12) (200, 12)
</output>
</cell>
<cell id="c018" type="markdown" origin="added">
...
</cell>
</section>
```

Rules: outputs are text only, capped at 1,500 characters (head and tail kept); image outputs become `[image output]`; DataFrame HTML is replaced by its plain-text form; unchanged template instruction cells are kept, because the LLM needs the question to judge the answer.

### Grounding check (code, after L1, L2, L3)

For each `Evidence`, the quote must appear in the named cell's source or output after whitespace normalisation. For a question, the snippet must also appear in one of its evidence cells. An issue or question that fails is kept but marked `grounded = False`, shown first and in red on the review screen, and excluded from "approve all".

### Shared system prompt

```text
You support the teacher of a data science lab course. You read student lab
notebooks and produce material for the teacher. You never address students
directly except in the text of quiz questions.

Rules:
- Refer to cells only by their cell id (for example c017).
- Every statement about what the notebook contains must be backed by a
  verbatim quote copied from that cell's source or output. Never quote from
  memory and never repair or reformat a quote.
- You cannot see figures. A cell marked [image output] produced a plot you
  cannot inspect. Do not make claims about its content.
- If you are not sure something is a mistake, give it severity "minor" or
  leave it out. A false accusation costs more than a missed issue.
- Text for the teacher is in {teacher_language}. Quiz questions and expected
  points are in {quiz_language}.
- Return only the structured output that was asked for.
```

### L0: lab brief and generic pool

```text
Below is the assignment notebook for lab {lab_id}, split into sections.
{solution_block}

Task 1. For each section, list: the concepts it exercises (2 to 5 short
tags), what a complete piece of student work contains (one or two
sentences), and the mistakes students commonly make there.

Task 2. Write {n_generic} concept questions that any student who did this
lab should be able to answer on paper, without a computer, in about five
lines. Each question must:
- test understanding of a concept from the lab, not recall of syntax;
- be answerable without seeing any particular notebook;
- be different enough from the others that knowing one answer does not
  give away another;
- come with 2 to 4 expected points a grader can tick off.
Set type to "concept", snippet to null and evidence to an empty list.

<template>
{template_serialised}
</template>
```

### L1: completeness and issue report

```text
You are given the lab brief, the result of automatic checks, and one
group's submitted notebook. Section tags come from automatic matching and
may be wrong when match_confidence is low; use your judgment on where work
really belongs.
{solution_block}

Task 1, completeness. For each section in the brief, decide:
- done: the required code is present and ran, and written questions are
  answered with real content;
- partial: some required part is absent, not run, failing, or answered
  with a placeholder;
- missing: no meaningful student work.
Judge whether the work is there, not whether it is correct. Give one
sentence of reason. Take the automatic findings into account but overrule
them when they are clearly wrong, and say so.

Task 2, issues. List problems in this group's work, most important first,
at most {max_issues}. Look for: bugs; methodological errors (for example
leakage, evaluating on training data, wrong metric for the task); results
that are too good, too bad or inconsistent with the code; written answers
that misinterpret the group's own results; choices that work but are hard
to defend. For each: section, category, severity, a short concept tag
(reuse tags from the brief when one fits), a title, a two-sentence
explanation for the teacher, and evidence quotes.

Task 3, choices. List up to {max_choices} decisions this group made that
are correct or defensible and that a student who understands the work
should be able to justify (a parameter value, a preprocessing step, a
model, a metric). Give evidence quotes.

<brief>
{brief_json}
</brief>
<automatic_findings>
{hard_findings}
</automatic_findings>
<notebook>
{notebook_serialised}
</notebook>
```

### L2: question generation

```text
Write {n_questions} quiz questions for the students of this group. Each
student will answer on paper, alone, without a computer and without the
notebook, in about five lines and three minutes per question. The purpose
is to check that the student understands what their own group did and why.

Targets by type: {type_targets}
- why: justify a choice the group made.
- spot_issue: show a piece of the group's code or output and ask the
  student to assess it. Use neutral wording such as "Comment on the
  validity of this result" or "What does this cell do to the test data?".
  Never hint that something is wrong. At least one spot_issue question
  must be about code that is correct.
- what_if: change one thing in the group's code or data and ask what
  happens and why.

Requirements for every question:
- It is about this group's notebook specifically. A question that could
  be asked of any group is rejected.
- It is self-contained: include in "snippet" the exact code or output the
  student needs, copied verbatim, 15 lines at most. Trim by choosing a
  contiguous excerpt, not by editing lines.
- It cannot be answered by reading the snippet aloud; it needs reasoning.
- It does not depend on a figure.
- Questions cover at least {min_sections} different sections and do not
  overlap in what they test.
- Build questions from the issues first, most severe first, then from the
  choices. Do not reveal in the question that an issue was found.
- difficulty: 1 = direct explanation, 2 = requires connecting two ideas,
  3 = requires predicting behaviour not shown in the notebook.
- expected_points: 2 to 4 short points a grader can tick off. For a
  question built on an issue, the first point states the issue.
- evidence: the cells and quotes the question relies on.

<brief>
{brief_json}
</brief>
<issues>
{grounded_issues_json}
</issues>
<choices>
{choices_json}
</choices>
<notebook>
{notebook_serialised}
</notebook>
```

### L3: regenerate one question

```text
Replace the question below with a new one for the same group. Keep the
same type ({type}) unless the teacher's note says otherwise. The new
question must test something different from the rejected question and
from the other questions already in the pool. All requirements for quiz
questions apply (self-contained, verbatim snippet of 15 lines at most,
neutral wording, 2 to 4 expected points, evidence quotes).

Teacher's note (may be empty): {teacher_hint}

<rejected_question>
{question_json}
</rejected_question>
<other_questions_in_pool>
{other_prompts}
</other_questions_in_pool>
<issues_and_choices_not_yet_used>
{unused_json}
</issues_and_choices_not_yet_used>
<notebook>
{notebook_serialised}
</notebook>
```

### L4: class summary

```text
Below are the issues found across {n_groups} groups for lab {lab_id}, each
with its group id, section and concept tag. Group issues that are the same
underlying mistake, even when tags or wording differ. Return the patterns
that affect at least {min_groups} groups, most frequent first, at most 8.
For each: the concept, one sentence stating how many groups and what they
did ("7 of 10 groups ..."), the list of group ids, and one sentence the
teacher can use to open a two-minute debrief. Count carefully: the number
in the sentence must equal the length of the group id list.

<issues>
{all_issues_compact}
</issues>
```

Code recomputes the count in each summary sentence from `group_ids` and flags any mismatch.

## 8. Assigning questions to students

The search space is tiny, so enumerate every option and pick the best; no heuristic is needed. Each student gets `k` questions (default 3) from the group's approved pool (default 6), following the configured type mix.

1. List every `k`-subset of the approved pool: 20 subsets for 3 out of 6.
2. Enumerate every combination of one subset per student: 400 for a group of two, 8,000 for a group of three.
3. Score each combination and keep the lowest, comparing in this order:
   - type-mix violations, summed over students;
   - students with identical subsets (must be 0);
   - questions shared between students, counted over all pairs (a group of two can reach 0; a group of three with a pool of 6 cannot);
   - approved questions that nobody receives;
   - spread of total difficulty between students (max minus min);
   - students whose questions all come from one section.
4. Break ties with a random generator seeded by `hash(lab_id, group_id)`, so the result is reproducible.

**Edge cases**

- Too few approved questions of a type: the mix is a penalty, not a hard rule, so an assignment is still returned and the review screen shows a warning.
- Fewer than `k` approved questions in the pool: fill from the lab fallback pool and warn.
- Late group or absent student: questions come from the lab fallback pool.
- Assignments are recomputed whenever a question is approved or rejected, and frozen when sheets are printed. After that, edits do not move questions between students.

## 9. Streamlit screens and state

Session state holds pointers only; the `LabRun` on disk is the state. Every edit calls `store.save(run)`, and every page starts with `run = store.load(run_id)`.

| Screen | What the teacher does | What it writes |
| --- | --- | --- |
| Home | Picks an existing run or starts a new one | `run_id` in session state |
| Upload | Uploads template, optional solution, submissions zip, roster. Sees a pre-flight table: groups found, students per group, files that failed to parse. Clicks Generate and watches a per-group progress list | Template, groups, students, then analysis and pools as each group finishes |
| Review | Left: group list with badges (completeness, issues, ungrounded items, low-confidence matching). Right, three tabs per group: Completeness (hard findings beside the LLM judgment, override dropdown), Issues, Questions. Each question card shows type, difficulty, snippet, editable text and expected points, with Approve, Reject and Regenerate (optional hint). Top bar: type distribution and a preview of who gets what. A Class tab holds the summary and the lab pool | Question edits and status, completeness overrides, assignments |
| Print | Checks every student has 3 approved questions, locks assignments, renders student sheets and the teacher key, offers both as HTML to print | `assignments_locked` |
| Grade | Walks students in the order the sheets were printed. For each: the three questions, expected points, one button per configured grade level for each answer, an Absent toggle, a note field, Next. Progress counter. Export CSV | Grades |

**Session state keys:** `run_id`, `selected_group`, `selected_student_idx`, `review_tab`, `generation_status`. Nothing else.

**Generation:** `pipeline.run()` processes groups in a thread pool (5 at a time) and saves after each group, so a crash or a refresh loses nothing. Results are cached on notebook hash plus prompt version plus model, so clicking Generate twice costs nothing.

**CSV columns:** `student_id, name, group_id, completeness, q1, q2, q3, quiz_points, total`. Points per level and the completeness weight sit in the config file and are applied in `scoring.py`.

## 10. Printed sheet

One A4 sheet per student, double-sided at most, printed from one HTML file with a page break after each student.

- **Header:** course and lab, date, student name, group, sheet code (for example `L3-G07-B`), and the rules in one line: no documents, no devices, time allowed.
- **Each question:** number, question text, then the snippet in a bordered monospace box (9 pt, wrapped, 15 lines at most), then a ruled answer area of about six lines. The fixed answer area is what bounds answer length.
- **Not printed:** question type, difficulty, expected points.
- **Footer:** sheet code and page number on every page, so loose pages can be re-paired.
- **Order:** by group, then student, so handing out follows the room. The Grade screen uses the same order as the stack of paper.
- **Teacher key:** a second HTML file, same order, with expected points under each question. Optional to print, since the Grade screen shows the same.

```css
@page { size: A4; margin: 15mm; }
.sheet { break-after: page; }
.question { break-inside: avoid; margin-bottom: 8mm; }
pre.snippet { font: 9pt/1.3 monospace; white-space: pre-wrap; border: 0.5pt solid #000; padding: 2mm; }
.answer { height: 42mm; background: repeating-linear-gradient(#fff, #fff 6.8mm, #999 6.8mm, #999 7mm); }
```

If browser printing proves fiddly, WeasyPrint can render the same HTML and CSS to PDF without changing the template.

## 11. LLM provider abstraction

One method, typed by the Pydantic schema; the rest of the code never sees a provider SDK.

```python
from typing import Generic, Protocol, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMResult(BaseModel, Generic[T]):
    parsed: T
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float

class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, schema: type[T],
                 temperature: float = 0.2, max_tokens: int = 4000) -> LLMResult[T]: ...

def make_client(cfg: LLMConfig) -> LLMClient: ...   # factory keyed on cfg.provider
```

- **`OpenAICompatClient`**: base URL, key and model from config. Covers most hosted APIs, Swiss or EU hosts that expose the same interface, and local servers such as Ollama or vLLM. This is the one to build first.
- **`AnthropicClient`**: same interface over the Anthropic SDK.
- **`FakeClient`**: returns recorded responses keyed on prompt hash. Used by tests and for UI work without spending tokens.

Structured output: pass `schema.model_json_schema()` through the provider's JSON-schema or tool mechanism, then always validate with Pydantic. On a validation error, retry once with the error message appended. For local models without schema support, fall back to "JSON in a fenced block" plus the same validate-and-retry.

```toml
[llm]
provider = "openai_compat"
base_url = "https://..."
model = "..."
api_key_env = "TP2TE_API_KEY"

[llm.overrides]
summary = { model = "..." }      # cheaper model for L4 if wanted
```

Every call is appended to `llm_log.jsonl` in the run directory: call type, prompt version, model, token counts, latency, raw response. API keys come from the environment, never from the run file.

## 12. Cost and latency for 10 groups

Roughly 0.55 million input tokens and 0.055 million output tokens per lab: a few dollars on a strong hosted model, and two to four minutes of waiting when groups run in parallel. These are estimates built on assumed sizes; the first run on real notebooks will replace them, since the log records actual tokens.

Assumed sizes: serialised notebook 12k tokens, template 4k, solution 6k, lab brief 2k.

| Call | Count | Input each | Output each | Input total | Output total |
| --- | --- | --- | --- | --- | --- |
| L0 lab brief | 1 | 12k | 3k | 12k | 3k |
| L1 analysis | 10 | 21k | 2.5k | 210k | 25k |
| L2 questions | 10 | 18k | 2k | 180k | 20k |
| L3 regenerate | 10 (assumed) | 14k | 0.5k | 140k | 5k |
| L4 class summary | 1 | 6k | 1k | 6k | 1k |
| **Total** | 32 |  |  | **548k** | **54k** |

**Cost.** At an assumed price of $3 per million input tokens and $15 per million output tokens, that is about $1.65 + $0.80 = $2.50 per lab, or about $12 for five labs. A small model at a tenth of that price costs about $0.25 per lab. These prices are placeholders, not quotes; check your provider's current rates. Cost is not the constraint either way.

**Latency.** L1 and L2 each take roughly 30 to 60 seconds per group, driven by output length. With 5 groups in parallel, the full run takes about 2 to 4 minutes; sequentially, 10 to 20 minutes. A local model on a laptop will be several times slower and weaker on a 20k-token context, so plan that as an overnight run and expect to reject more questions.

## 13. Testing

Test the deterministic parts with unit tests and the LLM parts against a small set of past submissions that you annotate by hand.

**Unit tests, no LLM**

- Matching: take the template, apply scripted perturbations (move a cell, split one, merge two, delete a heading, rename a heading, add ten new cells) and assert the section attribution.
- Hard checks: one fixture notebook per finding kind.
- Grounding: quotes with changed whitespace pass; invented quotes fail.
- Assignment: for random pools and group sizes, no two students share an identical set, the mix holds when it can, and the same seed gives the same result.
- Scrubber: after serialisation, no roster name, email or home-directory path appears in any prompt. This test guards the privacy promise, so it runs on every real notebook too, as a pre-flight check.

**Evaluation on past submissions (`tp2te eval`)**

- Pick one past lab and 5 submissions, including a messy one and a near-perfect one. Annotate each: status per section, and the issues you would want caught.
- Report per run: section-matching accuracy, agreement on completeness, share of your issues found, number of issues you judge false, grounding pass rate, and the share of questions you would approve without editing.
- Targets to ship: no false "major" issue in the set, at least 70% of questions approvable unedited, grounding pass rate above 90%.
- Keep prompt versions in the report, so a prompt change that makes things worse is visible.

**Dry run before the first real class**

- Full pipeline on one complete past lab, timed from upload to printed sheets, including your review.
- Print two sheets and have someone who knows the material answer them.
- Grade those two on the Grade screen and export the CSV.

## 14. Build order

Five milestones, each with a check that must pass before the next starts. The pipeline is built as a CLI first; Streamlit goes on top at milestone 4.

1. **M1: read notebooks.** Repo, models, `notebook.py`, `scrub.py`, `ingest.py`, `store.py`. Done when `tp2te run` parses the real submissions zip and roster, writes a run file, and the scrubber test finds no roster name in any serialised notebook.
2. **M2: matching, checks, first LLM calls.** `matching.py`, `checks.py`, `OpenAICompatClient`, `FakeClient`, L0 and L1. Done when the perturbation tests pass and the CLI prints a completeness table and an issue report for three real notebooks.
3. **M3: questions and sheets.** L2, grounding check, `assign.py`, `render.py`. Done when the CLI goes from zip to printable HTML sheets, every printed question is grounded, and the assignment tests pass. Then iterate on the L1 and L2 prompts with Arthur against the real notebooks.
4. **M4: Upload, Review and Print screens.** Editing, approve and reject, L3 regenerate, locked assignments. Done when a full lab can be generated, reviewed and printed from the browser, and a refresh loses nothing.
5. **M5: Grade screen, CSV, class summary.** Done when grades entered on screen survive a restart and the CSV matches them, and `tp2te eval` produces its report on the annotated notebooks.

**If something must be dropped, drop in this order:** class summary, the completeness tab (show hard findings only), the Grade screen (export a CSV template and fill it by hand), L3 regenerate. Do not drop the grounding check, the scrubber, or disk persistence.

## 15. What I need from you

The matching rules and the prompts are drafts until they have met a real template and real submissions.

**Files**

- [ ] The assignment notebook for the next lab, and one from a past lab
- [ ] The solution notebook for each, if one exists
- [ ] 3 to 5 anonymised past submissions for that past lab, including a messy one
- [ ] A sample of the Moodle zip structure (file and folder names only is enough)
- [ ] The roster format you can export (columns)

**Decisions**

- [ ] Language of the quiz questions, and of the teacher-facing reports
- [ ] Which LLM providers the school allows for student work, or who decides
- [ ] How completeness and quiz grades combine into the lab grade, for the CSV

## 16. Changes forced by the real notebooks

Log of deviations from sections 4 and 5, per section 0. Source: the 9 TP2 submissions in `data/example/` (Nov 11, 2026).

- `Cell.nb_cell_id` (new field): the nbformat cell id. 72 of ~80 template cell ids are identical in all 9 submissions, so the id is the strongest matching anchor for M2. One group's notebook has 7 template cells with regenerated ids (copied cells), so fuzzy matching stays as the fallback.
- `LabRun.template` is optional: a run can be built from submissions alone, before the template is attached (M1).
- `core/config.py` (new module): TOML -> typed config, including the `LLMConfig` of section 11 and `scrub.extra_names`.
- Roster is optional. File names are always `<prefix>_<LastName>_<LastName>….ipynb` with every member's last name, so without a roster each file is a group and its students are those last names; the prefix is auto-detected (tokens shared by most files) or set as `ingest.filename_prefix`. Full names as written in the notebook ("- Zemzemi Mohamed") are harvested from the last names, used as display names and scrubbed. With a roster (CSV, `,` or `;`, columns `group`, `student`, optional `file`), files are attributed by the name tokens in their path.
- Scrubbing: staff names (teachers, assistants) appear in every notebook and are left as they are (decided Nov 11). `scrub.extra_names` exists if that changes.

### M2 (Nov 12, 2026)

- **Sections come from exercise numbers, not only headings.** The TP2 template has four `##` parts; the exercises are bold numbers at the top of a cell (`**3.1)`, `<b>1.1)</b>`). Both open a section (`template.item_pattern`); ids come from the number (`S3.1`, `S4`). Only sections with a task or question cell are judged for completeness (17 in TP2). Task cells: code cells holding only comments, or a comment with a task marker ("votre code ici"); question cells: markdown cells with an answer slot (`*Réponse:*`, `*Description head():*`). All markers are config (`[template]`); `extra_task_cells` lists cells no marker catches (TP2 4.7, "modify one line").
- `Cell.hidden` (new field): **past submissions contain the assistants' grading** ("Corrections Partie 1: Points obtenus: 4/5, Remarques: …"). Cells with a staff marker are kept on disk but never serialised into a prompt and never checked. They are a ready-made annotation for `tp2te eval` (M5).
- Matching uses a global monotone alignment (weighted LCS over a cell similarity matrix, nbformat id as the heaviest anchor) instead of greedy matching followed by a longest increasing subsequence. Same intent as section 6 steps 2 to 4, order-consistent by construction; out-of-order cells with an id match or similarity ≥ 0.9 are kept as moved anchors.
- Answer detection by subtraction also looks at the added cells right after a task or question cell, since students often answer in a new cell below.
- `core/prompts.py` (new module): renders the Jinja prompts, versions them by file hash (no manual bump), and is the single path to the LLM (`call_llm`). It refuses to send a prompt containing a roster name, email or home path (a guard independent of the scrubber) and writes every call, including failures, to `llm_log.jsonl`.
- `llm/fake.py` also holds `CachingClient`: real calls are recorded under `runs/.llm_cache/` in the format `FakeClient` replays, so a repeated run costs nothing and a real run can be replayed offline.
- The grounding check lives in `core/questions.py` from M2 on, because L1 issues already need it.
- CLI: `tp2te report <run_dir>` reprints the completeness table and issue report; `run` gains `--solution`, `--brief`, `--groups`, `--max-groups`, `--no-llm`.
- Measured sizes on TP2 (replacing the assumptions of section 12): serialised notebook about 50k characters, template 33k, solution 62k; an L1 prompt with the solution is about 115k characters, roughly 30k tokens, against 21k assumed.
