"""All Pydantic models. This file is the contract of SPEC.md section 5.

Deviations from the spec, forced by the real notebooks (see SPEC.md section 16):
- ``Cell.nb_cell_id``: the nbformat cell id. Template ids survive in most
  submitted cells, which makes them the strongest matching anchor.
- ``LabRun.template`` is optional: a run can be created from submissions and a
  roster before the template is available (M1).
- ``Cell.hidden``: staff cells (assistants' grading, "Points obtenus: 4/5") are
  kept on disk but never serialised into a prompt and never checked (M2).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


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
    nb_cell_id: str | None = None     # nbformat id as found in the file (spec change)
    hidden: bool = False              # staff cell: never in a prompt (spec change)


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
    template: Template | None = None   # spec change: optional until a template is given
    brief: LabBrief | None = None
    lab_pool: QuestionPool | None = None
    students: list[Student]
    groups: dict[str, GroupRun]
    assignments: list[StudentAssignment] = []
    assignments_locked: bool = False   # set when sheets are printed
    grades: list[Grade] = []
    summary: list[SummaryItem] = []
