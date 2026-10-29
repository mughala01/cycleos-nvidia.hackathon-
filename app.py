# CycleOS — standalone setup
# Run this cell first in the new notebook. It restores local history when available,
# defines save_history(), and configures Nebius + NVIDIA Nemotron.

import os, json
from getpass import getpass
from openai import OpenAI

HISTORY_FILE = "/tmp/cycleos_history.json"
def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                data = {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = {}
    for key in ("check_ins", "tasks", "task_history", "calendar_events", "cycle_data", "emotional_notes", "generated_plans", "conversations", "daily_anchors", "capacity_packages", "life_tasks"):
        data.setdefault(key, [])
    return data

history = load_history()
def save_history():
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, HISTORY_FILE)

api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
if not api_key:
    raise ValueError("A Nebius API key is required. Re-run this cell and enter your key.")
client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=api_key)
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
print("CycleOS setup ready. Saved open tasks:", sum(1 for t in history.get("tasks", []) if isinstance(t, dict) and not t.get("completed")))


# CycleOS — audited dashboard implementation
# Run AFTER the cell that creates `client`, `MODEL_ID`, `history`, and `save_history`.
# This cell replaces the old dashboard foundation and does not run any input() loops.

import json
import re
import uuid
import copy
import csv
import os
from datetime import date, datetime
import gradio as gr

# Keep the existing notebook history; initialise only missing sections.
for _key in ["check_ins", "tasks", "task_history", "generated_plans", "emotional_notes", "conversations", "daily_anchors", "calendar_events"]:
    if not isinstance(history.get(_key), list):
        history[_key] = []
if not isinstance(history.get("user_preferences"), dict):
    history["user_preferences"] = {}
if not isinstance(history.get("capacity_packages"), dict):
    _old_capacity_packages = history.get("capacity_packages")
    history["capacity_packages"] = {}
    if _old_capacity_packages:
        history.setdefault("legacy_capacity_packages", _old_capacity_packages)

# Preferences persist in the existing history file, so daily use does not repeat setup.
_CYCLEOS_DEFAULT_PREFS = {
    "capacity_definition": "Use the time I say I have; low capacity changes how I work, not whether important work matters.",
    "work_block_minutes": 25,
    "break_minutes": 5,
    "default_available_minutes": 120,
    "cycle_tracking_mode": "off",
    "last_period_start": "",
    "average_cycle_length": 28,
    "average_period_length": 5,
    "cycle_context_note": "",
    "setup_complete": False,
}
for _k, _v in _CYCLEOS_DEFAULT_PREFS.items():
    history["user_preferences"].setdefault(_k, _v)


def _cycleos_save():
    """Use the notebook's existing save function; never replace the history file."""
    save_history()


def _cycleos_json(content):
    """Parse a JSON object from common model response formats; fail safely."""
    if isinstance(content, dict):
        return content
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The model did not return a JSON object.")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    return data


def _cycleos_call(system_prompt, payload, max_tokens=1800):
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ],
        temperature=0.25,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("The model returned an empty response. Nothing new was saved.")
    return _cycleos_json(text)


def _cycleos_clean_lines(value):
    return [line.strip(" •\t-") for line in (value or "").splitlines() if line.strip(" •\t-")]


def _cycleos_valid_iso_date(value):
    """Return a real YYYY-MM-DD date, or None; regex alone accepts impossible dates."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed.isoformat() if parsed.isoformat() == value else None


def _cycleos_learned_duration(task_id, fallback=None):
    """Return a conservative median from real outcomes for this exact stable task ID."""
    eligible = {"completed", "partial", "interrupted"}
    observations = []
    for event in history.get("task_history", []):
        if not isinstance(event, dict) or event.get("task_id") != task_id:
            continue
        if event.get("status") not in eligible:
            continue
        actual = event.get("actual_minutes")
        if isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual > 0:
            observations.append(int(actual))
    recent = observations[-5:]
    if len(recent) < 2:
        return {
            "learned_estimated_minutes": fallback,
            "duration_evidence_count": len(recent),
            "duration_learning_note": "Not enough real outcomes to adjust duration yet."
        }
    ordered = sorted(recent)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else round((ordered[middle - 1] + ordered[middle]) / 2)
    return {
        "learned_estimated_minutes": median,
        "duration_evidence_count": len(recent),
        "duration_learning_note": f"Based on {len(recent)} recent recorded outcomes; user estimate is not overwritten."
    }


def _cycleos_existing_open_tasks():
    result = []
    for item in history.get("tasks", []):
        if not isinstance(item, dict) or item.get("completed"):
            continue
        title = str(item.get("title") or item.get("task") or "").strip()
        if not title:
            continue
        task_id = str(item.get("id") or ("task-" + uuid.uuid4().hex[:10]))
        item["id"] = task_id
        result.append({
            "id": task_id,
            "title": title,
            "due_date": item.get("due_date") or item.get("deadline"),
            "estimated_minutes": item.get("estimated_minutes") or item.get("estimated_time_minutes"),
            **_cycleos_learned_duration(
                task_id,
                item.get("estimated_minutes") or item.get("estimated_time_minutes")
            ),
            "category": item.get("category") or "other",
            "source": "saved task",
        })
    return result


EXTRACTION_PROMPT = """You are CycleOS, a careful everyday-life planning assistant. Extract actionable tasks and dated events from the user's text. This is for the whole of life, not just studying: include work, studying, cooking/food, exercise, cleaning/home, errands/admin, social plans, appointments, rest/self-care, and personal projects. Do not turn feelings, worries, or vague hopes into tasks unless a concrete action is stated. Do not invent deadlines, event dates, or estimates. Return ONLY valid JSON: {\"emotional_support\":\"one concise validating sentence if useful, otherwise empty\",\"tasks\":[{\"title\":\"short actionable title\",\"due_date\":null,\"estimated_minutes\":null,\"priority\":\"high|normal|low\",\"category\":\"work|study|cooking_food|exercise|home|admin|social|appointment|self_care|personal_project|other\",\"evidence\":\"brief exact phrase supporting this task\"}]}. Use due_date for either a deadline OR the date of an event/plan. Dates must be YYYY-MM-DD only when the user gave enough information to determine the date; otherwise null. Estimated minutes must be a positive integer only when stated or strongly explicit; otherwise null. Never invent facts."""

PLANNING_PROMPT = """You are CycleOS, a capacity-aware EXECUTION planner. Use user_preferences.capacity_definition, preferred_work_block_minutes and preferred_break_minutes as the user's personal defaults. Do not make the user repeat saved preferences or saved open tasks. Daily interaction should require only a short brain dump and today's available time. Plan across the whole of everyday life, not just studying or work: cooking and eating, workouts, household tasks, errands, social plans, appointments, rest/self-care and personal projects are all valid tasks. Protect wellbeing without abandoning the user's real responsibilities. Do not shame, diagnose, or assume cycle phase determines ability. Cycle information is optional personal context: use it only if the user has enabled it, give gentle advance heads-ups for dated plans that may overlap an estimated period window, suggest practical preparation choices, and never assume the user will feel unwell or automatically cancel/reschedule anything. Use only task IDs supplied in the payload; never invent IDs, tasks, dates, or durations. Never schedule tasks marked completed=true, and respect recorded_outcomes when adapting a plan. If learned_estimated_minutes has at least two outcomes, use it only as a tentative hint; never claim learning from less evidence. Respect the user's chosen day mode and never schedule more task minutes than available minutes. Leave 20% unscheduled when possible, but do not use that reserve rule to drop a genuinely urgent fixed-deadline deliverable when the user explicitly needs to finish it; instead reduce scope and use short focused blocks.

CRITICAL DEADLINE RULE: If the user has a fixed, near, or non-negotiable deadline, do NOT respond with only rest advice, telling them to stop, or asking them to confirm a deadline they have already stated. Low capacity changes HOW a task is done, not WHETHER it matters. Build a realistic deadline execution plan: identify the minimum submission-ready version; do the highest-value existing work first; polish/fix what already exists; cut optional scope; leave new ideas/features out; include a concrete sequence of actions, timeboxes, and a clear done condition inside each action. Make the first step an actual work action, not generic advice. Treat breaks as supports around work, not substitutes for work. Never ask the user to open another app just to re-check a deadline already given. Use saved tasks, anchors, events and prior conversation first; ask only for missing information that genuinely blocks planning. Use the user's saved capacity definition, work-block length and break preference. Never recommend pushing through serious immediate danger, but ordinary low energy/irritability alone is not a reason to refuse to help plan.

On a low-capacity day WITHOUT a near fixed deadline, keep the plan small and defer non-urgent work. Put deadline-critical work in must_protect, optional improvements in if_capacity, and non-essential work in can_wait. Do not add project features or new ideas when the user asks to polish/finish. If the deadline or required deliverable is genuinely unknown, state the uncertainty briefly while still making progress on known work. Return ONLY valid JSON with this shape: {\"day_mode\":\"short label\",\"what_is_weighing_on_you\":\"one short sentence\",\"must_protect\":[{\"task_id\":\"ID from input\",\"action\":\"specific executable action with a done condition\",\"minutes\":10}],\"if_capacity\":[{\"task_id\":\"ID from input\",\"action\":\"specific optional action\",\"minutes\":10}],\"can_wait\":[\"ID from input\"],\"one_next_step\":\"the first concrete action to start now\",\"reasoning_note\":\"brief scope/time trade-off\"}. Every task reference must be one of the provided IDs. Each minutes value must be a positive integer. Keep it realistic and concise."""


def _cycleos_validate_plan(plan, tasks, available_minutes, reserve_buffer=True):
    """Validate IDs, remove duplicate assignments, and enforce a hard time budget.

    Keep a 20% flexibility buffer by default. For a clearly detected fixed/near
    deadline, the caller may disable that buffer while the hard total limit remains.
    """
    if not isinstance(plan, dict):
        raise ValueError("The planner did not return a JSON object.")
    valid_ids = {
        t["id"] for t in tasks
        if isinstance(t, dict) and t.get("id") and not t.get("completed", False)
    }
    cleaned = dict(plan)
    warnings = []
    cleaned["day_mode"] = str(cleaned.get("day_mode") or "Capacity-aware")[:80]
    cleaned["what_is_weighing_on_you"] = str(cleaned.get("what_is_weighing_on_you") or "")[:500]
    cleaned["one_next_step"] = str(cleaned.get("one_next_step") or "Start with the first listed action.")[:500]
    cleaned["reasoning_note"] = str(cleaned.get("reasoning_note") or "")[:500]
    available_minutes = max(0, int(available_minutes))
    seen_ids = set()

    # Keep the first occurrence of each task across both scheduled sections.
    for section in ("must_protect", "if_capacity"):
        entries = cleaned.get(section, [])
        if not isinstance(entries, list):
            warnings.append(f"The {section} section was malformed and was omitted.")
            entries = []
        accepted = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("task_id") not in valid_ids:
                warnings.append("An AI-suggested action was removed because it did not match an unfinished task.")
                continue
            task_id = entry["task_id"]
            if task_id in seen_ids:
                warnings.append("A repeated task assignment was removed to prevent double-scheduling.")
                continue
            try:
                raw_minutes = entry.get("minutes", 10)
                if isinstance(raw_minutes, bool):
                    raise ValueError
                minutes = int(raw_minutes)
                if float(raw_minutes) != minutes or minutes <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                minutes = 10
                warnings.append("An invalid duration was replaced with a 10-minute default.")
            accepted.append({
                "task_id": task_id,
                "action": str(entry.get("action") or "Take the next concrete step")[:500],
                "minutes": minutes,
            })
            seen_ids.add(task_id)
        cleaned[section] = accepted

    raw_wait = cleaned.get("can_wait", [])
    if not isinstance(raw_wait, list):
        raw_wait = []
        warnings.append("The can-wait section was malformed and was omitted.")
    scheduled_ids = {x["task_id"] for section in ("must_protect", "if_capacity") for x in cleaned[section]}
    cleaned["can_wait"] = list(dict.fromkeys(
        task_id for task_id in raw_wait
        if isinstance(task_id, str) and task_id in valid_ids and task_id not in scheduled_ids
    ))

    reserve = int(available_minutes * 0.2) if reserve_buffer and available_minutes >= 25 else 0
    task_budget = max(0, available_minutes - reserve)
    used = 0
    for section in ("must_protect", "if_capacity"):
        kept = []
        for entry in cleaned[section]:
            remaining = task_budget - used
            if remaining <= 0:
                if section == "must_protect":
                    warnings.append("There is not enough time for every essential action; review priorities or reduce scope.")
                continue
            if entry["minutes"] > remaining:
                if section == "must_protect":
                    warnings.append("An essential action did not fit the available time and was omitted rather than shown with a misleading shortened duration.")
                continue
            kept.append(entry)
            used += entry["minutes"]
        cleaned[section] = kept

    cleaned["validation_warnings"] = list(dict.fromkeys(warnings))
    cleaned["scheduled_minutes"] = used
    cleaned["available_minutes"] = available_minutes
    cleaned["reserve_minutes"] = reserve
    return cleaned


def _cycleos_cycle_heads_up(tasks):
    """Create opt-in, approximate period-window heads-ups for dated tasks/events."""
    prefs = history.get("user_preferences", {})
    if prefs.get("cycle_tracking_mode") != "manual_estimates":
        return []
    raw_start = str(prefs.get("last_period_start") or "").strip()
    if not raw_start:
        return []
    try:
        last_start = date.fromisoformat(raw_start)
        cycle_len = int(prefs.get("average_cycle_length", 28))
        period_len = int(prefs.get("average_period_length", 5))
    except (ValueError, TypeError):
        return []
    today = date.today()
    # Generate estimated windows for the next 90 days, including the current cycle if relevant.
    windows = []
    start = last_start
    while start + __import__('datetime').timedelta(days=period_len - 1) < today:
        start += __import__('datetime').timedelta(days=cycle_len)
    for _ in range(5):
        end = start + __import__('datetime').timedelta(days=period_len - 1)
        if end >= today:
            windows.append((start, end))
        start += __import__('datetime').timedelta(days=cycle_len)
    heads = []
    for task in tasks:
        if task.get("completed"):
            continue
        raw_date = task.get("due_date")
        if not isinstance(raw_date, str):
            continue
        try:
            event_date = date.fromisoformat(raw_date[:10])
        except ValueError:
            continue
        for start, end in windows:
            if start <= event_date <= end:
                title = str(task.get("title") or "This plan")
                heads.append(f"- **{event_date.strftime('%a %d %b')}: {title}** may overlap your estimated period window ({start.strftime('%d %b')}–{end.strftime('%d %b')}). If useful, decide whether you want to prepare supplies, plan transport/food/rest, or keep things exactly as they are. CycleOS will not assume symptoms or change the plan automatically.")
                break
    return heads


def _cycleos_render_plan(plan, tasks, available_minutes):
    task_names = {t["id"]: t["title"] for t in tasks}
    lines = [f"## Your CycleOS plan", f"**Day mode:** {plan.get('day_mode') or 'Capacity-aware'}", ""]
    if plan.get("what_is_weighing_on_you"):
        lines += [f"> {plan['what_is_weighing_on_you']}", ""]
    cycle_heads = _cycleos_cycle_heads_up(tasks)
    if cycle_heads:
        lines += ["### Cycle heads-up · based on your optional manual settings", *cycle_heads, ""]
    urgent_entries = plan.get("must_protect", [])
    if urgent_entries:
        lines.append("### Do these in order")
        lines.append("Keep the scope tight. Finish the first block before moving to the next.")
        for index, item in enumerate(urgent_entries, 1):
            name = task_names.get(item.get("task_id"), "Task")
            lines.append(f"{index}. **{name}** — {item.get('action', 'Take the next step')} (**{item.get('minutes', 10)} min**) ")
        lines.append("")
    for key, heading in [("if_capacity", "### If there is capacity")]:
        lines.append(heading)
        entries = plan.get(key, [])
        if not entries:
            lines.append("- Nothing extra scheduled here.")
        for item in entries:
            name = task_names.get(item.get("task_id"), "Task")
            lines.append(f"- **{name}** — {item.get('action', 'Take the next step')} ({item.get('minutes', 10)} min)")
        lines.append("")
    lines.append("### Can wait")
    if plan.get("can_wait"):
        lines.extend([f"- {task_names.get(task_id, 'Task')}" for task_id in plan["can_wait"]])
    else:
        lines.append("- No additional tasks explicitly deferred.")
    lines += ["", f"### One next step", str(plan.get("one_next_step") or "Start with the smallest useful action."), ""]
    if plan.get("reasoning_note"):
        lines += [f"_Trade-off: {plan['reasoning_note']}_", ""]
    scheduled = sum(x.get("minutes", 0) for x in plan.get("must_protect", []) + plan.get("if_capacity", []))
    lines.append(f"**Time check:** {scheduled} of {available_minutes} available minutes scheduled; the rest stays flexible.")
    for warning in plan.get("validation_warnings", []):
        lines.append(f"\n> **Review:** {warning}")
    return "\n".join(lines)


def _cycleos_detect_deadline_pressure(text, tasks, day_mode):
    """Detect explicit deadline pressure without inventing a deadline date."""
    combined = (text or "").casefold()
    phrases = (
        "deadline", "due today", "due tomorrow", "due tonight", "submit today",
        "submission", "last minute", "non-negotiable", "can't miss", "cannot miss",
        "must be done", "needs to be finished", "in two days", "in 2 days",
        "tomorrow", "tonight", "by friday", "by monday", "by thursday",
    )
    explicit_pressure = any(phrase in combined for phrase in phrases)
    today = date.today()
    dated_urgent = False
    for task in tasks:
        due = task.get("due_date")
        if isinstance(due, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", due):
            try:
                days = (date.fromisoformat(due) - today).days
                if 0 <= days <= 3 and not task.get("completed", False):
                    dated_urgent = True
                    break
            except ValueError:
                pass
    return {
        "near_deadline_detected": bool(explicit_pressure or dated_urgent),
        "deadline_detection_basis": "explicit user wording" if explicit_pressure else ("task due date within 3 days" if dated_urgent else "no near deadline detected from supplied details"),
        "deadline_instruction": "Build an executable, scope-reduced completion plan; do not replace work with generic rest advice. Polish existing work, add no new features, and define a clear done condition." if (explicit_pressure or dated_urgent) else "Use normal capacity-aware planning.",
    }


def create_plan(energy, focus, mood, hours, extra_minutes, day_mode, brain_dump, task_text, session):
    session = dict(session or {})
    try:
        available = max(0, min(1440, int(float(hours or 0) * 60) + int(extra_minutes or 0)))
    except (TypeError, ValueError):
        return "Please enter a valid number of available hours.", session, ""
    today = date.today().isoformat()
    manual_lines = _cycleos_clean_lines(task_text)
    dump = (brain_dump or "").strip()

    # Step 1: extract first. IDs exist BEFORE the planner receives tasks.
    extracted = {"emotional_support": "", "tasks": []}
    if dump:
        try:
            extracted = _cycleos_call(EXTRACTION_PROMPT, {"today": today, "text": dump}, max_tokens=900)
        except Exception as exc:
            return f"**I couldn't safely generate a plan yet.** No new plan was saved.\n\nDetails: `{type(exc).__name__}: {exc}`", session, ""

    tasks = _cycleos_existing_open_tasks()
    known_titles = {t["title"].casefold() for t in tasks}
    new_tasks = []
    for line in manual_lines:
        if line.casefold() not in known_titles:
            task = {
                "id": "manual-" + uuid.uuid4().hex[:8],
                "title": line,
                "due_date": None,
                "estimated_minutes": None,
                "category": "other",
                "source": "manual input",
            }
            tasks.append(task)
            new_tasks.append(task)
            known_titles.add(line.casefold())
    for item in extracted.get("tasks", []) if isinstance(extracted.get("tasks", []), list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title or title.casefold() in known_titles:
            continue
        evidence = str(item.get("evidence") or "").strip()
        if evidence and evidence.casefold() not in dump.casefold():
            # The model's evidence must actually appear in the user's input.
            continue
        task = {
            "id": "extracted-" + uuid.uuid4().hex[:8],
            "title": title[:180],
            "due_date": _cycleos_valid_iso_date(item.get("due_date")),
            "estimated_minutes": item.get("estimated_minutes") if isinstance(item.get("estimated_minutes"), int) and not isinstance(item.get("estimated_minutes"), bool) and item.get("estimated_minutes") > 0 else None,
            "learned_estimated_minutes": None,
            "duration_evidence_count": 0,
            "duration_learning_note": "No recorded outcomes for this new task yet.",
            "priority": item.get("priority") if item.get("priority") in {"high", "normal", "low"} else "normal",
            "category": item.get("category") if item.get("category") in {"work", "study", "cooking_food", "exercise", "home", "admin", "social", "appointment", "self_care", "personal_project", "other"} else "other",
            "source": "brain dump",
            "evidence": evidence,
        }
        tasks.append(task)
        new_tasks.append(task)
        known_titles.add(title.casefold())

    # Keep the active prompt bounded while retaining the user's saved task inventory.
    # Tell the user when the context cap means not every open task was sent to the model.
    plan_tasks = tasks[:60]
    omitted_task_count = max(0, len(tasks) - len(plan_tasks))
    deadline_context = _cycleos_detect_deadline_pressure((dump or "") + "\n" + (task_text or ""), plan_tasks, day_mode)
    payload = {
        "today": today,
        "check_in": {"energy": int(energy), "focus": int(focus), "mood": mood, "available_minutes": available, "day_mode": day_mode},
        "brain_dump": dump,
        "emotional_support": str(extracted.get("emotional_support") or ""),
        "tasks": plan_tasks,
        "deadline_context": deadline_context,
        "daily_anchors": history.get("daily_anchors", []),
        "capacity_packages": history.get("capacity_packages", {}),
        "user_preferences": history.get("user_preferences", {}),
        "preferred_work_block_minutes": history.get("user_preferences", {}).get("work_block_minutes", 25),
        "preferred_break_minutes": history.get("user_preferences", {}).get("break_minutes", 5),
        "capacity_definition": history.get("user_preferences", {}).get("capacity_definition", ""),
        "upcoming_events": [e for e in history.get("calendar_events", []) if isinstance(e, dict)][:25],
        "optional_cycle_context": {
            "enabled": history.get("user_preferences", {}).get("cycle_tracking_mode") == "manual_estimates",
            "last_period_start": history.get("user_preferences", {}).get("last_period_start", ""),
            "average_cycle_length_days": history.get("user_preferences", {}).get("average_cycle_length", 28),
            "average_period_length_days": history.get("user_preferences", {}).get("average_period_length", 5),
            "user_note": history.get("user_preferences", {}).get("cycle_context_note", ""),
            "instruction": "Only offer gentle date-based preparation heads-ups; never infer symptoms, ability, mood, or need to cancel plans from cycle dates. The user decides what to do."
        },
    }
    try:
        raw_plan = _cycleos_call(PLANNING_PROMPT, payload, max_tokens=1600)
        plan = _cycleos_validate_plan(
            raw_plan, plan_tasks, available,
            reserve_buffer=not deadline_context["near_deadline_detected"],
        )
    except Exception as exc:
        return f"**I couldn't safely generate a plan yet.** No new plan was saved.\n\nDetails: `{type(exc).__name__}: {exc}`", session, ""

    check_in = {"date": today, "energy": int(energy), "focus": int(focus), "mood": mood, "available_minutes": available, "day_mode": day_mode}
    # Keep rollback points so a failed disk write does not leave a half-saved history in memory.
    history_lengths = {key: len(history.setdefault(key, [])) for key in ("check_ins", "emotional_notes", "tasks", "generated_plans")}
    if omitted_task_count:
        plan.setdefault("validation_warnings", []).append(
            f"{omitted_task_count} older open task(s) were not included in this plan because the planning context is capped at 60 tasks."
        )
    history["check_ins"].append(check_in)
    if dump:
        history["emotional_notes"].append({"date": today, "text": dump, "source": "dashboard"})
    # Persist newly extracted tasks only after a successful plan; they already have IDs referenced by that plan.
    for task in new_tasks:
        history["tasks"].append({**task, "completed": False, "created_date": today})
    plan_record = {"id": "plan-" + uuid.uuid4().hex[:10], "date": today, "check_in": check_in, "task_ids": [t["id"] for t in plan_tasks], "plan": plan, "brain_dump": dump, "available_minutes": available, "source": "dashboard"}
    history["generated_plans"].append(plan_record)
    try:
        _cycleos_save()
    except Exception as exc:
        for key, old_len in history_lengths.items():
            del history[key][old_len:]
        return f"**I generated a plan, but could not save it.** History was rolled back in memory. `{type(exc).__name__}: {exc}`", session, ""

    session["active_plan"] = plan_record
    session["tasks"] = plan_tasks
    session["available_minutes"] = available
    session["update_log"] = []
    rendered = _cycleos_render_plan(plan, plan_tasks, available)
    return rendered, session, "Plan saved to your notebook history."


def update_plan(update_text, remaining_minutes, session):
    """Update the current plan against an explicit new time budget, saving atomically in memory."""
    session = dict(session or {})
    active = session.get("active_plan")
    if not active:
        return "Create a plan first; there is no active plan to update.", session
    update_text = (update_text or "").strip()
    if not update_text:
        return "Tell CycleOS what changed first.", session

    old_available = int(session.get("available_minutes", active.get("available_minutes", 0)) or 0)
    if remaining_minutes not in (None, ""):
        try:
            new_available = int(float(remaining_minutes))
            if float(remaining_minutes) != new_available or not 0 <= new_available <= 1440:
                raise ValueError
        except (TypeError, ValueError):
            return "Time left must be a whole number from 0 to 1,440 minutes.", session
        budget_note = f"The user explicitly set the new remaining time budget to {new_available} minutes. This overrides the previous budget."
    else:
        # Safely infer only clear statements, never arbitrary durations mentioned in passing.
        match = re.search(r"\b(?:only\s+)?(\d{1,4})\s*(?:minutes?|mins?)\s*(?:left|remaining|available|to go)\b", update_text, flags=re.I)
        if not match:
            match = re.search(r"\b(?:i\s+)?(?:only\s+)?(?:have|got)\s+(\d{1,4})\s*(?:minutes?|mins?)\s*(?:left|remaining|available|to go)\b", update_text, flags=re.I)
        if match:
            new_available = int(match.group(1))
            if new_available > 1440:
                return "I found a time budget above 24 hours. Please enter a valid value in the Time left field.", session
            budget_note = f"The user's update clearly states {new_available} minutes remain. Use this as the new budget, replacing the previous {old_available} minutes."
        else:
            new_available = old_available
            budget_note = f"No new time budget was specified, so retain the current budget of {old_available} minutes."

    try:
        previous = copy.deepcopy(active["plan"])
        tasks = session.get("tasks", [])
        deadline_context = _cycleos_detect_deadline_pressure(
            update_text + "\n" + str(active.get("brain_dump", "")), tasks,
            active.get("check_in", {}).get("day_mode", "auto")
        )
        payload = {
            "today": date.today().isoformat(),
            "previous_plan": previous,
            "tasks": tasks,
            "deadline_context": deadline_context,
            "recorded_outcomes": [row for row in history.get("task_history", []) if isinstance(row, dict) and row.get("task_id") in {t.get("id") for t in tasks}][-30:],
            "user_preferences": history.get("user_preferences", {}),
            "capacity_definition": history.get("user_preferences", {}).get("capacity_definition", ""),
            "preferred_work_block_minutes": history.get("user_preferences", {}).get("work_block_minutes", 25),
            "preferred_break_minutes": history.get("user_preferences", {}).get("break_minutes", 5),
            "available_minutes": new_available,
            "update_from_user": update_text,
            "time_budget_instruction": budget_note,
            "instruction": "Adapt the existing plan to the new reality. Preserve task IDs. Do not restart from scratch or assume completed tasks are undone. The new available_minutes is a hard upper limit. If a near/fixed deadline is present, make an ordered execution plan for the minimum submission-ready outcome: polish/fix existing work, cut optional scope, add no new ideas/features, state a concrete done condition, and do not substitute generic rest advice for work. Keep steps timeboxed. Never schedule more than the supplied time budget; leave at least 20% unscheduled when possible unless the explicit fixed-deadline context requires using that time.",
        }
        raw = _cycleos_call(PLANNING_PROMPT, payload, max_tokens=1300)
        updated = _cycleos_validate_plan(
            raw, tasks, new_available,
            reserve_buffer=not deadline_context["near_deadline_detected"],
        )

        # Stage changes first. If persistence fails, leave the prior session plan intact.
        candidate_active = copy.deepcopy(active)
        candidate_active["plan"] = updated
        candidate_active["available_minutes"] = new_available
        candidate_active.setdefault("updates", []).append({
            "at": datetime.now().isoformat(timespec="seconds"),
            "text": update_text,
            "available_minutes": new_available,
        })
        plan_snapshot = {**candidate_active, "id": "plan-" + uuid.uuid4().hex[:10], "date": date.today().isoformat(), "source": "dashboard_update"}
        generated_len = len(history.setdefault("generated_plans", []))
        conversations_len = len(history.setdefault("conversations", []))
        history["generated_plans"].append(plan_snapshot)
        history["conversations"].append({"date": date.today().isoformat(), "user": update_text, "assistant": "Updated the existing plan.", "source": "dashboard"})
        try:
            _cycleos_save()
        except Exception:
            del history["generated_plans"][generated_len:]
            del history["conversations"][conversations_len:]
            raise

        session["active_plan"] = candidate_active
        session["available_minutes"] = new_available
        return _cycleos_render_plan(updated, tasks, new_available) + "\n\n_Updated from your existing plan._", session
    except Exception as exc:
        return f"Update failed; your previous plan remains active. `{type(exc).__name__}: {exc}`", session


def record_outcome(task_id, status, actual_minutes, note, session):
    session = dict(session or {})
    active = session.get("active_plan")
    if not active:
        return "Create a plan first.", session
    tasks = session.get("tasks", [])
    selected = next((t for t in tasks if t.get("id") == task_id), None)
    if not selected:
        return "That task is not in the current plan. Nothing was recorded.", session
    if status not in {"completed", "partial", "interrupted", "deferred"}:
        return "Choose a valid outcome status. Nothing was recorded.", session
    try:
        minutes = int(actual_minutes) if actual_minutes not in (None, "") else None
        if minutes is not None and (minutes < 0 or float(actual_minutes) != minutes):
            raise ValueError
    except (TypeError, ValueError):
        return "Actual minutes must be a non-negative whole number.", session

    event = {
        "date": date.today().isoformat(),
        "task_id": task_id,
        "task_title": selected.get("title"),
        "status": status,
        "planned_minutes": next((x.get("minutes") for section in ("must_protect", "if_capacity") for x in active["plan"].get(section, []) if x.get("task_id") == task_id), None),
        "actual_minutes": minutes,
        "note": (note or "").strip(),
        "plan_id": active.get("id"),
    }
    previous_selected_completed = selected.get("completed", False)
    saved_matches = [saved for saved in history.get("tasks", []) if isinstance(saved, dict) and saved.get("id") == task_id]
    previous_saved_states = [(saved, saved.get("completed"), saved.get("completed_date")) for saved in saved_matches]
    history.setdefault("task_history", []).append(event)
    if status == "completed":
        selected["completed"] = True
        for saved in saved_matches:
            saved["completed"] = True
            saved["completed_date"] = date.today().isoformat()
    else:
        selected["last_outcome"] = status
        selected["last_outcome_note"] = (note or "").strip()

    try:
        _cycleos_save()
    except Exception as exc:
        history["task_history"].pop()
        if previous_selected_completed:
            selected["completed"] = previous_selected_completed
        else:
            selected.pop("completed", None)
        for saved, old_completed, old_completed_date in previous_saved_states:
            if old_completed is None:
                saved.pop("completed", None)
            else:
                saved["completed"] = old_completed
            if old_completed_date is None:
                saved.pop("completed_date", None)
            else:
                saved["completed_date"] = old_completed_date
        return f"Could not save the outcome, so no change was kept. `{type(exc).__name__}`", session

    count = sum(
        1 for row in history.get("task_history", [])
        if isinstance(row, dict)
        and row.get("task_id") == task_id
        and row.get("status") in {"completed", "partial", "interrupted"}
        and isinstance(row.get("actual_minutes"), (int, float))
        and not isinstance(row.get("actual_minutes"), bool)
        and row.get("actual_minutes") > 0
    )
    learning_message = (
        " CycleOS can now use the recent median as a tentative planning hint."
        if count >= 2 else
        f" {count}/2 timed outcomes recorded; CycleOS will wait for more evidence before adapting its duration hint."
    )
    if status == "completed":
        outcome_message = " This task is marked complete and will be excluded from the next plan update."
    else:
        outcome_message = " This outcome will be available as context when you update the plan."
    return f"Recorded **{status}** for **{selected.get('title')}**.{outcome_message}{learning_message} Your saved estimate is never silently overwritten.", session


def _cycleos_parse_setup_task(line):
    """Store a setup task and parse common explicit deadline formats without an API call."""
    raw = line.strip(" •\t-")
    if not raw:
        return None
    due = None
    patterns = [
        (r"\b(\d{4}-\d{2}-\d{2})\b", "%Y-%m-%d"),
        (r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b", "%d %B %Y"),
        (r"\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\b", "%d %b %Y"),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, raw)
        if match:
            try:
                due = datetime.strptime(match.group(1), fmt).date().isoformat()
                break
            except ValueError:
                pass
    return {
        "id": "setup-" + uuid.uuid4().hex[:8],
        "title": raw[:180],
        "due_date": due,
        "estimated_minutes": None,
        "priority": "normal",
        "source": "one-time setup",
        "completed": False,
        "created_date": date.today().isoformat(),
    }


def import_cycle_data_file(file_path):
    """Import only period-start dates from a user-selected CSV/JSON export; never retain the raw file."""
    if not file_path:
        return "Choose a CSV or JSON export first. Nothing was changed."
    try:
        path = str(file_path)
        raw_dates = []
        accepted_keys = {
            "period_start", "period start", "period_start_date", "period start date",
            "start_date", "start date", "menstruation_start", "menstruation start",
            "periodstart", "periodstartdate"
        }
        def add_candidate(value):
            if not isinstance(value, str):
                return
            value = value.strip()[:10]
            try:
                parsed = date.fromisoformat(value)
            except ValueError:
                return
            if parsed <= date.today():
                raw_dates.append(parsed)
        if path.lower().endswith(".csv"):
            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    return "I couldn't find column headers in that CSV. Nothing was changed."
                matched = [name for name in reader.fieldnames if name.strip().casefold().replace("-", "_") in accepted_keys or name.strip().casefold().replace("-", "_").replace("_", " ") in accepted_keys]
                if not matched:
                    return "I couldn't identify a period-start date column. Try a CSV with a column named period_start or start_date. Nothing was changed."
                for row in reader:
                    for key in matched:
                        add_candidate(row.get(key))
        elif path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
            def walk(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        normalized = str(key).strip().casefold().replace("-", "_")
                        if normalized in accepted_keys or normalized.replace("_", " ") in accepted_keys:
                            if isinstance(value, list):
                                for item in value:
                                    add_candidate(item)
                            else:
                                add_candidate(value)
                        else:
                            walk(value)
                elif isinstance(obj, list):
                    for item in obj:
                        walk(item)
            walk(payload)
        else:
            return "Please choose a .csv or .json export. Nothing was changed."
        starts = sorted(set(raw_dates))
        if not starts:
            return "No usable period-start dates were found. No data was saved; check the export format."
        latest = starts[-1]
        old_preferences = copy.deepcopy(history.get("user_preferences", {}))
        prefs = history.setdefault("user_preferences", {})
        prefs["cycle_tracking_mode"] = "manual_estimates"
        prefs["last_period_start"] = latest.isoformat()
        if len(starts) >= 2:
            recent_starts = starts[-6:]
            intervals = [(b - a).days for a, b in zip(recent_starts, recent_starts[1:]) if 20 <= (b-a).days <= 45]
            if intervals:
                prefs["average_cycle_length"] = round(sorted(intervals)[len(intervals)//2])
        prefs["cycle_imported_at"] = datetime.now().isoformat(timespec="seconds")
        # Store only the latest start and derived cycle length, not the file or the full history.
        try:
            _cycleos_save()
        except Exception:
            history["user_preferences"] = old_preferences
            raise
        return (f"**Cycle dates imported.** Latest period start saved: {latest.isoformat()}. "
                f"Read {len(starts)} distinct past period-start date(s). Cycle heads-ups are enabled; "
                "dates are estimates, and CycleOS will not assume symptoms or change plans automatically. "
                "The raw export was not retained.")
    except Exception as exc:
        return f"Import failed; no success is claimed. Check the file format. `{type(exc).__name__}: {exc}`"


def save_cycleos_setup(capacity_definition, work_block, break_minutes, default_minutes, anchors_text, saved_tasks_text, cycle_mode, last_period_start, cycle_length, period_length, cycle_note):
    """Persist setup once; future check-ins reuse these preferences and saved commitments."""
    try:
        block = int(work_block)
        pause = int(break_minutes)
        daily_minutes = int(float(default_minutes))
        if block not in {15, 20, 25, 30, 45, 50, 60}:
            raise ValueError("Choose one of the available work-block lengths.")
        if pause not in {0, 5, 10, 15}:
            raise ValueError("Choose one of the available break lengths.")
        if not 0 <= daily_minutes <= 1440:
            raise ValueError("Usual available time must be between 0 and 1,440 minutes.")
        mode = cycle_mode if cycle_mode in {"off", "manual_estimates"} else "off"
        start_text = (last_period_start or "").strip()
        if start_text and _cycleos_valid_iso_date(start_text) is None:
            raise ValueError("Last period start must be a real date in YYYY-MM-DD format, or be left blank.")
        cycle_len = int(float(cycle_length or 28))
        period_len = int(float(period_length or 5))
        if not 20 <= cycle_len <= 45:
            raise ValueError("Average cycle length must be between 20 and 45 days.")
        if not 1 <= period_len <= 10:
            raise ValueError("Average period length must be between 1 and 10 days.")
        old_preferences = copy.deepcopy(history.get("user_preferences", {}))
        old_anchors = copy.deepcopy(history.get("daily_anchors", []))
        old_tasks_length = len(history.get("tasks", []))
        prefs = {
            **history.get("user_preferences", {}),
            "capacity_definition": (capacity_definition or "").strip() or _CYCLEOS_DEFAULT_PREFS["capacity_definition"],
            "cycle_tracking_mode": mode,
            "last_period_start": start_text if mode == "manual_estimates" else "",
            "average_cycle_length": cycle_len,
            "average_period_length": period_len,
            "cycle_context_note": (cycle_note or "").strip()[:500] if mode == "manual_estimates" else "",
            "work_block_minutes": block,
            "break_minutes": pause,
            "default_available_minutes": daily_minutes,
            "setup_complete": True,
            "setup_updated": date.today().isoformat(),
        }
        anchors = [line.strip(" •\t-") for line in (anchors_text or "").splitlines() if line.strip(" •\t-")]
        old_anchors = history.get("daily_anchors", [])
        history["user_preferences"] = prefs
        history["daily_anchors"] = anchors
        known = {str(t.get("title", "")).casefold() for t in history.get("tasks", []) if isinstance(t, dict)}
        added = 0
        for line in (saved_tasks_text or "").splitlines():
            task = _cycleos_parse_setup_task(line)
            if task and task["title"].casefold() not in known:
                history["tasks"].append(task)
                known.add(task["title"].casefold())
                added += 1
        try:
            _cycleos_save()
        except Exception:
            history["user_preferences"] = old_preferences
            history["daily_anchors"] = old_anchors
            del history["tasks"][old_tasks_length:]
            raise
        cycle_msg = "Cycle heads-ups are off." if mode == "off" else ("Manual cycle context saved; date estimates are approximate and won't change plans automatically." if start_text else "Manual cycle context enabled; add your last period start date later to generate date-based heads-ups.")
        return f"**Setup saved.** Your capacity preferences and {len(anchors)} anchors are saved; added {added} new task(s). {cycle_msg} Next time, just brain-dump what changed."
    except Exception as exc:
        return f"Setup could not be saved: `{type(exc).__name__}: {exc}`"


_prefs = history.get("user_preferences", {})
_default_minutes = int(_prefs.get("default_available_minutes", 120) or 0)
_default_hours = _default_minutes // 60
_default_extra = _default_minutes % 60
_setup_anchors = "\n".join(str(x) for x in history.get("daily_anchors", []) if x)
_setup_tasks = "\n".join(
    str(t.get("title", "")) for t in history.get("tasks", [])
    if isinstance(t, dict) and not t.get("completed") and t.get("source") == "one-time setup"
)

CYCLEOS_CSS = """
:root { --cycle-ink: #3d3028; --cycle-muted: #786b60; --cycle-terracotta: #b96543; --cycle-cinnamon: #9b4f35; --cycle-oat: #eee2d0; --cycle-paper: #f8f1e7; --cycle-olive: #777653; }
.gradio-container { background: var(--cycle-paper) !important; color: var(--cycle-ink) !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3 { color: #75462f !important; }
.gradio-container button.primary { background: #a95f3f !important; border-color: #a95f3f !important; color: #fffaf3 !important; }
.gradio-container button.secondary { background: #e9d8c2 !important; border-color: #d9c3a8 !important; color: #604431 !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select { border-color: #d9c7b2 !important; }
.gradio-container .block, .gradio-container .panel { box-shadow: none !important; border-color: #e4d4c1 !important; }
.gradio-container .tabs button { color: #75462f !important; }

/* Keep generated Markdown legible against cream panels in light and dark themes. */
.gradio-container .prose, .gradio-container .prose *,
.gradio-container .markdown, .gradio-container .markdown *,
.gradio-container [data-testid="markdown"], .gradio-container [data-testid="markdown"] *,
.gradio-container .md, .gradio-container .md * {
  color: #3d3028 !important;
}
.gradio-container .prose a, .gradio-container .markdown a,
.gradio-container [data-testid="markdown"] a { color: #914425 !important; }
.gradio-container .prose code, .gradio-container .markdown code { color: #542f24 !important; }

"""
with gr.Blocks(title="CycleOS") as cycleos_app:
    gr.Markdown("# CycleOS\n### Your life, organised around your real capacity.\nStudy, work, cooking, movement, home admin, appointments and social plans all belong here. Your saved tasks and preferences carry forward; tell CycleOS what changed and it will build a practical plan without making you reorganise your whole life first.")

    with gr.Accordion("One-time setup · customise how CycleOS works for you", open=not bool(_prefs.get("setup_complete"))):
        gr.Markdown("Set this once. You can change it any time. Add known commitments here so CycleOS can keep them in its saved task list instead of asking you to hunt for them each day.")
        setup_capacity = gr.Textbox(value=str(_prefs.get("capacity_definition", "")), lines=2, label="What does low capacity look like for you?", placeholder="e.g. I can study for 2 hours in 25-minute blocks with 5-minute breaks.")
        with gr.Row():
            setup_block = gr.Dropdown([15, 20, 25, 30, 45, 50, 60], value=int(_prefs.get("work_block_minutes", 25)), label="Usual work block (minutes)")
            setup_break = gr.Dropdown([0, 5, 10, 15], value=int(_prefs.get("break_minutes", 5)), label="Break between blocks (minutes)")
            setup_available = gr.Number(value=_default_minutes, minimum=0, maximum=1440, step=15, label="Usual available time per day (minutes)")
        setup_anchors = gr.Textbox(value=_setup_anchors, lines=3, label="Regular anchors (optional, one per line)", placeholder="e.g. Work 10:00–20:00\nMedication / meals\nClass on Tuesdays")
        setup_tasks = gr.Textbox(value=_setup_tasks, lines=4, label="Existing tasks, appointments and deadlines (optional, one per line)", placeholder="e.g. Submit hackathon — deadline 2026-10-29\nDinner with friends — 2026-11-02\nMeal prep — Sunday")
        gr.Markdown("#### Optional cycle heads-up")
        gr.Markdown("CycleOS can flag dated plans that may overlap an estimated period window, so you can choose whether to prepare. It will never assume how you will feel or automatically cancel/reschedule plans. This is a manual estimate, not medical advice or a tracker connection.")
        setup_cycle_mode = gr.Dropdown([("Off — don't use cycle information", "off"), ("Manual estimates — give me date heads-ups", "manual_estimates")], value=_prefs.get("cycle_tracking_mode", "off"), label="Cycle context (manual or imported dates)")
        with gr.Row():
            setup_last_period = gr.Textbox(value=str(_prefs.get("last_period_start", "")), label="Last period start (YYYY-MM-DD; optional)", placeholder="2026-10-12")
            setup_cycle_length = gr.Number(value=int(_prefs.get("average_cycle_length", 28)), minimum=20, maximum=45, step=1, label="Usual cycle length (days)")
            setup_period_length = gr.Number(value=int(_prefs.get("average_period_length", 5)), minimum=1, maximum=10, step=1, label="Usual period length (days)")
        setup_cycle_note = gr.Textbox(value=str(_prefs.get("cycle_context_note", "")), lines=2, label="Anything you want CycleOS to know (optional)", placeholder="e.g. Give me a heads-up about social plans; don't assume I need to cancel them.")
        gr.Markdown("**Connect a cycle tracker — current prototype options**\n\nAutomatic direct syncing from apps such as Clue or Flo is not available inside this Colab prototype. If your tracker lets you export a CSV/JSON file containing period-start dates, you can import it here once. CycleOS stores only the dates it needs for heads-ups, not the raw export. A future mobile version could request permission to read supported health-platform data (for example Apple Health or Android Health Connect), rather than asking you to re-enter dates.")
        cycle_export_file = gr.File(label="Optional: import cycle tracker export (.csv or .json)", file_types=[".csv", ".json"], type="filepath")
        cycle_import_button = gr.Button("Import cycle dates", variant="secondary")
        cycle_import_status = gr.Markdown()
        cycle_import_button.click(import_cycle_data_file, inputs=[cycle_export_file], outputs=[cycle_import_status], queue=False)
        setup_button = gr.Button("Save my setup", variant="secondary")
        setup_status = gr.Markdown()
        setup_button.click(save_cycleos_setup, inputs=[setup_capacity, setup_block, setup_break, setup_available, setup_anchors, setup_tasks, setup_cycle_mode, setup_last_period, setup_cycle_length, setup_period_length, setup_cycle_note], outputs=[setup_status], queue=False, show_progress="minimal")

    session_state = gr.State({})
    gr.Markdown("### Today's plan")
    with gr.Row():
        energy = gr.Slider(1, 5, value=3, step=1, label="Energy", info="1 = very low · 5 = high")
        focus = gr.Slider(1, 5, value=3, step=1, label="Focus", info="1 = very low · 5 = high")
        mood = gr.Dropdown(["Okay", "Overwhelmed", "Anxious", "Low", "Restless", "Positive"], value="Okay", label="How are you feeling?")
    with gr.Row():
        available_hours = gr.Number(value=_default_hours, minimum=0, maximum=24, step=1, label="Time available (hours)")
        available_extra_minutes = gr.Dropdown([0, 15, 30, 45], value=next((x for x in [0, 15, 30, 45] if x == _default_extra), 0), label="Extra minutes")
        day_mode = gr.Dropdown(["auto", "bare_minimum", "low_capacity", "steady", "high_capacity"], value="auto", label="Day mode", info="CycleOS uses your saved definition of capacity.")
    brain_dump_input = gr.Textbox(lines=5, label="What needs doing or has changed?", placeholder="Brain-dump tasks, deadlines, interruptions, worries, or just say what happened. Saved tasks are already included.")
    with gr.Accordion("Add or edit tasks directly (optional)", open=False):
        tasks_input = gr.Textbox(lines=3, label="New tasks (one per line)", placeholder="e.g. Polish README — due 29 October")
    start_button = gr.Button("Build my plan", variant="primary")
    plan_output = gr.Markdown("Your plan will appear here.")
    save_status = gr.Markdown()
    create_event = start_button.click(create_plan, inputs=[energy, focus, mood, available_hours, available_extra_minutes, day_mode, brain_dump_input, tasks_input, session_state], outputs=[plan_output, session_state, save_status])

    gr.Markdown("---\n### If the day changes\nTell CycleOS what changed. It will update this plan rather than starting over.")
    quick_update = gr.Textbox(lines=2, label="What changed?", placeholder="e.g. I lost 30 minutes, finished task 1, or only have 45 minutes left.")
    remaining_time = gr.Number(value=None, minimum=0, maximum=1440, step=1, label="Time left now (optional, minutes)")
    update_button = gr.Button("Adapt my plan")
    update_output = gr.Markdown()
    update_button.click(update_plan, inputs=[quick_update, remaining_time, session_state], outputs=[update_output, session_state])

    with gr.Accordion("Record progress (optional; helps CycleOS learn your estimates)", open=False):
        outcome_task = gr.Dropdown(choices=[], label="Task", interactive=True)
        outcome_status = gr.Dropdown(["completed", "partial", "interrupted", "deferred"], value="completed", label="Outcome")
        actual_minutes = gr.Number(value=None, minimum=0, step=1, label="Actual minutes (optional)")
        outcome_note = gr.Textbox(lines=1, label="Optional note", placeholder="e.g. Interrupted by a call")
        outcome_button = gr.Button("Save progress")
        outcome_output = gr.Markdown()
        task_id_list = gr.Textbox(label="Current task list", interactive=False)
        def show_task_ids(session):
            tasks = (session or {}).get("tasks", [])
            choices = [(f"{t['title']} · {t['id']}", t['id']) for t in tasks]
            labels = [label for label, _value in choices]
            return gr.update(choices=choices, value=choices[0][1] if choices else None), "\n".join(labels) if labels else "Create a plan to populate the task list."
        create_event.then(show_task_ids, inputs=[session_state], outputs=[outcome_task, task_id_list])
        outcome_button.click(record_outcome, inputs=[outcome_task, outcome_status, actual_minutes, outcome_note, session_state], outputs=[outcome_output, session_state], queue=False)

    gr.Markdown("---\n**Privacy note:** this notebook writes to one local history file. Keep the demo private; do not enable a public Gradio share link for real personal data without per-user storage and appropriate privacy controls. Calendar/email auto-import is not connected in this version; CycleOS uses saved tasks, your one-time setup, and whatever you tell it here. Cycle heads-ups use optional manual estimates or imported period-start dates. Direct tracker sync is not connected in this Colab prototype.")

print("CycleOS dashboard defined. Run the final cell to launch the demo. Use synthetic data only on the public share link.")


cycleos_app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), share=False, show_error=True, theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="stone"), css=CYCLEOS_CSS)