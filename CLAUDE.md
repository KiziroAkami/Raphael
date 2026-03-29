# Raphael — Personal Working Guidelines

This file defines how I work with Claude on every project. It loads automatically at session start.

---

## Before Any Project

1. Verify machine setup is complete: Homebrew, Node, `gh`, SSH, git identity, `gh auth login`
2. Create project folder + GitHub repo:
   ```bash
   mkdir ~/Projects/your-project && cd ~/Projects/your-project
   git init && gh repo create your-project --private --source=. --remote=origin
   ```
3. Create a project-specific `CLAUDE.md` with stack, conventions, and constraints

---

## Every Session

**State the goal first.** Not "let's work on the project" — be specific:
> "Today I want to implement user login. Backend is stubbed. I have 2 hours."

**Model check:**
- Sonnet is default — stay here for planning, reviews, fixes, single-file work
- Switch to Opus (`/model opus`) only for major multi-file implementation or non-obvious debugging
- Switch back after the heavy coding is done

---

## Starting Something New

```
1. /plan [describe what you want]   ← Always. Even for small things.
2. Read and review the plan
3. Confirm → then implement
```

Never skip the plan step. A 5-minute plan prevents a 2-hour refactor.

---

## During Implementation

- Ask "why" before asking for a fix — understanding comes first
- Review what Claude writes before approving — don't just hit Enter
- If something feels wrong, say so even without knowing why: "this doesn't feel right, explain it"
- Keep an eye on file sizes — flag if something is growing too large
- Use `/tdd` — tests are written before implementation, always

---

## Before Every Commit

```bash
/code-review          # catch quality issues
/security-scan        # if touching auth, user input, or credentials
```

Commit format:
```
feat: what you added
fix: what you fixed
refactor: what you restructured
```

---

## When Stuck

1. State expected vs. actual: "I expected X, I got Y"
2. Ask Claude to explain *why* before asking for a fix
3. Build error → `/build-fix`
4. Stuck 10+ minutes → compact and restate the problem fresh

---

## Weekly Habits

```bash
/instinct-status      # see what patterns Claude has learned from your sessions
/security-scan        # before anything goes near production
```

---

## The Short Version

```
New feature    → /plan first, always
Implementing   → You steer, Claude writes, /tdd enforced
Before commit  → /code-review + /security-scan if needed
Stuck          → Expected vs. actual, "why" before "fix"
```
