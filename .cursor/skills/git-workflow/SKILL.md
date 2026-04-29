---
name: git-workflow
description: >-
  Enforces an opinionated git workflow for a Python data science codebase that
  builds AI agents and agentic systems. Use when the user asks to commit, git,
  push, branch, handle version control, or says "save my work". Applies commit
  timing rules, conventional commit formatting, branch naming, pre-commit
  checks, and strict exclusions for secrets and data artifacts.
---

# Git Workflow For Agentic Python DS Projects

## Purpose

Use this workflow to keep history clean, prevent secret leaks, and maintain a
stable main branch while shipping AI-agent features quickly.

## Working Session Commit Rules

1. Always commit at the end of each working session, even if work is partial.
2. Commit before switching to a different task or component.
3. Keep commits atomic: one logical change per commit.
4. If a commit message needs the word "and", split into separate commits.

## Commit Message Standard (Mandatory)

Format every commit as:

`type(scope): description`

### Allowed types

- `feat`: new agent functionality
- `fix`: bug fixes
- `refactor`: restructuring code without behavior change intent
- `docs`: README, docs, or explanatory comments
- `chore`: config, tooling, dependency, or environment changes

### Scope rules

- Scope must identify the changed component.
- Use concrete scopes such as:
  - `mcp-server`
  - `cleaning-agent`
  - `semantic-layer`
  - `orchestration`
  - `env`
  - `prompts`

### Description rules

- Lowercase only
- Imperative mood
- Under 72 characters
- No trailing period

### Valid examples

- `feat(mcp-server): add query tool for semantic layer lookups`
- `fix(cleaning-agent): handle empty dataframe edge case`
- `chore(env): add api key placeholder to env example`

## Never Commit These

- `.env` files, API keys, credentials, tokens, private keys, or any secret
- large data files and datasets
- `__pycache__/`
- `*.pyc`
- virtual environments (`.venv/`, `venv/`, `env/`)
- `.ipynb_checkpoints/`

If any of these appear in `git status` or staged changes, stop and remove them
from tracking before commit.

## Required Pre-Commit Checks

Run this checklist before every commit:

1. Run the project entry point or main script and confirm it still works.
2. Run `git diff` (and staged diff if needed) to inspect exact changes.
3. Verify no secrets or credentials are present in staged files.
4. Confirm the commit is atomic and scoped to one logical change.

Do not commit if any check fails.

## Branch Strategy

- Use feature branches for new work.
- Branch naming must be `feature/descriptive-name`.
- Keep `main` stable; never commit broken code directly to `main`.
- Delete feature branches after merge.

### Branch examples

- `feature/data-cleaning-agent`
- `feature/slackbot-integration`
- `feature/mcp-query-routing`

## Operational Guidance For Agents

When asked to "save my work", "commit", "push", or handle git:

1. Review changes and group into atomic commit units.
2. Propose split commits if the change mixes unrelated concerns.
3. Enforce conventional commit format exactly.
4. Block commits that include secrets, datasets, cache artifacts, or env files.
5. Prefer safety over speed: keep history reviewable and production-ready.

## Definition Of Done For Git Hygiene

- Commits are atomic and conventionally formatted.
- No secrets or heavy artifacts are committed.
- Feature work is done on `feature/*` branches.
- `main` remains stable and merge-ready.
