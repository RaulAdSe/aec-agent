# 🔀 Git Workflow - AEC Compliance Agent

## 📋 Table of Contents

1. [Overview](#overview)
2. [Branch Strategy](#branch-strategy)
3. [Commit Conventions](#commit-conventions)
4. [Pull Request Process](#pull-request-process)
5. [Code Review Guidelines](#code-review-guidelines)
6. [Release Process](#release-process)
7. [Common Workflows](#common-workflows)
8. [Git Commands Cheatsheet](#git-commands-cheatsheet)

---

## Overview

This project follows a **Feature Branch Workflow** with:
- `main` branch for stable releases
- `develop` branch for integration
- Feature branches for new work

---

## Branch Strategy

### Branch Types

```
main (protected)
  │
  └─── develop (default, protected)
         │
         ├─── feature/pilar-1-extraction
         ├─── feature/pilar-2-calculations
         ├─── feature/pilar-3-rag
         ├─── feature/pilar-4-agent
         ├─── fix/door-detection-bug
         ├─── docs/api-reference
         └─── refactor/geometry-performance
```

### Branch Naming Convention

| Type | Pattern | Example | Purpose |
|------|---------|---------|---------|
| Feature | `feature/<name>` | `feature/pilar-3-rag` | New features |
| Fix | `fix/<name>` | `fix/door-detection-bug` | Bug fixes |
| Docs | `docs/<name>` | `docs/rag-tutorial` | Documentation |
| Refactor | `refactor/<name>` | `refactor/geometry-perf` | Code improvements |
| Test | `test/<name>` | `test/unit-coverage` | Testing |
| Chore | `chore/<name>` | `chore/update-deps` | Maintenance |

### Branch Descriptions

#### `main`
- **Purpose**: Production-ready code
- **Protected**: Yes
- **Direct commits**: ❌ Not allowed
- **Merge from**: `develop` only
- **Tagged**: Yes (v1.0.0, v1.1.0, etc.)

#### `develop`
- **Purpose**: Integration branch
- **Protected**: Yes
- **Default branch**: Yes
- **Direct commits**: ❌ Not allowed
- **Merge from**: Feature branches via PR
- **Merge to**: `main` for releases

#### Feature Branches
- **Purpose**: Develop new features
- **Protected**: No
- **Base**: `develop`
- **Lifetime**: Until feature complete
- **Naming**: `feature/<descriptive-name>`

---

## Commit Conventions

### Commit Message Format

```
<type>: <subject>

<optional body>

<optional footer>
```

### Commit Types

| Type | Description | Example |
|------|-------------|---------|
| `Add` | New feature | `Add: RAG vectorstore manager` |
| `Fix` | Bug fix | `Fix: door width calculation error` |
| `Update` | Update existing feature | `Update: improve room detection` |
| `Refactor` | Code restructuring | `Refactor: simplify geometry module` |
| `Remove` | Remove code/files | `Remove: deprecated function` |
| `Docs` | Documentation only | `Docs: add RAG tutorial notebook` |
| `Test` | Add/update tests | `Test: add unit tests for graph` |
| `Style` | Code style/formatting | `Style: format with black` |
| `Chore` | Maintenance | `Chore: update dependencies` |

### Subject Line Rules

1. **Capitalize** first word
2. **No period** at the end
3. **Imperative mood** (Add, not Added)
4. **50 characters** or less
5. **Descriptive** and clear

### Body Rules (Optional)

- Blank line after subject
- Explain **what** and **why**, not how
- Wrap at 72 characters
- Use bullet points if needed

### Footer (Optional)

- Reference issues: `Closes #123`
- Breaking changes: `BREAKING CHANGE: ...`

### Examples

#### Good Commits

```bash
Add: Pydantic schemas for project data model

- Add ProjectMetadata schema
- Add Room, Door, Wall schemas
- Include field validation and defaults

Closes #15
```

```bash
Fix: door detection in DXF extraction

Door blocks were not being recognized due to case-sensitive
comparison. Now using case-insensitive matching.

Fixes #23
```

```bash
Test: add comprehensive geometry unit tests

- Test area calculation for rectangles and L-shapes
- Test centroid calculation
- Test point containment
- Achieve 95% coverage

Closes #18
```

```bash
Docs: add RAG tutorial notebook

Step-by-step guide for:
- Creating vectorstore
- Querying documents
- Understanding retrieval
```

```bash
Refactor: optimize graph creation performance

- Cache polygon conversions
- Reduce redundant calculations
- 3x speed improvement

Closes #42
```

#### Bad Commits

```bash
# ❌ Too vague
updated files

# ❌ Wrong tense
Added new feature

# ❌ Too long subject
Add comprehensive RAG system with vectorstore, embeddings, and QA chain

# ❌ Not descriptive
fix bug

# ❌ Multiple unrelated changes
Add RAG system, fix door bug, update docs, refactor geometry
```

---

## Pull Request Process

### 1. Create Feature Branch

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/pilar-3-rag
```

### 2. Make Changes

```bash
# Make changes
# ...

# Stage changes
git add src/rag/

# Commit with good message
git commit -m "Add: vectorstore manager with ChromaDB

- Add VectorstoreManager class
- Implement create_from_pdfs() method
- Add load_existing() method
- Include progress tracking"
```

### 3. Push to Remote

```bash
# First push
git push -u origin feature/pilar-3-rag

# Subsequent pushes
git push
```

### 4. Create Pull Request

**On GitHub**:

1. Click "New Pull Request"
2. Base: `develop` ← Compare: `feature/pilar-3-rag`
3. Fill in PR template:

```markdown
## Description
Implements RAG system for querying building code documentation.

## Changes Made
- ✅ Vectorstore manager with ChromaDB
- ✅ Document loading from PDFs
- ✅ QA chain with Gemini LLM
- ✅ Unit tests (85% coverage)
- ✅ Tutorial notebook

## Type of Change
- [x] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation

## Testing
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Manual testing completed
- [x] Tutorial notebook verified

## Checklist
- [x] Code follows style guidelines
- [x] Self-review completed
- [x] Comments added to complex code
- [x] Documentation updated
- [x] Tests added/updated
- [x] All tests passing
- [x] No console errors

## Screenshots/Demo
[If applicable]

## Related Issues
Closes #20
```

### 5. Code Review

Wait for review from team member. Address feedback:

```bash
# Make requested changes
# ...

# Commit fixes
git add .
git commit -m "Fix: address PR review comments

- Rename function for clarity
- Add error handling
- Update docstrings"

# Push changes
git push
```

### 6. Merge Pull Request

Once approved:

1. **Squash and Merge** (preferred)
   - Combines all commits into one
   - Keeps `develop` history clean
   
2. **Merge Commit**
   - Preserves all commits
   - Use for large features

3. **Rebase and Merge**
   - Linear history
   - Use for small features

### 7. Delete Branch

```bash
# On GitHub, delete branch after merge

# Locally
git checkout develop
git pull origin develop
git branch -d feature/pilar-3-rag
```

---

## Code Review Guidelines

### For Authors

**Before Requesting Review**:
- ✅ All tests pass
- ✅ Code is self-reviewed
- ✅ Documentation updated
- ✅ No console errors or warnings
- ✅ Follows style guide

**During Review**:
- Respond to all comments
- Ask for clarification if needed
- Be open to feedback
- Make requested changes promptly

### For Reviewers

**What to Check**:
1. **Functionality**: Does it work as intended?
2. **Tests**: Are there adequate tests?
3. **Code Quality**: Is it readable and maintainable?
4. **Documentation**: Are changes documented?
5. **Style**: Follows project conventions?
6. **Performance**: Any obvious bottlenecks?
7. **Security**: Any security concerns?

**How to Review**:
- Be constructive and specific
- Explain the "why" behind suggestions
- Approve if minor issues (can be fixed later)
- Request changes if major issues
- Use GitHub's review features

**Review Comments Examples**:

```markdown
# ✅ Good
Consider using a try-except here to handle potential KeyError 
when accessing room data. This would make the function more robust.

# ❌ Bad
This is wrong.
```

```markdown
# ✅ Good
Great use of Pydantic validation! One suggestion: we could add 
a custom validator for the boundary field to ensure it's a closed 
polygon (first point == last point).

# ❌ Bad
Add validation.
```

---

## Release Process

### 1. Prepare Release

```bash
# Ensure develop is up to date
git checkout develop
git pull origin develop

# Run all tests
pytest

# Check coverage
pytest --cov=src --cov-report=term

# Build documentation
# Generate CHANGELOG
```

### 2. Create Release Branch

```bash
git checkout -b release/v1.0.0
```

### 3. Update Version

```python
# Update version in relevant files
# - setup.py
# - __init__.py
# - README.md
```

### 4. Commit Release Prep

```bash
git add .
git commit -m "Chore: prepare v1.0.0 release

- Update version numbers
- Update CHANGELOG
- Update README"
```

### 5. Merge to Main

```bash
# Create PR: release/v1.0.0 → main
# After approval, merge
git checkout main
git pull origin main
```

### 6. Tag Release

```bash
git tag -a v1.0.0 -m "Release v1.0.0 - Initial POC

Complete implementation of Agentic AI for AEC compliance.

Features:
- Data extraction from DWG/DXF
- Geometric calculations
- RAG system for building codes
- ReAct agent with 6 tools

See CHANGELOG.md for details."

git push origin v1.0.0
```

### 7. Merge Back to Develop

```bash
git checkout develop
git merge main
git push origin develop
```

### 8. Create GitHub Release

On GitHub:
1. Go to Releases
2. Click "Draft new release"
3. Select tag `v1.0.0`
4. Copy release notes
5. Publish release

---

## Common Workflows

### Starting New Feature

```bash
# Update develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/my-feature

# Make changes
# ...

# Commit
git add .
git commit -m "Add: feature description"

# Push
git push -u origin feature/my-feature

# Create PR on GitHub
```

### Fixing Bug

```bash
# Create fix branch
git checkout develop
git pull origin develop
git checkout -b fix/bug-description

# Fix bug
# ...

# Add tests to prevent regression
# ...

# Commit
git add .
git commit -m "Fix: bug description

Explain what was wrong and how it's fixed"

# Push and create PR
git push -u origin fix/bug-description
```

### Updating Feature Branch

```bash
# Get latest from develop
git checkout develop
git pull origin develop

# Update feature branch
git checkout feature/my-feature
git merge develop

# Or rebase (cleaner history)
git rebase develop

# Resolve conflicts if any
# ...

# Push (force push if rebased)
git push --force-with-lease
```

### Resolving Conflicts

```bash
# When merge conflict occurs
git status  # See conflicted files

# Open files and resolve conflicts
# Look for <<<<<<< HEAD markers

# After resolving
git add <resolved-files>
git commit -m "Merge: resolve conflicts with develop"

# Or if rebasing
git rebase --continue
```

### Squashing Commits

```bash
# Interactive rebase last 3 commits
git rebase -i HEAD~3

# In editor, change 'pick' to 'squash' for commits to combine
# Save and exit

# Edit combined commit message
# Save and exit

# Force push (only if not yet merged)
git push --force-with-lease
```

### Undoing Changes

```bash
# Discard uncommitted changes
git restore <file>
git restore .  # All files

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Revert a commit (creates new commit)
git revert <commit-hash>
```

---

## Git Commands Cheatsheet

### Basic Operations

```bash
# Clone repository
git clone https://github.com/user/repo.git

# Check status
git status

# Stage changes
git add <file>
git add .  # All files

# Commit changes
git commit -m "message"

# Push to remote
git push
git push -u origin <branch>  # First time

# Pull from remote
git pull
git pull origin develop
```

### Branching

```bash
# List branches
git branch
git branch -a  # Include remote

# Create branch
git branch <name>
git checkout -b <name>  # Create and switch

# Switch branch
git checkout <name>
git switch <name>  # Newer command

# Delete branch
git branch -d <name>  # Safe delete
git branch -D <name>  # Force delete

# Rename branch
git branch -m <old> <new>
```

### Viewing History

```bash
# View commit history
git log
git log --oneline  # Compact
git log --graph --oneline --all  # Visual

# View changes
git diff
git diff <branch>
git diff <commit1> <commit2>

# View file history
git log -- <file>
git blame <file>
```

### Stashing

```bash
# Save changes temporarily
git stash
git stash save "description"

# List stashes
git stash list

# Apply stash
git stash apply
git stash pop  # Apply and remove

# Drop stash
git stash drop
```

### Remote Operations

```bash
# List remotes
git remote -v

# Add remote
git remote add origin <url>

# Fetch from remote
git fetch origin

# Pull with rebase
git pull --rebase

# Push force (careful!)
git push --force-with-lease
```

### Tags

```bash
# List tags
git tag

# Create tag
git tag v1.0.0
git tag -a v1.0.0 -m "message"  # Annotated

# Push tags
git push origin v1.0.0
git push --tags  # All tags

# Delete tag
git tag -d v1.0.0  # Local
git push origin :v1.0.0  # Remote
```

---

## GitHub Actions

### Automated Tests

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest --cov=src
```

### Linting

```yaml
# .github/workflows/lint.yml
name: Lint

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Run flake8
      run: flake8 src/ tests/
    - name: Check formatting
      run: black --check src/ tests/
```

---

## Best Practices

### Do's ✅

- **Commit often**: Small, logical commits
- **Write good messages**: Clear and descriptive
- **Pull before push**: Avoid conflicts
- **Review your own code**: Before requesting review
- **Test before committing**: All tests should pass
- **Keep branches short-lived**: Merge frequently
- **Update documentation**: Keep docs in sync with code

### Don'ts ❌

- **Don't commit to main/develop directly**: Always use feature branches
- **Don't commit broken code**: Tests should pass
- **Don't commit secrets**: Use .env for sensitive data
- **Don't force push to shared branches**: Only to your own feature branches
- **Don't commit commented code**: Remove it
- **Don't commit debug statements**: Remove console.log, print()
- **Don't mix unrelated changes**: One feature per branch

---

## Troubleshooting

### "Merge conflict"

```bash
# See conflicted files
git status

# Open files, look for <<<<<<< HEAD

# Resolve conflicts

# Mark as resolved
git add <file>

# Complete merge
git commit
```

### "Detached HEAD state"

```bash
# Create branch from current state
git checkout -b recovery-branch

# Or discard and go back
git checkout develop
```

### "Pushed wrong commit"

```bash
# If not yet merged, revert and force push
git revert <commit-hash>
git push

# Or reset (if sole author)
git reset --hard HEAD~1
git push --force-with-lease
```

### "Lost commits after reset"

```bash
# Find lost commit
git reflog

# Recover
git checkout <commit-hash>
git branch recovery-branch
```

---

## Summary

### Quick Reference

```bash
# Daily workflow
git checkout develop
git pull origin develop
git checkout -b feature/my-feature
# ... make changes ...
git add .
git commit -m "Add: feature description"
git push -u origin feature/my-feature
# Create PR on GitHub

# Update from develop
git checkout develop
git pull origin develop
git checkout feature/my-feature
git merge develop

# After PR merged
git checkout develop
git pull origin develop
git branch -d feature/my-feature
```

### Commit Message Template

```
<type>: <subject>

<body>

<footer>
```

**Types**: Add, Fix, Update, Refactor, Remove, Docs, Test, Style, Chore

---

**Version**: 1.0  
**Last Updated**: October 2025