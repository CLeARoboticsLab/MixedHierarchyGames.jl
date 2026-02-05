# Infrastructure Verification

Automated checks to enforce via git hooks or CI. These are binary pass/fail rules that don't require judgment.

## Pre-Commit Hooks

### 1. Test File Before Implementation

If a new `src/*.jl` file is added, reject unless a corresponding `test/test_*.jl` file exists in the same commit.

```bash
# Check: new src files must have corresponding test files
for f in $(git diff --cached --name-only --diff-filter=A -- 'src/*.jl'); do
    base=$(basename "$f" .jl)
    if ! git diff --cached --name-only | grep -q "test/test_${base}.jl"; then
        echo "ERROR: New src file $f has no corresponding test/test_${base}.jl"
        exit 1
    fi
done
```

### 2. Function Length Limit

No function body should exceed 50 lines. Warn on commit.

```bash
# Check: Julia functions over 50 lines
julia -e '
for f in ARGS
    lines = readlines(f)
    func_start = 0
    for (i, line) in enumerate(lines)
        if startswith(strip(line), "function ")
            func_start = i
        elseif strip(line) == "end" && func_start > 0
            len = i - func_start
            if len > 50
                println("WARNING: $(f):$(func_start) - function is $len lines (limit: 50)")
            end
            func_start = 0
        end
    end
end
' $(git diff --cached --name-only -- 'src/*.jl')
```

### 3. Commit Size Guard

Warn if a commit touches more than 100 lines or more than 3 files.

```bash
lines=$(git diff --cached --stat | tail -1 | grep -oE '[0-9]+ insertion' | grep -oE '[0-9]+')
files=$(git diff --cached --name-only | wc -l)

if [ "${lines:-0}" -gt 100 ]; then
    echo "WARNING: Commit has $lines insertions (guideline: <100). Consider splitting."
fi
if [ "$files" -gt 3 ]; then
    echo "WARNING: Commit touches $files files (guideline: ≤3). Consider splitting."
fi
```

## CI Checks

### 4. No @test_broken on Main

`@test_broken` must never exist on main. PRs must not introduce new `@test_broken` unless explicitly approved by the user. Warn if any pre-existing `@test_broken` are found.

```bash
# Error: new @test_broken added by this PR
new_broken=$(git diff origin/main...HEAD -- 'test/*.jl' | grep '^+' | grep -v '^+++' | grep '@test_broken')
if [ -n "$new_broken" ]; then
    echo "ERROR: PR introduces new @test_broken:"
    echo "$new_broken"
    echo "This requires explicit user approval."
    exit 1
fi

# Warning: pre-existing @test_broken in codebase
existing=$(grep -r "@test_broken" test/ --include="*.jl" -l 2>/dev/null)
if [ -n "$existing" ]; then
    count=$(grep -r "@test_broken" test/ --include="*.jl" | wc -l)
    echo "WARNING: $count pre-existing @test_broken found in: $existing"
    echo "Consider resolving these in a future PR."
fi
```

### 5. No Magic Numbers in New Code

Flag bare numeric literals (excluding 0, 1, 2, common indices) in changed lines.

```bash
# Heuristic: flag floats like 0.5, 1e-6, 100000 in new code
git diff --cached -U0 -- 'src/*.jl' | grep '^+' | grep -v '^+++' | \
    grep -E '[0-9]+\.[0-9]+|[0-9]+e[+-]?[0-9]+' | \
    grep -v 'const \|# ' && \
    echo "WARNING: Possible magic numbers in new code. Consider named constants."
```

## Status

- [ ] Hook 1: Test file before implementation
- [ ] Hook 2: Function length limit
- [ ] Hook 3: Commit size guard
- [ ] CI 4: No @test_broken at merge
- [ ] CI 5: No magic numbers

None of these are currently wired up. Implement as git pre-commit hooks or GitHub Actions when ready.
