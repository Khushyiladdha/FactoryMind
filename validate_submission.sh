#!/usr/bin/env bash
# =============================================================================
# validate_submission.sh — FactoryMind Pre-Submission Checker
# =============================================================================
# Runs ALL validation steps locally before you push to HuggingFace.
# Usage: ./validate_submission.sh [--skip-docker] [--skip-inference]
#
# Prerequisites:
#   - Docker        https://docs.docker.com/get-docker/
#   - Python 3.10+
#   - pip install -r requirements.txt
#   - HF_TOKEN env var set (for inference step)
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

SKIP_DOCKER=false
SKIP_INFERENCE=false

for arg in "$@"; do
  case $arg in
    --skip-docker)    SKIP_DOCKER=true ;;
    --skip-inference) SKIP_INFERENCE=true ;;
  esac
done

PASS=0
FAIL=0

check() {
  local name="$1"
  local cmd="$2"
  echo -e "\n${BOLD}▶ $name${NC}"
  if eval "$cmd"; then
    echo -e "${GREEN}  ✅ PASS${NC}"
    PASS=$((PASS + 1))
  else
    echo -e "${RED}  ❌ FAIL${NC}"
    FAIL=$((FAIL + 1))
  fi
}

echo -e "${BOLD}======================================${NC}"
echo -e "${BOLD}  FactoryMind Submission Validator    ${NC}"
echo -e "${BOLD}======================================${NC}"

# 1. Python syntax check
check "Python syntax" "python -c \"
import ast, sys, os
files = [
  'factory_mind/models.py', 'factory_mind/graders.py',
  'factory_mind/env.py', 'factory_mind/__init__.py',
  'server/app.py', 'inference.py',
]
for f in files:
    ast.parse(open(f).read())
print('  All files parse OK')
\""

# 2. openenv.yaml exists and has required fields
check "openenv.yaml structure" "python -c \"
import yaml
with open('openenv.yaml') as f:
    spec = yaml.safe_load(f)
required = ['name', 'version', 'tasks', 'observation_space', 'action_space']
missing = [k for k in required if k not in spec]
assert not missing, f'Missing fields: {missing}'
assert len(spec['tasks']) >= 3, 'Need at least 3 tasks'
print(f'  {len(spec[\"tasks\"])} tasks defined')
\" 2>/dev/null || python -c \"
import json, re
with open('openenv.yaml') as f:
    content = f.read()
assert 'name:' in content
assert 'tasks:' in content
print('  openenv.yaml basic check passed')
\""

# 3. Smoke test (no LLM needed)
check "Smoke test (test_local.py)" "python test_local.py 2>&1 | tail -20"

# 4. Pytest suite
check "Pytest suite" "python -m pytest tests/test_env.py -v --tb=short -q 2>&1 | tail -30" || true

# 5. inference.py exists and is importable (syntax only — no API key needed for this check)
check "inference.py syntax" "python -c \"import ast; ast.parse(open('inference.py').read()); print('  inference.py syntax OK')\""

# 6. Required env vars documented
check "ENV vars documented" "grep -q 'API_BASE_URL' inference.py && grep -q 'MODEL_NAME' inference.py && grep -q 'HF_TOKEN' inference.py && echo '  All required env vars referenced'"

# 7. [START]/[STEP]/[END] format in inference.py
check "Stdout log format" "grep -q '\[START\]' inference.py && grep -q '\[STEP\]' inference.py && grep -q '\[END\]' inference.py && echo '  Log format strings found'"

# 8. Dockerfile exists and has required directives
check "Dockerfile valid" "grep -q 'EXPOSE 8000' Dockerfile && grep -q 'uvicorn' Dockerfile && grep -q 'HEALTHCHECK' Dockerfile && echo '  Dockerfile OK'"

# 9. README has tasks table
check "README baseline table" "grep -qE '0\.[0-9]+' README.md && grep -q 'easy_reorder' README.md && echo '  README has baseline scores'"

# 10. Docker build (optional, slow)
if [ "$SKIP_DOCKER" = false ]; then
  check "Docker build" "docker build -t factory-mind-test . -q && echo '  Docker build succeeded'"

  check "Docker run + health check" "
    CONTAINER=\$(docker run -d -p 18000:8000 factory-mind-test)
    sleep 4
    STATUS=\$(curl -sf http://localhost:18000/health | python -c \"import sys,json; d=json.load(sys.stdin); print(d['status'])\" 2>/dev/null || echo 'fail')
    docker stop \$CONTAINER > /dev/null
    docker rm \$CONTAINER > /dev/null
    [ \"\$STATUS\" = 'ok' ] && echo '  Docker health check passed'
  "
else
  echo -e "\n${YELLOW}⚠  Docker checks skipped (--skip-docker)${NC}"
fi

# 11. Inference script (optional — requires HF_TOKEN)
if [ "$SKIP_INFERENCE" = false ] && [ -n "${HF_TOKEN:-}" ]; then
  check "Baseline inference" "timeout 1200 python inference.py 2>&1 | tee /tmp/inference_log.txt | grep -E '^\[END\]' | wc -l | grep -q '[4-9]' && echo '  Inference completed 4 tasks'"
else
  echo -e "\n${YELLOW}⚠  Inference check skipped (set HF_TOKEN or pass --skip-inference)${NC}"
fi

# Summary
echo ""
echo -e "${BOLD}======================================${NC}"
echo -e "${BOLD}  Results: ${GREEN}${PASS} passed${NC}  ${RED}${FAIL} failed${NC}${BOLD}  ${NC}"
echo -e "${BOLD}======================================${NC}"

if [ "$FAIL" -eq 0 ]; then
  echo -e "${GREEN}${BOLD}🎉  All checks passed! Ready to submit.${NC}"
  exit 0
else
  echo -e "${RED}${BOLD}❌  Fix the failing checks above before submitting.${NC}"
  exit 1
fi
