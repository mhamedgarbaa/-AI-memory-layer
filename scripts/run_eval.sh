#!/usr/bin/env bash
# Run the full DeepEval evaluation suite.
#
# Usage:
#   ./scripts/run_eval.sh                   # all tests (requires running stack)
#   ./scripts/run_eval.sh --unit-only       # classifier + query analyzer only (no stack)
#   ./scripts/run_eval.sh --rag-only        # RAG recall tests only
#   SKIP_LIVE=1 ./scripts/run_eval.sh       # skip all live-stack tests
#
# Environment:
#   BASE_URL         — web service (default: http://localhost:8000)
#   ORCHESTRATOR_URL — orchestrator (default: http://localhost:8003)
#   OPENAI_API_KEY   — required for DeepEval LLM judge
#   DEEPEVAL_MODEL   — judge model (default: gpt-4o-mini)
#   COGNIFY_WAIT     — seconds to wait after seeding RAG facts (default: 15)

set -euo pipefail

UNIT_ONLY=false
RAG_ONLY=false

for arg in "$@"; do
  case $arg in
    --unit-only) UNIT_ONLY=true ;;
    --rag-only)  RAG_ONLY=true  ;;
  esac
done

if [ "$UNIT_ONLY" = true ]; then
  echo "==> Running unit evaluations (classifier + query analyzer)"
  pytest tests/deepeval/test_document_classifier.py \
         tests/deepeval/test_query_analyzer.py \
         -v --tb=short
elif [ "$RAG_ONLY" = true ]; then
  echo "==> Running RAG recall evaluations"
  pytest tests/deepeval/test_rag_recall.py -v --tb=short
else
  echo "==> Running full DeepEval evaluation suite"
  pytest tests/deepeval/ -v --tb=short
fi
