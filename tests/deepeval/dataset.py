"""
Static evaluation dataset for cognee-layer DeepEval tests.

Each entry is keyed by test area and represents a ground-truth case used across
document_classifier, query_analyzer, and RAG recall evaluations.
"""

# ── Document classifier ground-truth cases ────────────────────────────────────

CLASSIFIER_CASES = [
    {
        "id": "clf-001",
        "text": (
            "This Employee Code of Conduct Policy sets out the standards of behaviour "
            "expected from all staff. Violations of these guidelines may result in "
            "disciplinary action up to and including termination."
        ),
        "expected_label": "policy",
        "expected_dataset": "ds_policy",
    },
    {
        "id": "clf-002",
        "text": (
            "General Data Protection Regulation (GDPR) Article 17: Right to erasure. "
            "The data subject shall have the right to obtain from the controller the "
            "erasure of personal data without undue delay."
        ),
        "expected_label": "regulation",
        "expected_dataset": "ds_regulation",
    },
    {
        "id": "clf-003",
        "text": (
            "Q3 2023 Sales Forecast: Total pipeline $4.2M, expected close rate 38%. "
            "Top competitors: Salesforce, HubSpot. Revenue target for this quarter: $1.6M."
        ),
        "expected_label": "market",
        "expected_dataset": "ds_market",
    },
    {
        "id": "clf-004",
        "text": (
            "New Employee Onboarding Checklist. HR must complete payroll setup within "
            "3 days of start date. Employee headcount as of January 2024: 142 FTEs."
        ),
        "expected_label": "employees",
        "expected_dataset": "ds_employees",
    },
    {
        "id": "clf-005",
        "text": (
            "Supplier Agreement — SLA Terms. Vendor must maintain 99.9% uptime. "
            "Procurement clause 4.2: payment terms net-30. This contract governs "
            "all services provided under master services agreement MSA-2024-001."
        ),
        "expected_label": "contracts",
        "expected_dataset": "ds_contracts",
    },
    {
        "id": "clf-006",
        "text": (
            "Annual Audit Report 2023. Key KPI findings: customer satisfaction score 4.1/5, "
            "on-time delivery 94%. Executive summary: operational metrics exceeded targets "
            "in 3 of 5 business units."
        ),
        "expected_label": "reports",
        "expected_dataset": "ds_reports",
    },
    {
        "id": "clf-007",
        "text": (
            "Purchase-to-Pay BPMN Workflow. Step 1: procurement receives invoice. "
            "Step 2: automated 3-way match task. Step 3: approval activity. "
            "The pipeline triggers payment automation on successful match."
        ),
        "expected_label": "workflow",
        "expected_dataset": "ds_workflow",
    },
]


# ── Query analyzer ground-truth cases ─────────────────────────────────────────

QUERY_ANALYZER_CASES = [
    {
        "id": "qa-001",
        "text": "What happened in Q2 2023 with our main supplier?",
        "expected_intent": "search",
        "expected_temporal": True,
        "expected_search_type": "TEMPORAL",
        "description": "Explicit quarter reference → temporal search",
    },
    {
        "id": "qa-002",
        "text": "Remember that the project deadline is next Monday.",
        "expected_intent": "save",
        "expected_temporal": True,
        "expected_search_type": None,  # intent is save, search_type irrelevant
        "description": "Save intent with temporal marker",
    },
    {
        "id": "qa-003",
        "text": "How does our investment policy relate to GDPR compliance?",
        "expected_intent": "search",
        "expected_temporal": False,
        "expected_search_type": "GRAPH_COMPLETION",
        "description": "Relationship query → GRAPH_COMPLETION",
    },
    {
        "id": "qa-004",
        "text": "Please remove the BPI supplier entry from our knowledge base.",
        "expected_intent": "delete",
        "expected_temporal": False,
        "expected_search_type": None,
        "description": "Explicit delete/remove intent",
    },
    {
        "id": "qa-005",
        "text": "Give me a high-level summary of our procurement policies.",
        "expected_intent": "search",
        "expected_temporal": False,
        "expected_search_type": "GRAPH_SUMMARY_COMPLETION",
        "description": "Summary/overview query → GRAPH_SUMMARY_COMPLETION",
    },
    {
        "id": "qa-006",
        "text": "Update the employee onboarding checklist to include the new IT setup step.",
        "expected_intent": "improve",
        "expected_temporal": False,
        "expected_search_type": None,
        "description": "Update/improve intent",
    },
    {
        "id": "qa-007",
        "text": "What was the revenue last year in the EMEA region?",
        "expected_intent": "search",
        "expected_temporal": True,
        "expected_search_type": "TEMPORAL",
        "description": "Historical revenue query → temporal",
    },
]


# ── RAG recall Q&A pairs ───────────────────────────────────────────────────────
# Used for live API evaluation — facts are seeded before the test run.

RAG_SEED_FACTS = [
    {
        "user_id": "eval_user_rag",
        "fact": (
            "Bpifrance investment policy mandates that at least 60% of portfolio "
            "companies must be French SMEs with fewer than 250 employees."
        ),
    },
    {
        "user_id": "eval_user_rag",
        "fact": (
            "The supplier diversity policy requires at least 20% of procurement "
            "budget to be allocated to certified minority-owned businesses."
        ),
    },
    {
        "user_id": "eval_user_rag",
        "fact": (
            "EPIC annual report 2022 recorded total assets of €48.2 billion "
            "and a net profit of €1.1 billion for the fiscal year."
        ),
    },
]

RAG_EVAL_CASES = [
    {
        "id": "rag-001",
        "user_id": "eval_user_rag",
        "query": "What percentage of Bpifrance portfolio must be French SMEs?",
        "expected_answer": "At least 60% of portfolio companies must be French SMEs with fewer than 250 employees.",
        "context_hint": "investment policy mandate 60% French SMEs",
    },
    {
        "id": "rag-002",
        "user_id": "eval_user_rag",
        "query": "What is the supplier diversity procurement target?",
        "expected_answer": "20% of the procurement budget must go to certified minority-owned businesses.",
        "context_hint": "supplier diversity 20% minority-owned",
    },
    {
        "id": "rag-003",
        "user_id": "eval_user_rag",
        "query": "What were EPIC's total assets in the 2022 annual report?",
        "expected_answer": "Total assets were €48.2 billion with a net profit of €1.1 billion.",
        "context_hint": "EPIC 2022 total assets €48.2 billion",
    },
]
