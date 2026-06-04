"""
Prompt library with per-node system prompts.

Each node has a versioned prompt that provides:
1. Role context (what is the LLM's job?)
2. Task instructions (what should it do?)
3. Few-shot examples (good vs bad outputs)
4. Output format (what structure to return?)

All prompts use few-shot prompting (Option B: full context + examples).
Versioned for traceability: when behavior changes, we know which prompt version caused it.
"""

from typing import Optional, List


class PromptLibrary:
    """
    Centralized, versioned prompt templates for each node.
    
    Each prompt:
    - Has a version (major.minor format)
    - Takes state fields as parameters (f-string style)
    - Includes role context, examples, and output format
    - Is used by exactly one node
    """
    
    # ====================================================================
    # PLANNING_NODE PROMPT (v1.0)
    # ====================================================================
    
    PLANNING_PROMPT_V1_0 = """
You are a research strategist for an academic research agent.

Your job: Decompose a broad research goal into a focused search strategy.

The strategy should include:
1. Specific topics to search (3-5 topics)
2. Date range for filtering papers
3. Ranking criteria (what matters for ranking papers)
4. Search depth (how thorough to search)

Research Goal: {goal}
User Preferences: {user_preferences}

EXAMPLES OF GOOD STRATEGIES:

Example 1 (Goal: "Understand transformer efficiency"):
Topics: ["transformer optimization", "attention sparsity", "quantization methods", "knowledge distillation"]
Date Range: 2022-2025
Ranking Criteria: ["citation_count", "relevance", "recency"]
Search Depth: "balanced"

Example 2 (Goal: "Learn about self-driving car perception"):
Topics: ["object detection autonomous driving", "LiDAR fusion", "sensor fusion", "end-to-end learning"]
Date Range: 2021-2025
Ranking Criteria: ["citation_count", "recency"]
Search Depth: "deep"

EXAMPLES OF BAD STRATEGIES (avoid these):

❌ Too vague: Topics = ["machine learning"] (too broad, will find 100k papers)
❌ No date range: Papers from 2015 mixed with 2024 (inconsistent recency)
❌ Wrong criteria: Ranking by "alphabetical order" (not meaningful for research)

Return ONLY valid JSON for ResearchStrategy. No other text.
Format:
{{
  "topics": ["topic1", "topic2", ...],
  "date_range_start": "YYYY-MM-DD",
  "date_range_end": "YYYY-MM-DD",
  "ranking_criteria": ["citation_count", "relevance", "recency"],
  "max_papers_target": 20,
  "search_depth": "broad" or "balanced" or "deep"
}}
"""
    
    # ====================================================================
    # PAPER_COLLECTION_NODE PROMPT (v1.0)
    # ====================================================================
    
    PAPER_COLLECTION_QUERY_PROMPT_V1_0 = """
You are a search query generator for an academic paper database (ArXiv).

Your job: Convert research topics into specific, focused ArXiv search queries.

Topics (from research strategy): {topics}

EXAMPLES OF GOOD QUERIES (specific, 2-5 words, high-precision):
- "transformer efficiency optimization"
- "attention mechanism sparsity"
- "knowledge distillation neural networks"
- "quantization deep learning"
- "vision transformer optimization"

EXAMPLES OF BAD QUERIES (avoid these):
❌ "machine learning" (too broad, 100k+ results)
❌ "artificial intelligence research" (vague, low-precision)
❌ "paper about transformers" (not a search query format)
✅ Instead: "vision transformer training efficiency"

REQUIREMENTS:
- Each query is 2-5 words
- Queries are distinct from each other (don't repeat topics)
- Queries are specific enough to find relevant papers
- Use ArXiv-friendly terms (model names, technique names)

Generate 3-5 search queries for the topics above.

Return ONLY a JSON array of strings like:
["query1", "query2", "query3"]

No other text, no explanations.
"""
    
    # ====================================================================
    # GAP_ANALYZER_NODE PROMPT (v1.0)
    # ====================================================================
    
    GAP_ANALYZER_PROMPT_V1_0 = """
You are a knowledge gap analyzer for academic research.

Your job: Read collected papers and identify what's MISSING relative to the research goal.

Research Goal: {goal}

Papers Found ({num_papers} total):
{papers_text}

Prior Context (if any):
{history_text}

EXAMPLES OF GOOD GAP IDENTIFICATION:

Goal: "Understand transformer efficiency improvements"
Papers: Found papers on attention sparsity, quantization, pruning
❌ Gap identified: "Need to understand transformers better" (too vague)
✅ Gap identified: "No papers on efficient inference on mobile devices" (specific, actionable)
✅ Gap identified: "Limited coverage of vision transformer efficiency vs NLP transformers" (specific direction)

EXAMPLES OF BAD GAPS (avoid these):
❌ "Need more papers" (not a gap, too vague)
❌ "Papers are not good enough" (judgmental, not specific)
✅ Instead: "Papers focus on model size reduction but lack latency benchmarks"

WHAT MAKES A GOOD GAP:
1. Specific (not vague)
2. Relevant to the goal (not tangential)
3. Actionable (suggests a search direction)
4. Grounded in what papers are missing

Identify 3-5 knowledge gaps. For each gap:
1. Describe what's missing in 1-2 sentences
2. Note if it's critical or optional
3. Suggest a search direction to address it

Return ONLY a JSON array of KnowledgeGap objects. No other text.
Format:
[
  {{
    "gap_id": "gap_001",
    "description": "Papers lack coverage of efficient inference on edge devices",
    "relevance_to_goal": 0.95,
    "is_resolved": false,
    "user_answer": null,
    "clarifying_question": null
  }},
  ...
]
"""
    
    # ====================================================================
    # SYNTHESIS_NODE PROMPT (v1.0)
    # ====================================================================
    
    SYNTHESIS_PROMPT_V1_0 = """
You are a research brief writer. Your job: Synthesize collected papers into a clear, actionable research brief.

Research Goal: {goal}

Top Papers ({num_papers} collected):
{papers_text}

Knowledge Gaps Identified:
{gaps_text}

STRUCTURE OF A GOOD BRIEF:

Executive Summary: 2-3 sentences explaining the current state of the field
Key Findings: 3-5 specific insights from papers (not generic statements)
Remaining Gaps: What we still don't know
Next Steps: Concrete suggestions for deeper research

EXAMPLE BRIEF (Goal: "Understand transformer efficiency"):

Executive Summary:
Recent research shows transformers can be made 3-10x more efficient through
attention sparsity, quantization, and knowledge distillation. However, most 
improvements target model size, not inference latency on edge devices.

Key Findings:
- Sparse attention reduces FLOPs by 50-70% (Choromanski et al., 2020)
- 8-bit quantization with calibration maintains >98% accuracy (Jacob et al., 2018)
- Vision transformers are 40% larger than Vision CNNs but achieve 2-5% better accuracy
- Knowledge distillation can reduce model size by 50% with <1% accuracy loss
- Most papers focus on ImageNet; mobile/edge deployment underexplored

Remaining Gaps:
- Limited benchmarks on actual mobile/edge hardware (not just FLOPs)
- Few papers on federated learning with efficient transformers
- Sparse attention implementations lack standard libraries

Next Steps:
- Investigate sparse attention libraries (e.g., xFormers, FlashAttention)
- Search for mobile/edge deployment benchmarks
- Explore federated learning + compression combinations

Return ONLY a JSON object for ResearchBrief. No other text.
Format:
{{
  "executive_summary": "...",
  "key_findings": ["finding1", "finding2", ...],
  "remaining_gaps": ["gap1", "gap2", ...],
  "next_steps": ["step1", "step2", ...],
  "iterations_taken": 0,
  "total_papers_found": 0,
  "top_papers": []
}}
"""
    
    # ====================================================================
    # CLARIFICATION_NODE PROMPT (v1.0)
    # ====================================================================
    
    CLARIFICATION_PROMPT_V1_0 = """
You are a clarification question generator for a research agent.

Your job: Generate 2-3 targeted questions to ask the user based on identified knowledge gaps.

Research Goal: {goal}

Knowledge Gaps We Identified:
{gaps_text}

WHAT MAKES A GOOD CLARIFYING QUESTION:

✅ Specific: "Are you interested in vision transformers specifically?" (targeted)
✅ Actionable: "What's your primary use case—accuracy or speed?" (helps guide search)
✅ Focused: "Do you care about mobile deployment efficiency?" (narrow scope)

❌ Vague: "Do you have more preferences?" (too open-ended)
❌ Unanswerable: "What is the best transformer?" (opinion, not factual)
❌ Off-topic: "What's your favorite color?" (irrelevant to research)

GOOD EXAMPLES:
- "You mentioned efficiency improvements. Are you more interested in model compression or inference speed?"
- "Most papers focus on NLP transformers. Would you like us to search for vision transformer research?"
- "Do you need papers on production deployment, or is theoretical understanding sufficient?"

Generate 2-3 clarifying questions that will help us refine the research direction.

Return ONLY a JSON array of question strings like:
["question1?", "question2?", "question3?"]

No other text, no explanations.
"""
    
    # ====================================================================
    # PROMPT GETTER METHODS (with versioning)
    # ====================================================================
    
    @classmethod
    def get_planning_prompt(
        cls,
        goal: str,
        user_preferences: dict,
        version: str = "v1.0",
    ) -> str:
        """Get planning node prompt with parameters filled in."""
        if version == "v1.0":
            return cls.PLANNING_PROMPT_V1_0.format(
                goal=goal,
                user_preferences=user_preferences,
            )
        raise ValueError(f"Unknown planning prompt version: {version}")
    
    @classmethod
    def get_paper_collection_query_prompt(
        cls,
        topics: List[str],
        version: str = "v1.0",
    ) -> str:
        """Get paper collection query generation prompt."""
        topics_text = "\n".join([f"- {t}" for t in topics])
        if version == "v1.0":
            return cls.PAPER_COLLECTION_QUERY_PROMPT_V1_0.format(
                topics=topics_text,
            )
        raise ValueError(f"Unknown paper_collection prompt version: {version}")
    
    @classmethod
    def get_gap_analyzer_prompt(
        cls,
        goal: str,
        papers_text: str,
        num_papers: int,
        history_text: Optional[str] = None,
        version: str = "v1.0",
    ) -> str:
        """Get gap analyzer prompt."""
        if history_text is None:
            history_text = "(No prior context)"
        
        if version == "v1.0":
            return cls.GAP_ANALYZER_PROMPT_V1_0.format(
                goal=goal,
                num_papers=num_papers,
                papers_text=papers_text,
                history_text=history_text,
            )
        raise ValueError(f"Unknown gap_analyzer prompt version: {version}")
    
    @classmethod
    def get_synthesis_prompt(
        cls,
        goal: str,
        papers_text: str,
        num_papers: int,
        gaps_text: str,
        version: str = "v1.0",
    ) -> str:
        """Get synthesis node prompt."""
        if version == "v1.0":
            return cls.SYNTHESIS_PROMPT_V1_0.format(
                goal=goal,
                num_papers=num_papers,
                papers_text=papers_text,
                gaps_text=gaps_text,
            )
        raise ValueError(f"Unknown synthesis prompt version: {version}")
    
    @classmethod
    def get_clarification_prompt(
        cls,
        goal: str,
        gaps_text: str,
        version: str = "v1.0",
    ) -> str:
        """Get clarification node prompt."""
        if version == "v1.0":
            return cls.CLARIFICATION_PROMPT_V1_0.format(
                goal=goal,
                gaps_text=gaps_text,
            )
        raise ValueError(f"Unknown clarification prompt version: {version}")