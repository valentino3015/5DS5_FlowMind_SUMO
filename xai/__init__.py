"""XAI layer: explains the orchestrator's decisions after a run (why, why not, what if).

Owner: Adem.

It only READS what a run wrote (decisions.csv, optionally states.jsonl) and never
controls a traffic light. That is why it lives outside flowmind/: the control loop
stays free of any LLM. The optional local LLM (xai/llm.py) only rephrases facts.
"""
