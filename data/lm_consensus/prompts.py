"""Prompts and tree shape for the language-model consensus trees (paper Section 6.3, Appendix F).

The reference (base) policy is the language model prompted with the issue and every
participant's opinion (paper Figure 6); participant i's policy is the same model prompted with
the issue and only participant i's opinion (paper Figure 7). The wording follows the authors'
released code (system prompt + user prompt); the statement is the assistant's reply under the
model's chat template.
"""

from __future__ import annotations

BRANCHING = 2  # candidate tokens per prefix: the base policy's 2 most likely tokens
DEPTH = 16  # statement length in tokens
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

REFERENCE_SYSTEM = (
    "You are generating a consensus statement that represents the views of multiple participants.\n"
    "Your task is to continue the statement in a way that addresses the issue and considers all "
    "participants' opinions. Be concise and keep the statement short (less than 50 tokens) and "
    "focused. ONLY WRITE THE STATEMENT AND NOTHING ELSE."
)
REFERENCE_USER = (
    "Issue:\n{issue}\n\nParticipants' opinions:\n{opinions_text}\n\n"
    "Consensus statement (less than 50 tokens):\n"
)
AGENT_SYSTEM = (
    "You are generating a statement that represents the views of a single participant.\n"
    "Your task is to continue the statement in a way that addresses the issue and considers ONLY "
    "this participant's opinion. Be concise and keep the statement short (less than 50 tokens) and "
    "focused. ONLY WRITE THE STATEMENT AND NOTHING ELSE."
)
AGENT_USER = (
    "Issue:\n{issue}\n\nParticipant's opinion:\n{opinion}\n\n"
    "Statement reflecting ONLY this participant's opinion (less than 50 tokens):\n"
)

# Candidate tokens are never special tokens and never contain these strings (the authors bias
# generation against the same strings), so every statement in the tree is a single line of text.
BANNED_SUBSTRINGS = ("\n", "\r", "\t", "...", '"', "**", ":", "•", "�")


def reference_messages(issue: str, opinions: list[str]) -> list[dict]:
    opinions_text = "\n\n".join(f"Participant {i + 1}: {op}" for i, op in enumerate(opinions))
    return [
        {"role": "system", "content": REFERENCE_SYSTEM},
        {"role": "user", "content": REFERENCE_USER.format(issue=issue, opinions_text=opinions_text)},
    ]


def agent_messages(issue: str, opinion: str) -> list[dict]:
    return [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": AGENT_USER.format(issue=issue, opinion=opinion)},
    ]
