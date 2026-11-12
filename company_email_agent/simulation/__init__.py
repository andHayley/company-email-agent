"""Simulated-user evaluation (course-provided). See README.md in this directory.

    persona.py       the persona file format and its loader
    persona_user.py  the simulated user: a UserIO driven by the "user" model
    runner.py        run_persona: one conversation, from first message to score
    checks.py        programmatic checks over a conversation record
    judge.py         the judge prompt and the verdict parser
    evaluate.py      run many personas, print the table, compute points
    record.py        ConversationRecord, the data all of the above share
"""

from company_email_agent.simulation.evaluate import (
    adapter_agent_factory,
    format_table,
    points,
    required_successes,
    run_eval,
)
from company_email_agent.simulation.persona import Persona, load_persona, load_personas, parse_persona
from company_email_agent.simulation.persona_user import PersonaUser
from company_email_agent.simulation.record import ConversationRecord, SessionRecord
from company_email_agent.simulation.runner import run_persona

__all__ = [
    "ConversationRecord",
    "Persona",
    "PersonaUser",
    "SessionRecord",
    "adapter_agent_factory",
    "format_table",
    "load_persona",
    "load_personas",
    "parse_persona",
    "points",
    "required_successes",
    "run_eval",
    "run_persona",
]
