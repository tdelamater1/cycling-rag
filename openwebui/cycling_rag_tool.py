"""
title: Cycling RAG
author: td
description: Query your intervals.icu cycling training data using natural language. Supports period comparisons, session recommendations, and general training questions.
version: 0.1.0
license: MIT
"""

import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        CYCLING_RAG_URL: str = Field(
            default="http://localhost:8000",
            description="Base URL of the Cycling RAG FastAPI service (no trailing slash).",
        )
        TIMEOUT_SECONDS: int = Field(
            default=120,
            description="Request timeout in seconds. Increase if the LLM is slow to respond.",
        )

    def __init__(self):
        self.valves = self.Valves()

    def query_training_data(self, question: str) -> str:
        """
        Query the athlete's cycling training data with a natural language question.

        Use this tool when the user asks anything about their cycling training history,
        performance trends, fitness, fatigue, form (TSB), power output, or wants a
        session recommendation. Examples:
        - "Compare my last 90 days to the same period last year"
        - "How has my fitness been trending?"
        - "Recommend an interval session for today"
        - "What was my average TSS last month?"

        :param question: The natural language question about cycling training data.
        :return: A natural language answer grounded in the athlete's actual training data.
        """
        try:
            response = requests.post(
                f"{self.valves.CYCLING_RAG_URL}/query",
                json={"question": question},
                timeout=self.valves.TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()["answer"]
        except requests.exceptions.ConnectionError:
            return f"Could not connect to the Cycling RAG service at {self.valves.CYCLING_RAG_URL}. Is it running?"
        except requests.exceptions.Timeout:
            return "The Cycling RAG service timed out. The model may be slow — try increasing TIMEOUT_SECONDS in the tool valves."
        except requests.exceptions.HTTPError as e:
            return f"Cycling RAG service returned an error: {e}"
