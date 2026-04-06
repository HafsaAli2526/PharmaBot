"""
rasa_bot/actions.py
Custom Rasa actions that call the PharmaAI retrieval & RAG pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

logger = logging.getLogger("pharmaai.rasa.actions")


def _run(coro):
    """Run async coroutine from sync Rasa action."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class ActionSearchAdverseEvents(Action):
    def name(self) -> Text:
        return "action_search_adverse_events"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        drug = tracker.get_slot("drug_name") or tracker.latest_message.get("text", "")
        query = f"adverse events side effects {drug}"

        async def _search():
            from pharmaai.retrieval.search import hybrid_search, SearchFilters
            from pharmaai.core.schemas import Domain
            filters = SearchFilters(domain=Domain.PHARMACOVIGILANCE)
            return await hybrid_search.search(query, top_k=5, filters=filters)

        try:
            results = _run(_search())
            if results:
                top = results[0].document
                dispatcher.utter_message(
                    text=f"Here are adverse events related to {drug}:\n\n"
                         f"**{top.title}** ({top.source})\n{top.content[:300]}…"
                )
            else:
                dispatcher.utter_message(
                    text=f"No adverse event data found for '{drug}'. Please check the drug name."
                )
        except Exception as exc:
            logger.error("Adverse event search failed: %s", exc)
            dispatcher.utter_message(text="Sorry, I encountered an error searching adverse events.")

        return [SlotSet("drug_name", drug)]


class ActionSummariseTrial(Action):
    def name(self) -> Text:
        return "action_summarise_trial"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        nct_id = tracker.get_slot("nct_id")
        query = tracker.latest_message.get("text", "")
        search_query = f"clinical trial {nct_id or query}"

        async def _search():
            from pharmaai.retrieval.search import hybrid_search, SearchFilters
            from pharmaai.core.schemas import Domain
            filters = SearchFilters(domain=Domain.RND)
            return await hybrid_search.search(search_query, top_k=3, filters=filters)

        try:
            results = _run(_search())
            if results:
                top = results[0].document
                dispatcher.utter_message(
                    text=f"Clinical Trial Summary:\n\n**{top.title}**\n{top.content[:400]}…"
                )
            else:
                dispatcher.utter_message(text="No clinical trial data found.")
        except Exception as exc:
            logger.error("Trial search failed: %s", exc)
            dispatcher.utter_message(text="Sorry, I encountered an error fetching trial data.")

        return []


class ActionGetRegulatoryInfo(Action):
    def name(self) -> Text:
        return "action_get_regulatory_info"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        query = tracker.latest_message.get("text", "")

        async def _rag():
            from pharmaai.inference.rag import rag_pipeline
            from pharmaai.core.schemas import AskRequest, Domain
            req = AskRequest(question=query, domain=Domain.REGULATION, top_k=5)
            return await rag_pipeline.answer(req)

        try:
            response = _run(_rag())
            dispatcher.utter_message(text=response.answer)
        except Exception as exc:
            logger.error("Regulatory search failed: %s", exc)
            dispatcher.utter_message(text="Sorry, I couldn't retrieve regulatory information.")

        return []


class ActionAnalyseFormula(Action):
    def name(self) -> Text:
        return "action_analyse_formula"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        compound = tracker.get_slot("compound") or tracker.latest_message.get("text", "")
        query = f"chemical formula dosage interaction {compound}"

        async def _search():
            from pharmaai.retrieval.search import hybrid_search, SearchFilters
            from pharmaai.core.schemas import Domain
            filters = SearchFilters(domain=Domain.FORMULAS)
            return await hybrid_search.search(query, top_k=5, filters=filters)

        try:
            results = _run(_search())
            if results:
                top = results[0].document
                dispatcher.utter_message(
                    text=f"Formula / Interaction Info:\n\n**{top.title}**\n{top.content[:400]}…"
                )
            else:
                dispatcher.utter_message(text="No formula data found for that compound.")
        except Exception as exc:
            logger.error("Formula analysis failed: %s", exc)
            dispatcher.utter_message(text="Sorry, I couldn't analyse the formula.")

        return [SlotSet("compound", compound)]


class ActionSendNotification(Action):
    def name(self) -> Text:
        return "action_send_notification"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        async def _notify():
            from pharmaai.inference.notifications import dispatcher as notif_dispatcher
            from pharmaai.core.schemas import NotificationPayload, AlertSeverity, NotificationChannel
            payload = NotificationPayload(
                title="PharmaAI Alert",
                body=tracker.latest_message.get("text", ""),
                severity=AlertSeverity.MEDIUM,
                channels=[NotificationChannel.SLACK],
            )
            await notif_dispatcher.dispatch(payload)

        try:
            _run(_notify())
            dispatcher.utter_message(text="Notification sent successfully.")
        except Exception as exc:
            logger.error("Notification failed: %s", exc)
            dispatcher.utter_message(text="Failed to send notification.")

        return []