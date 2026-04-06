"""
pharmaai/inference/notifications.py
Multi-channel notification dispatcher.
Supports: Twilio SMS, Slack webhook, SendGrid email, FCM push.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from twilio.rest import Client as TwilioClient

from pharmaai.core.config import get_settings
from pharmaai.core.schemas import AlertSeverity, NotificationChannel, NotificationPayload

logger = logging.getLogger("pharmaai.notifications")


SEVERITY_EMOJI = {
    AlertSeverity.LOW: "ℹ️",
    AlertSeverity.MEDIUM: "⚠️",
    AlertSeverity.HIGH: "🚨",
    AlertSeverity.CRITICAL: "🆘",
}


class NotificationDispatcher:
    def __init__(self):
        self._settings = get_settings().notifications
        self._http = httpx.AsyncClient(timeout=15)

    # ── Slack ─────────────────────────────────────────────────────────────────

    async def send_slack(self, payload: NotificationPayload) -> bool:
        url = self._settings.slack.webhook_url
        if not url:
            logger.warning("Slack webhook URL not configured.")
            return False
        emoji = SEVERITY_EMOJI.get(payload.severity, "⚠️")
        msg = {
            "text": f"{emoji} *{payload.title}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{emoji} *{payload.title}*\n{payload.body}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"Severity: `{payload.severity.value}`"}
                    ],
                },
            ],
        }
        try:
            r = await self._http.post(url, json=msg)
            r.raise_for_status()
            logger.info("Slack notification sent: %s", payload.title)
            return True
        except Exception as exc:
            logger.error("Slack send failed: %s", exc)
            return False

    # ── Twilio SMS ────────────────────────────────────────────────────────────

    async def send_sms(self, payload: NotificationPayload) -> bool:
        cfg = self._settings.twilio
        if not cfg.account_sid or not cfg.auth_token:
            logger.warning("Twilio credentials not configured.")
            return False
        try:
            client = TwilioClient(cfg.account_sid, cfg.auth_token)
            body = f"[PharmaAI {payload.severity.value.upper()}] {payload.title}: {payload.body}"
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    body=body[:160],
                    from_=cfg.from_number,
                    to=payload.recipient,
                ),
            )
            logger.info("SMS sent to %s", payload.recipient)
            return True
        except Exception as exc:
            logger.error("Twilio SMS failed: %s", exc)
            return False

    # ── SendGrid Email ────────────────────────────────────────────────────────

    async def send_email(self, payload: NotificationPayload) -> bool:
        cfg = self._settings.sendgrid
        if not cfg.api_key:
            logger.warning("SendGrid API key not configured.")
            return False
        try:
            sg = SendGridAPIClient(cfg.api_key)
            msg = Mail(
                from_email=cfg.from_email,
                to_emails=payload.recipient,
                subject=f"[PharmaAI Alert] {payload.title}",
                html_content=(
                    f"<h2>{payload.title}</h2>"
                    f"<p>{payload.body}</p>"
                    f"<p><small>Severity: {payload.severity.value}</small></p>"
                ),
            )
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: sg.send(msg)
            )
            logger.info("Email sent to %s", payload.recipient)
            return True
        except Exception as exc:
            logger.error("SendGrid email failed: %s", exc)
            return False

    # ── FCM Push ──────────────────────────────────────────────────────────────

    async def send_fcm(self, payload: NotificationPayload) -> bool:
        cfg = self._settings.fcm
        if not cfg.server_key:
            logger.warning("FCM server key not configured.")
            return False
        try:
            body = {
                "to": payload.recipient,
                "notification": {
                    "title": payload.title,
                    "body": payload.body,
                },
                "data": {"severity": payload.severity.value},
            }
            r = await self._http.post(
                "https://fcm.googleapis.com/fcm/send",
                headers={
                    "Authorization": f"key={cfg.server_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            r.raise_for_status()
            logger.info("FCM push sent to %s", payload.recipient)
            return True
        except Exception as exc:
            logger.error("FCM push failed: %s", exc)
            return False

    # ── Dispatcher ────────────────────────────────────────────────────────────

    async def dispatch(self, payload: NotificationPayload) -> dict[str, bool]:
        channels = payload.channels or list(NotificationChannel)
        tasks: dict[str, Any] = {}

        if NotificationChannel.SLACK in channels:
            tasks["slack"] = asyncio.create_task(self.send_slack(payload))
        if NotificationChannel.TWILIO in channels and payload.recipient:
            tasks["sms"] = asyncio.create_task(self.send_sms(payload))
        if NotificationChannel.EMAIL in channels and payload.recipient:
            tasks["email"] = asyncio.create_task(self.send_email(payload))
        if NotificationChannel.FCM in channels and payload.recipient:
            tasks["fcm"] = asyncio.create_task(self.send_fcm(payload))

        results: dict[str, bool] = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as exc:
                logger.error("Channel %s failed: %s", name, exc)
                results[name] = False
        return results

    async def alert_adverse_event(
        self,
        drug: str,
        reactions: str,
        severity: AlertSeverity = AlertSeverity.HIGH,
        recipients: list[str] | None = None,
    ) -> None:
        """Convenience method for adverse event alerts."""
        payload = NotificationPayload(
            title=f"New Adverse Event: {drug}",
            body=f"Drug: {drug}\nReactions: {reactions}",
            severity=severity,
            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
        )
        # Broadcast to all recipients
        for recipient in (recipients or []):
            payload.recipient = recipient
            await self.dispatch(payload)


dispatcher = NotificationDispatcher()