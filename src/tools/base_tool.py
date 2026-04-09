"""
CogniTwin BaseTool
==================
Tüm araçların (Calendar, Gmail, Dosya, LMS) türeyeceği soyut taban sınıf.

Aşama-2 uygulama sırası:
  1. calendar_tool.py  — Google Calendar oku / etkinlik ekle
  2. gmail_tool.py     — Gmail oku / taslak oluştur
  3. file_tool.py      — Yerel dosya oku / ara
  4. lms_tool.py       — Moodle ödev/not sorgula

Router entegrasyonu (router.py Route.TASK → ToolAgent):
  decision.task_type → "calendar_read" | "calendar_add" |
                        "gmail_read"   | "gmail_send"   |
                        "file_read"    | "lms_check"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    success: bool
    content: str               # Kullanıcıya gösterilecek metin
    raw: dict | None = None    # Ham API yanıtı (debug / loglama için)
    error: str = ""


class BaseTool(ABC):
    """
    Soyut araç taban sınıfı.

    Alt sınıflar zorunlu olarak tanımlamalı:
        name        : str   (örn. "google_calendar")
        description : str   (router için kısa açıklama)
        execute()           (asıl iş burada yapılır)
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, query: str, **kwargs) -> ToolResult:
        """
        Aracı çalıştırır.

        Parameters
        ----------
        query : str
            Kullanıcının orijinal metni (araç ihtiyaç duyduğu parametreleri
            buradan çıkarabilir).
        **kwargs : dict
            Araç tipine özgü ek parametreler (user_id, date_range, vb.)

        Returns
        -------
        ToolResult
        """
        ...

    def safe_execute(self, query: str, **kwargs) -> ToolResult:
        """
        Hata yakalayıcı sarmalayıcı — araç çökmesi pipeline'ı durdurmaz.
        """
        try:
            return self.execute(query, **kwargs)
        except Exception as exc:
            return ToolResult(
                success=False,
                content="Araç çalıştırılırken bir hata oluştu.",
                error=str(exc),
            )
