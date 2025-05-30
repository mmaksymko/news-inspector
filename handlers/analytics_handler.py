import re
from typing import Callable
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
    CallbackQuery,
)
from telegram.ext import (
    Application,
    CallbackContext,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

from utils.url_shortener import decode_url, encode_url
from service.article_helper import create_from_url, create_from_message, get_urls
import service.analytics_facade as analytics

class AnalyticsHandler:
    MAX_URLS = 10

    FAKE_ACTION = 'fake_news'
    ANALYTICS_ACTIONS: dict[str, Callable | None] = {
        "🚨 Перевірити на фейк": "fake_news",
        "🏴 Виявити пропаганду": analytics.propaganda_detection,
        "⚠️ Перевірити клікбейт": analytics.clickbait_detection,
        "🏷️ Класифікувати за жанром": analytics.category_classification
    }

    def __init__(self, app: Application) -> None:
        self.URL_PREFIX = "url"
        self.ANALYTICS_PREFIX = "analytics"
        self.FAKE_PREFIX = "fake"
        
        self.fake_actions = {
            "algo": "🧠 Алгоритм",
            "db": "📚 База"
        }
        
        # Text messages handler
        text_filter = filters.TEXT & ~filters.COMMAND
        app.add_handler(MessageHandler(text_filter, self.on_text), group=0)

        # Callback queries for our prefixes
        self.callbacks = {
            self.URL_PREFIX: self._handle_url,
            self.ANALYTICS_PREFIX: self._handle_analytics,
            self.FAKE_PREFIX: self._handle_fake
        }
        prefixes = map(re.escape, [key+":" for key in self.callbacks.keys()])
        callback_pattern = f"^({'|'.join(prefixes)}).+"
        app.add_handler(CallbackQueryHandler(self.on_callback, pattern=callback_pattern))

    async def on_text(self, update: Update, context: CallbackContext) -> None:
        message = update.message
        urls = get_urls(message)
        if context.user_data.get('awaiting_fake') or context.user_data.get('awaiting_news'):
            return

        if not urls:
            await self._handle_message_article(update, context)
            return

        if len(urls) > self.MAX_URLS:
            await message.reply_text(f"Забагато URL (макс. {self.MAX_URLS}). Надішліть менше.")
            return

        if len(urls) == 1:
            await self._process_url(update, context, urls[0])
        else:
            keyboard = [
                [InlineKeyboardButton(u, callback_data=f"{self.URL_PREFIX}:{encode_url(u)}")]
                for u in urls
            ]
            await message.reply_text(
                "Оберіть URL для аналізу:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

    async def on_callback(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        await query.answer()
        data = query.data or ""
        
        if spl:=data.split(":"):
            await self.callbacks[spl[0]](query, context)
        else:
            await query.message.reply_text("Невідомий запит.")

    async def _handle_message_article(self, update: Update, context: CallbackContext) -> None:
        message = update.message
        article = create_from_message(message)
        if not article:
            await message.reply_text("Не вдалося отримати статтю.")
            return

        context.user_data["article"] = article
        await self._show_menu(update, context)

    async def _process_url(
        self,
        update_or_query: Update|CallbackQuery,
        context: CallbackContext,
        raw_url: str,
    ) -> None:
        # Notify user
        loading_text = "Завантажую статтю…"
        if isinstance(update_or_query, Update):
            await update_or_query.message.reply_text(loading_text)
        else:
            await edit_query(update_or_query, loading_text)

        article = create_from_url(raw_url)
        if not article:
            error_text = "Не вдалося отримати статтю."
            if isinstance(update_or_query, Update):
                await update_or_query.message.reply_text(error_text)
            else:
                await update_or_query.edit_message_text(error_text)
            return

        context.user_data["article"] = article
        await self._show_menu(update_or_query, context)

    async def _handle_url(self, query: CallbackQuery, context: CallbackContext) -> None:
        encoded = query.data.split("", 1)[1]
        raw = decode_url(encoded)
        await self._process_url(query, context, raw)

    async def _show_menu(
        self,
        update_or_query: Update|CallbackQuery,
        context: CallbackContext,
    ) -> None:
        keyboard = [
            [InlineKeyboardButton(label, callback_data=f"{self.ANALYTICS_PREFIX}:{label}")]
            for label in self.ANALYTICS_ACTIONS
        ]
        menu_text = "Оберіть тип аналізу:"
        markup = InlineKeyboardMarkup(keyboard)

        if hasattr(update_or_query, "edit_message_text"):
            await update_or_query.edit_message_text(text=menu_text, reply_markup=markup)
        else:
            await update_or_query.message.reply_text(text=menu_text, reply_markup=markup)

    async def _handle_analytics(self, query: CallbackQuery, context: CallbackContext) -> None:
        _, label = query.data.split(":", 1)
        article = context.user_data.get("article")
        if not article:
            await query.edit_message_text("Статтю не знайдено для аналізу.")
            return


        if self.ANALYTICS_ACTIONS[label] == self.FAKE_ACTION:
            fake_buttons = [
                [InlineKeyboardButton(name, callback_data=f"{self.FAKE_PREFIX}:{suffix}") for suffix, name in self.fake_actions.items()]
            ]
            await query.edit_message_text("Оберіть метод перевірки фейку:\n", reply_markup=InlineKeyboardMarkup(fake_buttons))
            return

        # Execute analytics function
        func = self.ANALYTICS_ACTIONS[label]  # type: ignore
        await query.edit_message_text(f"Виконую “{label}”…")
        result = func(article)  # type: ignore
        await query.message.reply_text(f"Результат аналізу:\n{result}")
        await self._offer_other_analytics(query, exclude=label)

    async def _handle_fake(self, query: CallbackQuery, context: CallbackContext) -> None:
        method = query.data.split(":", 1)[1]
        article = context.user_data.get("article")
        if not article:
            await query.edit_message_text("Статтю не знайдено для аналізу.")
            return

        # Call correct fake detection
        if method == "algo":
            await query.edit_message_text("🔍 Запускаю алгоритмічну перевірку…")
            result = analytics.fake_news_detection(article)
            include = "db"
        else:
            await query.edit_message_text("🔎 Перевіряю по базі фейків…")
            result = await analytics.database_fake_news_detection(article)
            include = "algo"
            
        include = { text: f"{self.FAKE_PREFIX}:{prefix}" for prefix, text in self.fake_actions.items() if prefix != method }
        await query.message.reply_text(f"Результат аналізу:\n{result}")
        await self._offer_other_analytics(query, exclude="🚨 Перевірити на фейк", include=include)

    async def _offer_other_analytics(
        self,
        query: CallbackQuery,
        exclude: str|None = None,
        include: dict[str, str] = {}
    ) -> None:
        options = [
            k for k in self.ANALYTICS_ACTIONS
            if k != exclude
        ]
        if not options:
            return

        buttons = (
            [[InlineKeyboardButton(k, callback_data=v)] for k, v in include.items()] +
            [[InlineKeyboardButton(k, callback_data=f"{self.ANALYTICS_PREFIX}:{k}")] for k in options]
        )
        await query.message.reply_text(
            "✅ Аналіз завершено.\n" \
            "Бажаєте провести інший тип аналізу? Оберіть варіант:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

async def edit_query(query: CallbackQuery, *args, **kwargs):
    await query.edit_message_text(*args, **kwargs)