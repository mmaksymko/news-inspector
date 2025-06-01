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
    CommandHandler,
    CallbackQueryHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

import service.admin_service as admin_service
import service.analytics_facade as analytics_facade
from handlers.analytics_handler import AnalyticsHandler


class AdminHandler:
    ADMIN_PREFIX = "admin"
    MENU_PREFIX = "menu"
    AWAITING_FAKE = 1
    AWAITING_NEWS = 2

    STATS_HEADERS = {
        'fake_ml': "Визначення фейкових новин за допомогою алгоритму",
        'fake_db': "Визначення фейкових новин за допомогою бази даних",
        'propaganda': "Визначення пропаганди",
        'genres': "Визначення жанрів",
        'clickbait': "Визначення клікбейту",
    }

    def __init__(self, app: Application, analytics_handler: AnalyticsHandler) -> None:
        self.analytics_handler = analytics_handler

        app.add_handler(CommandHandler("admin", self.on_admin))

        app.add_handler(
            CallbackQueryHandler(
                self.on_callback,
                pattern=f"^{self.ADMIN_PREFIX}:get_stats$"
            )
        )

        app.add_handler(CommandHandler("menu", self.on_menu))
        app.add_handler(
            CallbackQueryHandler(
                self.on_menu_callback,
                pattern=f"^{self.MENU_PREFIX}:admin_panel$"
            )
        )

        # ConversationHandler для додавання фейку (без змін)
        conv_fake = ConversationHandler(
            entry_points=[
                CallbackQueryHandler(
                    self._start_add_fake,
                    pattern=f"^{self.ADMIN_PREFIX}:add_fake$"
                )
            ],
            states={
                self.AWAITING_FAKE: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self._receive_fake)
                ],
            },
            fallbacks=[
                CallbackQueryHandler(
                    self._cancel_add_fake,
                    pattern=f"^(?:{self.ADMIN_PREFIX}|{self.MENU_PREFIX}):back$"
                )
            ],
            per_user=True,
            per_chat=True,
        )
        app.add_handler(conv_fake, group=1)

        conv_analyze = ConversationHandler(
            entry_points=[
                CallbackQueryHandler(
                    self._start_analyze,
                    pattern=f"^(?:{self.ADMIN_PREFIX}|{self.MENU_PREFIX}):analyze$"
                )
            ],
            states={
                self.AWAITING_NEWS: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self._receive_news)
                ],
            },
            fallbacks=[
                CallbackQueryHandler(
                    self._cancel_analyze,
                    pattern=f"^(?:{self.ADMIN_PREFIX}|{self.MENU_PREFIX}):back$"
                )
            ],
            per_user=True,
            per_chat=True,
        )
        app.add_handler(conv_analyze, group=1)


    def _get_handle(self, update: Update, context: CallbackContext) -> str:
        if update.message and update.message.chat:
            return update.message.chat.username
        if context.args:
            return context.args[0]
        if update.callback_query and update.callback_query.from_user:
            return update.callback_query.from_user.username

    async def on_admin(self, update: Update, context: CallbackContext, is_admin: bool) -> None:
        context.user_data['is_admin'] = is_admin or admin_service.is_admin(self._get_handle(update, context))

        if not context.user_data['is_admin']:
            await update.message.reply_text("🛑 Ви не маєте прав адміністратора.")
            return

        await self._show_admin_menu(update, context)

    async def on_menu(self, update: Update, context: CallbackContext) -> None:
        handle = self._get_handle(update, context)
        is_admin = admin_service.is_admin(handle)
        context.user_data['is_admin'] = is_admin

        await self._show_menu(update, context, is_admin)

    async def on_menu_callback(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        await query.answer()

        _, action = query.data.split(':', 1)
        if action == "admin_panel":
            fake_update = Update(
                update.update_id,
                message=query.message
            )
            await self.on_admin(
                fake_update,
                context,
                admin_service.is_admin(self._get_handle(update, context))
            )

    async def on_callback(self, update: Update, context: CallbackContext) -> None:
        """
        This handler now ONLY matches 'admin:get_stats'.
        """
        query = update.callback_query
        await query.answer()

        _, action = query.data.split(':', 1)
        if action == 'get_stats':
            stats = admin_service.get_stats()
            formatted = '\n'.join([
                f'{self.STATS_HEADERS[name]}: {count}'
                for name, count in stats
            ])
            await query.edit_message_text(f"📊 Статистика за останні 30 днів:\n{formatted}")
            await self._show_admin_menu(query, context, reply=True)


    # ----------------------- FAKE‐ADDING FLOW (unchanged) -----------------------
    async def _start_add_fake(self, update: Update, context: CallbackContext) -> int:
        query = update.callback_query
        await query.answer()
        context.user_data['awaiting_fake'] = True

        back_button = InlineKeyboardButton("↩️ Назад", callback_data=f"{self.ADMIN_PREFIX}:back")
        await query.edit_message_text(
            "Введіть твердження для додавання як фейк:",
            reply_markup=InlineKeyboardMarkup([[back_button]])
        )
        return self.AWAITING_FAKE

    async def _receive_fake(self, update: Update, context: CallbackContext) -> int:
        statement = update.message.text.strip()
        analytics_facade.add_fake(statement)
        context.user_data.pop('awaiting_fake', None)

        await update.message.reply_text("✅ Твердження успішно додано як фейк.")
        await self._show_admin_menu(update, context)
        return ConversationHandler.END

    async def _cancel_add_fake(self, update: Update, context: CallbackContext) -> int:
        query = update.callback_query
        await query.answer()
        context.user_data.pop('awaiting_fake', None)
        await self._show_admin_menu(query, context)
        return ConversationHandler.END


    # -------------------- NEWS‐ANALYSIS FLOW --------------------
    async def _start_analyze(self, update: Update, context: CallbackContext) -> int:
        query = update.callback_query
        await query.answer()
        context.user_data['awaiting_news'] = True
        back_button = InlineKeyboardButton("↩️ Назад", callback_data=f"{self.MENU_PREFIX}:back")
        await query.edit_message_text(
            "Надішліть текст новини або перешліть повідомлення:",
            reply_markup=InlineKeyboardMarkup([[back_button]])
        )
        return self.AWAITING_NEWS

    async def _receive_news(self, update: Update, context: CallbackContext) -> int:
        await self.analytics_handler._handle_message_article(update, context)
        return ConversationHandler.END

    async def _cancel_analyze(self, update: Update, context: CallbackContext) -> int:
        query = update.callback_query
        await query.answer()
        context.user_data.pop('awaiting_news', None)

        if context.user_data.get('is_admin'):
            await self._show_admin_menu(query, context)
        else:
            is_admin = admin_service.is_admin(self._get_handle(update, context))
            await self._show_menu(query, context, is_admin)
        return ConversationHandler.END


    # -------------------- MENU RENDERERS --------------------
    async def _show_admin_menu(self, update_or_query, context: CallbackContext, reply: bool = False) -> None:
        buttons = [
            [InlineKeyboardButton("➕ Додати фейк", callback_data=f"{self.ADMIN_PREFIX}:add_fake")],
            [InlineKeyboardButton("📊 Переглянути статистику", callback_data=f"{self.ADMIN_PREFIX}:get_stats")],
            [InlineKeyboardButton("🕵️ Проаналізувати новину", callback_data=f"{self.ADMIN_PREFIX}:analyze")],
        ]
        markup = InlineKeyboardMarkup(buttons)
        menu_text = "Адмінські функції:"

        if hasattr(update_or_query, 'edit_message_text') and not reply:
            await update_or_query.edit_message_text(text=menu_text, reply_markup=markup)
        else:
            await update_or_query.message.reply_text(text=menu_text, reply_markup=markup)

    async def _show_menu(self, update_or_query, context: CallbackContext, is_admin: bool) -> None:
        buttons = []
        if is_admin:
            buttons.append([
                InlineKeyboardButton(
                    "🧑‍💻 Панель адміністратора",
                    callback_data=f"{self.MENU_PREFIX}:admin_panel"
                )
            ])
        buttons.append([
            InlineKeyboardButton(
                "🕵️ Проаналізувати новину",
                callback_data=f"{self.MENU_PREFIX}:analyze"
            )
        ])
        markup = InlineKeyboardMarkup(buttons)
        text = "Оберіть дію:"

        if hasattr(update_or_query, 'edit_message_text'):
            await update_or_query.edit_message_text(text=text, reply_markup=markup)
        else:
            await update_or_query.message.reply_text(text=text, reply_markup=markup)
