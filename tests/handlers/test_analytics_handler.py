import os
import dotenv
import pytest
from types import SimpleNamespace

dotenv.load_dotenv(r'M:\Personal\SE\bachelors\create_reqs\news-inspector\.env.test', override=True)
BOT_TOKEN = os.getenv('BOT_TOKEN')

os.environ["PINECONE_API_KEY"] = "fake-api-key"
os.environ["PINECONE_INDEX"] = "fake-index-name"
os.environ["PINECONE_MODEL_NAME"] = "intfloat/multilingual-e5-large"

import pinecone

class DummyIndex:
    def __init__(self, index_name=None):
        pass
    def query(self, *args, **kwargs):
        return {"matches": []}
    def upsert(self, *args, **kwargs):
        return SimpleNamespace(upserted_count=0)

class DummyPineconeClient:
    def __init__(self, api_key, environment):
        pass
    def Index(self, name):
        return DummyIndex(name)
    def describe_index(self, name):
        return SimpleNamespace(host="fake-host", name=name, dimension=1)
    def list_indexes(self):
        return SimpleNamespace(indexes=[])

pinecone.Pinecone = DummyPineconeClient


from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application

import service.analytics_facade as analytics
from utils.url_shortener import encode_url, decode_url
from service.article_helper import create_from_url, create_from_message, get_urls
from handlers.analytics_handler import AnalyticsHandler, edit_query

@pytest.fixture
def app():
    return Application.builder().token(BOT_TOKEN).build()


class DummyMessage:
    """
    A dummy message that:
    - Records every .reply_text(...) or .edit_message_text(...) call in .texts, .replied, .reply_markups.
    - Provides both edit_message_text(...) and an alias edit_text(...) in case the handler calls either.
    - Supplies a .chat and .text field so that callback_query.message.chat/id or .text never fails.
    """
    def __init__(self):
        self.texts = []           # All message texts sent or edited
        self.reply_markups = []   # All reply_markups used
        self.replied = []         # List of (text, reply_markup) tuples
        self.message_id_counter = 123
        self.chat = SimpleNamespace(id=1)
        self.text = ""            # In case handler reads message.text
        self.message_id = 123

    async def _edit_text_internal(self, text, reply_markup=None, message_id_to_edit=None):
        # Pretend to "edit" a message (or create a new record of edits).
        self.texts.append(text)
        self.replied.append((text, reply_markup))
        self.reply_markups.append(reply_markup)
        self.text = text  # Update the current text
        edited_message_obj = SimpleNamespace(
            message_id=message_id_to_edit or self.message_id_counter,
            chat=SimpleNamespace(id=1),
            text=text,
        )
        # Add methods to the returned object
        edited_message_obj.edit_message_text = lambda t, rm=None: self._edit_text_internal(t, rm, message_id_to_edit or self.message_id_counter)
        edited_message_obj.edit_text = lambda t, rm=None: self._edit_text_internal(t, rm, message_id_to_edit or self.message_id_counter)
        return edited_message_obj

    async def reply_text(self, text, reply_markup=None):
        # Simulate sending a new message in reply:
        self.texts.append(text)
        self.replied.append((text, reply_markup))
        self.reply_markups.append(reply_markup)
        current_message_id = self.message_id_counter
        self.message_id_counter += 1

        sent_message_obj = SimpleNamespace(
            message_id=current_message_id,
            chat=SimpleNamespace(id=1),
            text=text,
        )
        # Add methods to the returned object
        sent_message_obj.edit_message_text = lambda t, rm=None: self._edit_text_internal(t, rm, current_message_id)
        sent_message_obj.edit_text = lambda t, rm=None: self._edit_text_internal(t, rm, current_message_id)
        return sent_message_obj

    async def edit_message_text(self, text, reply_markup=None):
        # This simulates editing "the incoming message" itself.
        return await self._edit_text_internal(text, reply_markup,
                 self.message_id_counter - 1 if self.message_id_counter > 123 else 123)

    async def edit_text(self, text, reply_markup=None):
        return await self._edit_text_internal(text, reply_markup,
                 self.message_id_counter - 1 if self.message_id_counter > 123 else 123)


class DummyCallbackQuery:
    """
    A dummy CallbackQuery:
    - .data is the callback string, e.g. "url:abc" or "analytics:⚠️ Перевірити клікбейт"
    - .message is a DummyMessage, so .edit_message_text(...) calls route to that DummyMessage.
    """
    def __init__(self, data, message: DummyMessage):
        self.data = data
        self.message = message  # This message should be an instance of DummyMessage
        self.answered = False

    async def answer(self):
        self.answered = True

    async def edit_message_text(self, text, reply_markup=None):
        # Delegate to the DummyMessage
        return await self.message.edit_message_text(text, reply_markup=reply_markup)

    async def edit_text(self, text, reply_markup=None):
        return await self.message.edit_message_text(text, reply_markup=reply_markup)


def make_update_with_message(text: str, message_obj: DummyMessage):
    """
    Build a fake Update for an incoming message:
    - update.message is a SimpleNamespace with .text, .reply_text, .edit_message_text, .edit_text, .chat, .message_id.
    - update.effective_message = update.message
    - update.effective_chat.id = 1
    """
    msg = SimpleNamespace(
        text=text,
        message_id=message_obj.message_id_counter,
        chat=SimpleNamespace(id=1),
    )
    # Attach the DummyMessage's methods so handler.reply_text / handler.edit_* routes to DummyMessage
    msg.reply_text = message_obj.reply_text
    msg.edit_message_text = message_obj.edit_message_text
    msg.edit_text = message_obj.edit_text

    upd = SimpleNamespace(
        message=msg,
        effective_message=msg,
        effective_chat=SimpleNamespace(id=1),
    )
    # Add edit methods to the update's effective_message as well
    upd.effective_message.edit_message_text = message_obj.edit_message_text
    upd.effective_message.edit_text = message_obj.edit_text
    return upd


def make_update_with_callback(query_obj: DummyCallbackQuery):
    """
    Build a fake Update for a CallbackQuery:
    - update.callback_query is our DummyCallbackQuery
    - update.effective_message = query_obj.message (so handler can do update.effective_message.edit_* without error)
    - update.effective_chat.id = 1
    """
    upd = SimpleNamespace(
        callback_query=query_obj,
        effective_message=query_obj.message,
        effective_chat=SimpleNamespace(id=1),
    )
    return upd


@pytest.mark.asyncio
async def test_on_text_no_urls_and_create_fails(monkeypatch, app):
    user_data_dict = {}
    mock_context = SimpleNamespace(user_data=user_data_dict)

    # Force get_urls(...) → no URLs
    monkeypatch.setattr("handlers.analytics_handler.get_urls", lambda msg: [])
    # Force create_from_message(...) to return None
    async def mock_create_from_message(*args, **kwargs):
        return None
    monkeypatch.setattr("handlers.analytics_handler.create_from_message", mock_create_from_message)

    dummy_msg_replier = DummyMessage()
    update = make_update_with_message("Some text with no URLs", dummy_msg_replier)
    handler = AnalyticsHandler(app)

    await handler.on_text(update, mock_context)

    assert dummy_msg_replier.texts[-1] == "Оберіть тип аналізу:"
    print(dummy_msg_replier)


@pytest.mark.asyncio
async def test_on_text_multiple_urls_shows_selection(monkeypatch, app):
    user_data_dict = {}
    mock_context = SimpleNamespace(user_data=user_data_dict)

    urls = ["http://a.com", "http://b.com"]
    monkeypatch.setattr("handlers.analytics_handler.get_urls", lambda msg: urls)

    dummy_msg_replier = DummyMessage()
    update = make_update_with_message("two urls", dummy_msg_replier)
    handler = AnalyticsHandler(app)

    await handler.on_text(update, mock_context)

    last_text, last_markup = dummy_msg_replier.replied[-1]
    assert last_text == "Оберіть URL для аналізу:"
    assert isinstance(last_markup, InlineKeyboardMarkup)

    # The buttons' text should be exactly the two URLs in order:
    button_texts = [btn.text for row in last_markup.inline_keyboard for btn in row]
    assert button_texts == urls

    # Each callback_data is "url:<encoded_part>"
    for row, u in zip(last_markup.inline_keyboard, urls):
        btn = row[0]
        assert btn.callback_data.startswith("url:")
        encoded_part = btn.callback_data.split(":", 1)[1]
        assert encoded_part, "Encoded part of URL callback data should not be empty"

        # The real decode_url(...) should recover the original:
        decoded = decode_url(encoded_part)
        print(f"Decoded URL: {decoded}")
        assert decoded == u


@pytest.mark.asyncio
async def test_handle_analytics_without_article(monkeypatch, app):
    user_data_dict = {}
    mock_context = SimpleNamespace(user_data=user_data_dict)

    label = "🏴 Виявити пропаганду"
    data = f"analytics:{label}"

    dummy_msg_replier = DummyMessage()
    query = DummyCallbackQuery(data, dummy_msg_replier)
    update = make_update_with_callback(query)
    handler = AnalyticsHandler(app)
    user_data_dict.clear()

    await handler.on_callback(update, mock_context)

    assert "Статтю не знайдено для аналізу." in dummy_msg_replier.texts[-1]


@pytest.mark.asyncio
async def test_handle_analytics_fake_option(monkeypatch, app):
    # Use SimpleNamespace so handler.article.title works if needed
    user_data_dict = {"article": SimpleNamespace(foo="bar", title="Fake Article Test")}
    mock_context = SimpleNamespace(user_data=user_data_dict)

    label = "🚨 Перевірити на фейк"
    data = f"analytics:{label}"
    dummy_msg_replier = DummyMessage()
    query = DummyCallbackQuery(data, dummy_msg_replier)
    update = make_update_with_callback(query)
    handler = AnalyticsHandler(app)

    await handler.on_callback(update, mock_context)

    last_text, last_markup = dummy_msg_replier.replied[-1]
    assert "Оберіть метод перевірки фейку" in last_text
    assert isinstance(last_markup, InlineKeyboardMarkup)
    labels = [btn.text for row in last_markup.inline_keyboard for btn in row]
    assert set(labels) == set(handler.fake_actions.values())


@pytest.mark.asyncio
async def test_handle_analytics_real_action_calls_function(monkeypatch, app):
    # Use SimpleNamespace so handler.article.title is accessible
    article = SimpleNamespace(foo="bar", title="Clickbait Test Article", text="Some article text")
    user_data_dict = {"article": article}
    mock_context = SimpleNamespace(user_data=user_data_dict)

    label = "⚠️ Перевірити клікбейт"
    data = f"analytics:{label}"
    dummy_msg_replier = DummyMessage()
    query = DummyCallbackQuery(data, dummy_msg_replier)
    update = make_update_with_callback(query)

    # Monkey‐patch the clickbait_detection function to return a known value
    monkeypatch.setattr(analytics, "clickbait_detection", lambda art: "FAKE_RESULT")
    handler = AnalyticsHandler(app)

    await handler.on_callback(update, mock_context)

    # The handler should send "Виконую …" and then "Результат аналізу:\nFAKE_RESULT" etc.
    assert any(
        "Виконую" in txt and "Перевірити клікбейт" in txt
        for txt in dummy_msg_replier.texts
    )
    assert any(txt.startswith("Результат аналізу:\n") for txt in dummy_msg_replier.texts)

    final_reply_text, final_reply_markup = dummy_msg_replier.replied[-1]
    assert "✅ Аналіз завершено." in final_reply_text

    offered_labels = [btn.text for row in final_reply_markup.inline_keyboard for btn in row]
    assert label not in offered_labels
    expected_remaining = set(handler.ANALYTICS_ACTIONS.keys()) - {label}
    assert set(offered_labels).issubset(expected_remaining)
    assert len(set(offered_labels)) == len(expected_remaining)


@pytest.mark.asyncio
async def test_handle_fake_algo_and_offer_followup(monkeypatch, app):
    article = SimpleNamespace(foo="bar", title="Algo Fake Test", text="Some article text")
    user_data_dict = {"article": article}
    mock_context = SimpleNamespace(user_data=user_data_dict)

    data = "fake:algo"  # The handler expects something like this for "Алгоритм" fake‐check
    dummy_msg_replier = DummyMessage()
    query = DummyCallbackQuery(data, dummy_msg_replier)
    update = make_update_with_callback(query)

    monkeypatch.setattr(analytics, "fake_news_detection", lambda art: "SOME_FAKE")
    async def mock_db_fake_detection(*args, **kwargs):
        return "DB_FAKE"
    monkeypatch.setattr(analytics, "database_fake_news_detection", mock_db_fake_detection)
    handler = AnalyticsHandler(app)

    await handler.on_callback(update, mock_context)

    # It should run "Алгоритмічну перевірку" and return SOME_FAKE
    assert any("🔍 Запускаю алгоритмічну перевірку" in txt for txt in dummy_msg_replier.texts)
    assert any(txt == "Результат аналізу:\nSOME_FAKE" for txt in dummy_msg_replier.texts)

    markup = dummy_msg_replier.replied[-1][1]
    btn_texts = [btn.text for row in markup.inline_keyboard for btn in row]

    # "📚 База" (the other fake‐method) should be present
    assert "📚 База" in btn_texts
    # The "fake:algo" callback_data (just used) should not appear again
    assert not any(btn.callback_data == "fake:algo" for row in markup.inline_keyboard for btn in row)

    analytics_labels_expected = set(handler.ANALYTICS_ACTIONS.keys()) - {"🚨 Перевірити на фейк"}
    all_offered_buttons_texts = {btn.text for row in markup.inline_keyboard for btn in row}

    # The other fake method plus all other main analytics actions:
    assert handler.fake_actions["db"] in all_offered_buttons_texts
    for main_action_label in analytics_labels_expected:
        assert main_action_label in all_offered_buttons_texts

    expected_button_set = analytics_labels_expected | {handler.fake_actions["db"]}
    assert all_offered_buttons_texts == expected_button_set

