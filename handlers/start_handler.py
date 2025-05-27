from telegram.ext import Application, ContextTypes, CommandHandler, CallbackContext
from telegram import KeyboardButton, ReplyKeyboardMarkup, Update

HELLO_MESSAGE = """
Привіт! 👋  
Я — Інспектор, твій розумний асистент у світі новин. 🧠📰  

Надішли мені текст, посилання або переслану новину — і я швидко проведу аналіз.  
Ось що я вмію:  
🔎 Визначаю жанр новини  
🚨 Перевіряю на фейки за допомогою алгоритмів та бази даних  
📢 Виявляю ознаки пропаганди  
🧲 Аналізую, чи заголовок є клікбейтним  

Просто надішли мені щось — і я все покажу 😉.

Щоб переглянути меню, натисни /menu.
"""

class StartHandler:
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(HELLO_MESSAGE)
    
    def __init__(self, app: Application):
        app.add_handler(CommandHandler("start", self.start))
