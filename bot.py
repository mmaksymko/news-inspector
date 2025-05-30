import os
from dotenv import load_dotenv

load_dotenv('.env', override=True)
if os.getenv('ENVIRONMENT', '').lower() == 'local':
    load_dotenv('.env.local', override=True)

import logging
import logging_config
import sql

from telegram.ext import ApplicationBuilder

from handlers.admin_handler import AdminHandler
from handlers.start_handler import StartHandler
from handlers.analytics_handler import AnalyticsHandler

BOT_TOKEN = os.getenv('BOT_TOKEN')

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    StartHandler(app)
    analytics_handler = AnalyticsHandler(app)
    AdminHandler(app, analytics_handler)

    logging.info("Bot is running...")
    app.run_polling()
    

if __name__ == '__main__':
    main()
