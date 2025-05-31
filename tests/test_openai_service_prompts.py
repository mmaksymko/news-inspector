import os
import dotenv
import openai
import json
from typing import List, Dict
from service.openai_service import analyze_fake, extract_claims, get_completion

dotenv.load_dotenv(r'M:\Personal\SE\bachelors\create_reqs\news-inspector\.env.test', override=True)
openai.api_key = os.getenv('OPENAI_TOKEN')

# =============================================================================
# 0. Значно розширений датасет (словник із численними тестовими кейсами)
# =============================================================================
DATASET = {
    # Тестові кейси для analyze_fake
    "analyze_fake": [
        {
            "known_fakes": [
                "Вакцина від COVID-19 викликає безпліддя у 90% людей.",
                "Чипізація через 5G-мережу активує дистанційне управління мозком."
            ],
            "to_check": [
                "Чипізація через 5G-мережу активує дистанційне управління мозком."
            ],
            "description": "Точно такий самий фейк, що й у списку."
        },
        {
            "known_fakes": [
                "Вакцина від COVID-19 викликає безпліддя у 90% людей.",
                "Чипізація через 5G-мережу активує дистанційне управління мозком."
            ],
            "to_check": [
                "Вакцина може спричиняти короткочасні побічні ефекти, але не безпліддя."
            ],
            "description": "Нове твердження, не вказане в known_fakes."
        },
        {
            "known_fakes": [
                "Солона вода може вилікувати рак.",
                "Темна матерія керує нашими думками."
            ],
            "to_check": [
                "Солона вода лікує рак."
            ],
            "description": "Парафразований варіант фейкового твердження про лікування раку."
        },
        {
            "known_fakes": [],
            "to_check": [
                "Літальний апарат із домашнього винаходу вже здійснив політ на Марс."
            ],
            "description": "known_fakes порожній, тому ніяке твердження не повинно позначатися як фейк."
        },
        {
            "known_fakes": [
                "Металеві гречки викликають рак шлунка.",
                "Чіпи у пластикових пляшках збирають дані про споживачів без їх згоди."
            ],
            "to_check": [
                "Металеві гречки викликають рак шлунка.",
                "Пластикові пляшки містять мікрочіпи для стеження за кожним."
            ],
            "description": "Два твердження: перше точно з known_fakes, друге — новий фейк."
        },
        {
            "known_fakes": [
                "Сприйняття сейсмічних хвиль через особливі окуляри змінює свідомість.",
                "Наліпки з QR-кодом на деревах транслюють геноцидальну пропаганду."
            ],
            "to_check": [
                "Окуляри для сейсмічних хвиль змінюють свідомість.",
                "QR-коди на деревах транслюють геноцидальну пропаганду."
            ],
            "description": "Два твердження, синтаксично змінені, але логіка фейку збережена."
        },
    ],

    # Тестові кейси для extract_claims — зараз список із кількох статей
    "extract_claims": [
        {
            "article_ukrainian": """
            У Києві відкрили новий сучасний центр для сортування сміття. Центр здатний обробляти
            до 100 тонн відходів на добу. Міська влада заявила, що це зменшить кількість
            непереробленого сміття на 30% уже в цьому році. Частину коштів на будівництво
            виділили з місцевого бюджету, а частину – з гранту ЄС. Водночас громадські активісти
            критикують, що новий комплекс не матиме власної станції переробки пластику.
            """,
            "max_claims": 5,
            "description": "Новий сортувальний центр у Києві"
        },
        {
            "article_ukrainian": """
            У Львові відремонтували історичний кам'яний міст, датований XVIII століттям. 
            Міст витримує навантаження до 20 тонн, що дозволяє пустити через нього новий маршрут 
            муніципального транспорту. Ремонт обійшовся місту у 15 мільйонів гривень, з яких 
            5 мільйонів надали з обласного бюджету, а 10 мільйонів — украдені під виглядом гранту. 
            За оцінками експертів, термін служби мосту після реставрації становитиме не менше 50 років. 
            Місцеві краєзнавці стверджують, що під час робіт знайшли рідкісний фрагмент кераміки XVII століття.
            """,
            "max_claims": 6,
            "description": "Ремонт кам'яного мосту у Львові"
        },
        {
            "article_ukrainian": """
            У Харкові відкрили нову дитячу лікарню на 250 ліжок. Вона оснащена сучасним 
            обладнанням для діагностики та інтенсивної терапії. Будівництво вели 18 місяців, 
            а сума вкладень склала 200 мільйонів гривень. Уряди кількох країн ЄС надали по 10 мільйонів 
            гривень кожен, решту коштів виділив місцевий бюджет. Головний лікар заявив, що 
            якість медичних послуг зросте на 40% уже протягом першого року. Ба більше, першу дитину 
            прооперували в новому корпусі вже на третій день після відкриття.
            """,
            "max_claims": 7,
            "description": "Відкриття дитячої лікарні в Харкові"
        },
        {
            "article_ukrainian": """
            У Дніпрі відбулося щорічне свято традиційних ремесел. На ярмарок приїхали майстри 
            із 12 регіонів України. Було представлено понад 200 виробів із деревини, кераміки, 
            текстилю та металу. Організатори повідомили, що кількість відвідувачів цього року зросла
            на 25% порівняно з минулим. Також вперше провели майстер-клас із виготовлення скляних 
            прикрас, де взяли участь понад 50 дітей. Під час фестивалю зібрали близько 100 тисяч гривень 
            на підтримку локальних культурних ініціатив.
            """,
            "max_claims": 6,
            "description": "Щорічне свято ремесел у Дніпрі"
        },
    ]
}

# =============================================================================
# 1. Оновлений тест analyze_fake (тепер із розширеним набором кейсів)
# =============================================================================

def test_analyze_fake():
    """
    Тестуємо analyze_fake() за допомогою даних із DATASET["analyze_fake"].
    """
    test_cases: List[Dict] = DATASET["analyze_fake"]

    for idx, case in enumerate(test_cases, start=1):
        result = analyze_fake(case["known_fakes"], case["to_check"])

        # Перевірка формату відповіді
        assert isinstance(result, dict), \
            f"Test case #{idx} («{case['description']}») failed: Result is not a dict"
        assert "message" in result and "verdict" in result, \
            f"Test case #{idx} («{case['description']}») failed: Missing keys in result"
        assert isinstance(result["verdict"], bool), \
            f"Test case #{idx} («{case['description']}») failed: Verdict is not boolean"

        # Додаткове оцінювання через OpenAI
        evaluation_prompt = (
            "Ти отримав від fact_checker функцію analyze_fake та ось її JSON-відповідь:\n\n"
            f"{json.dumps(result, ensure_ascii=False, indent=2)}\n\n"
            "1) Перевір, будь ласка, що це справді дійсний JSON-об'єкт із двома ключами: "
            "\"message\" (рядок українською) і \"verdict\" (булеве значення). "
            "2) Визнач, чи поточний “verdict” коректний з огляду на те, що ти знаєш про відомі фейкові твердження. "
            "Напиши короткий коментар українською мовою, причому у форматі:\n"
            "- valid_format: true/false\n"
            "- verdict_correct: true/false\n"
            "- comment: <короткий коментар українською>\n"
        )
        eval_response = get_completion(
            prompt="Evaluate analyze_fake output",
            content=evaluation_prompt,
            temperature=0,
            model="gpt-4o-mini",
            max_tokens=256,
            format="json_object"
        )
        eval_json = json.loads(eval_response)

        assert isinstance(eval_json, dict), \
            f"Test case #{idx} («{case['description']}») failed: OpenAI evaluation is not a dict"
        assert eval_json.get("valid_format") is True, \
            f"Test case #{idx} («{case['description']}») failed: Invalid format"
        assert "verdict_correct" in eval_json, \
            f"Test case #{idx} («{case['description']}») failed: Missing verdict_correct in evaluation"

# =============================================================================
# 2. Оновлений тест extract_claims (з кількома статтями)
# =============================================================================

def test_extract_claims():
    """
    Тестуємо extract_claims() за допомогою кожної статті із DATASET["extract_claims"].
    """
    test_articles: List[Dict] = DATASET["extract_claims"]

    for idx, article_case in enumerate(test_articles, start=1):
        article_text = article_case["article_ukrainian"]
        max_claims = article_case["max_claims"]
        description = article_case.get("description", f"Case #{idx}")

        extracted_claims = extract_claims(article_text, max_claims=max_claims)

        # Базові асерти
        assert isinstance(extracted_claims, list), \
            f"Test case #{idx} («{description}») failed: Extracted claims should be a list"
        assert all(isinstance(claim, str) for claim in extracted_claims), \
            f"Test case #{idx} («{description}») failed: All claims should be strings"

        # Додаткове оцінювання через OpenAI
        evaluation_prompt_2 = (
            "Ти отримав від fact_checker функцію extract_claims та ось перелік "
            "витягнутих з української статті тверджень:\n\n"
            f"{json.dumps(extracted_claims, ensure_ascii=False, indent=2)}\n\n"
            "Перевір, будь ласка, кожне твердження на такі критерії:\n"
            "1) Чи це по-справжньому самостійна, граматично коректна і конкретна фраза без зайвих деталей.\n"
            "2) Чи кожне твердження унікальне (повторю або дубльоване значення маркуй як помилку).\n"
            "3) Чи всі твердження українською мовою.\n"
            "Напиши у форматі JSON-масиву об'єктів, де кожен об'єкт має поля:\n"
            "- \"claim\": <сам текст твердження>\n"
            "- \"valid\": true/false\n"
            "- \"comment\": <короткий коментар, чому воно неприпустиме або коректне>\n"
        )
        eval_response_2 = get_completion(
            prompt="Evaluate extract_claims output",
            content=evaluation_prompt_2,
            temperature=0,
            model="gpt-4o-mini",
            max_tokens=1024,
            format="json_object"
        )
        eval_json_2 = json.loads(eval_response_2)

        assert isinstance(eval_json_2, dict), \
            f"Test case #{idx} («{description}») failed: OpenAI evaluation should return a dictionary"
        assert "claims" in eval_json_2, \
            f"Test case #{idx} («{description}») failed: Response should contain a 'claims' key"
        claims_list = eval_json_2["claims"]
        assert isinstance(claims_list, list), \
            f"Test case #{idx} («{description}») failed: The 'claims' key should contain a list"
        assert all(isinstance(item, dict) and "claim" in item and "valid" in item for item in claims_list), \
            f"Test case #{idx} («{description}») failed: Invalid evaluation format"
