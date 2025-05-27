from collections import namedtuple


PropagandaTechniqueInfo = namedtuple("TechniqueInfo", ["name", "description", "emoji"])
PROPAGANDA_TECHNIQUES: dict[str, PropagandaTechniqueInfo] = {
    'Fearmongering': PropagandaTechniqueInfo(
        name="Нагнітання страху",
        description="Нагнітання страху для впливу на думки чи поведінку людей (наприклад, залякування катастрофами чи загрозами).",
        emoji="😨"
    ),
    'Doubt Casting': PropagandaTechniqueInfo(
        name="Навіювання сумнівів",
        description="Спроба викликати сумніви щодо фактів або джерел, навіть без доказів (наприклад, \"А ви впевнені, що це правда?\").",
        emoji="❓"
    ),
    'Flag Waving': PropagandaTechniqueInfo(
        name="Розмахування прапором",
        description="Посилання на патріотизм або національні символи, щоб підтримати певну ідею або дію.",
        emoji="🇺🇸"
    ),
    'Loaded Language': PropagandaTechniqueInfo(
        name="Навантажена мова",
        description="Слова з емоційним забарвленням, які впливають на ставлення слухача (наприклад, \"зрадник\" замість \"опонент\").",
        emoji="🗣️"
    ),
    'Demonizing the Enemy': PropagandaTechniqueInfo(
        name="Демонізація ворога",
        description="Зображення противника як небезпечного, нелюдського або злого, щоб викликати ненависть чи страх.",
        emoji="👿"
    ),
    'Smear': PropagandaTechniqueInfo(
        name="Наклеп",
        description="Поширення пліток, наклепів чи особистих нападів, щоб знизити довіру до когось.",
        emoji="🖌️"
    ),
    'Name Calling': PropagandaTechniqueInfo(
        name="Лайливе ім’я",
        description="Навішування ярликів або образливих назв, щоб дискредитувати людину або ідею.",
        emoji="🤬"
    ),
    'Virtue Words': PropagandaTechniqueInfo(
        name="Слова чеснот",
        description='Використання слів з позитивним значенням (як-от "свобода", "справедливість"), щоб надати ідеї привабливості.',
        emoji="🌟"
    ),
    'Conspiracy Theory': PropagandaTechniqueInfo(
        name="Теорія змови",
        description='Висування припущень про таємні змови без достатніх доказів (наприклад, "усе це підлаштовано").',
        emoji="🕵️"
    ),
    'Oversimplification': PropagandaTechniqueInfo(
        name="Надмірне спрощення",
        description="Подання складної теми у вигляді простого поділу на добре і зле, ігноруючи нюанси.",
        emoji="⚫⚪"
    )
}

FALLBACK_TECHNIQUE = PropagandaTechniqueInfo(
    name="Не виявлено",
    description="Жодних ознак пропаганди не виявлено.",
    emoji="✅"
)
