from collections import namedtuple

GenreInfo = namedtuple("GenreInfo", ["desсription", "emoji"])
GENRES: dict[str, GenreInfo] = {
    "Технології": GenreInfo(
        "Новини про інновації, цифрові розробки, ґаджети та ІТ-сферу. Висвітлює тренди та вплив технологій на сучасне життя.",
        "💻"
    ),
    "Світ": GenreInfo(
        "Головні події з-за кордону, міжнародні відносини та геополітичні процеси. Охоплює глобальні зміни та їх наслідки.",
        "🌍"
    ),
    "Політика": GenreInfo(
        "Інформація про діяльність влади, вибори, закони та політичні рішення. Аналізує вплив політики на суспільство й державу.",
        "🏛️"
    ),
    "Суспільство": GenreInfo(
        "Тенденції, події та проблеми в соціальному житті людей. Висвітлює питання культури, моралі та взаємодії громадян.",
        "🧑‍🤝‍🧑"
    ),
    "Економіка": GenreInfo(
        "Новини про фінанси, ринки, бізнес та економічну політику. Включає аналітику впливу економіки на повсякденне життя.",
        "💰"
    ),
    "Війна": GenreInfo(
        "Оперативна інформація з фронту, військові події та стратегічна аналітика. Охоплює перебіг воєнних дій і безпекову ситуацію.",
        "⚔️"
    ),
    "Реклама": GenreInfo(
        "Актуальні пропозиції, промоакції та комерційні оголошення. Створено для ознайомлення з товарами, послугами чи подіями.",
        "📢"
    ),
    "Авто": GenreInfo(
        "Новини автопрому, огляди моделей, поради водіям. Включає тенденції в транспортній інфраструктурі та електромобільності.",
        "🚗"
    ),
    "Наука": GenreInfo(
        "Досягнення в наукових дослідженнях, відкриття та перспективи розвитку. Висвітлює інновації в природничих і технічних науках.",
        "🔬"
    ),
    "Спорт": GenreInfo(
        "Події зі світу спорту, результати змагань, коментарі та аналітика. Включає новини про команди, атлетів та спортивну індустрію.",
        "🏅"
    ),
    "Культура": GenreInfo(
        "Огляди подій у мистецтві, літературі, театрі та кіно. Відображає культурне життя та творчі здобутки суспільства.",
        "🎭"
    ),
    "Здоров'я": GenreInfo(
        "Поради щодо здорового способу життя, медицина, профілактика хвороб. Інформує про новини та досягнення у сфері охорони здоров’я.",
        "🩺"
    ),
    "Кримінал": GenreInfo(
        "Зведення про правопорушення, розслідування та кримінальну хроніку. Охоплює інциденти з громадською небезпекою.",
        "🚨"
    ),
    "Курйози": GenreInfo(
        "Незвичні, смішні або парадоксальні події з життя. Жанр для легкого контенту з елементами подиву чи гумору.",
        "😂"
    ),
    "Кулінарія": GenreInfo(
        "Рецепти, поради з приготування та гастрономічні тенденції. Пропонує ідеї для домашньої кухні та кулінарних експериментів.",
        "🍳"
    ),
    "Сад-город": GenreInfo(
        "Поради для дачників і садівників, сезонні роботи та догляд за рослинами. Включає практичні інструкції та лайфхаки.",
        "🪴"
    ),
}