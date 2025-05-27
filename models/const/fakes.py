from collections import namedtuple

FakeInfo = namedtuple("FakeInfo", ["name", "description", "emoji"])
FAKES_ML: dict[bool, FakeInfo] = {
    False: FakeInfo(
        name="Достовірна інформація",
        description="Інформація не містить ознак неправдивості чи маніпуляцій і відповідає критеріям достовірності.",
        emoji="✅"
    ),
    True: FakeInfo(
        name="Фейкова інформація",
        description="Інформація має ознаки неправдивості, спотворення фактів або маніпулятивного викладення.",
        emoji="❌"
    ),
}