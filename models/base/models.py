from models.base.model_type import ModelType
from models.base.model import Model
from models import (
    FakesModel,
    GenresModel,
    ClickbaitModel,
    PropagandaModel
)

models: dict[ModelType, Model] = {
    ModelType.PROPAGANDA: PropagandaModel(),
    ModelType.FAKE_NEWS: FakesModel(),
    ModelType.GENRES: GenresModel(),
    ModelType.CLICKBAIT: ClickbaitModel(),
}
