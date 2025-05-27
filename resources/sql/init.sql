CREATE TABLE `article` (
  `id`   INTEGER PRIMARY KEY AUTOINCREMENT,
  `headline` TEXT,
  `content`  TEXT,
  `hash`     TEXT UNIQUE NOT NULL
);

CREATE TABLE `admin` (
  `id`     INTEGER PRIMARY KEY AUTOINCREMENT,
  `handle` TEXT UNIQUE NOT NULL
);

CREATE TABLE `article_url` (
  `article_id` INTEGER PRIMARY KEY,
  `url`        TEXT NOT NULL,
  FOREIGN KEY(`article_id`) REFERENCES `article`(`id`)
);

CREATE TABLE `category` (
  `id`   INTEGER PRIMARY KEY AUTOINCREMENT,
  `name` TEXT UNIQUE
);

CREATE TABLE `genre` (
  `id`          INTEGER PRIMARY KEY AUTOINCREMENT,
  `name`        TEXT UNIQUE
);

CREATE TABLE `propaganda_technique` (
  `id`          INTEGER PRIMARY KEY AUTOINCREMENT,
  `name`        TEXT UNIQUE
);

CREATE TABLE `analysis_result` (
  `id`          INTEGER PRIMARY KEY AUTOINCREMENT,
  `category_id` INTEGER,
  `article_id`  INTEGER NOT NULL,
  `analysed_at` TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  FOREIGN KEY(`category_id`) REFERENCES `category`(`id`),
  FOREIGN KEY(`article_id`)  REFERENCES `article`(`id`)
);

CREATE TABLE `fake_result` (
  `id`          INTEGER PRIMARY KEY AUTOINCREMENT,
  `analysis_id`    INTEGER,
  `percentage_score` REAL NOT NULL,
  FOREIGN KEY(`analysis_id`) REFERENCES `analysis_result`(`id`)
);

CREATE TABLE `db_fake_result` (
  `id`          INTEGER PRIMARY KEY AUTOINCREMENT,
  `analysis_id`    INTEGER,
  `message`        TEXT NOT NULL,
  `verdict`        BOOLEAN NOT NULL,
  FOREIGN KEY(`analysis_id`) REFERENCES `analysis_result`(`id`)
);

CREATE TABLE `clickbait_result` (
  `id`          INTEGER PRIMARY KEY AUTOINCREMENT,
  `analysis_id`    INTEGER,
  `percentage_score` REAL NOT NULL,
  FOREIGN KEY(`analysis_id`) REFERENCES `analysis_result`(`id`)
);

CREATE TABLE `analysis_result_genre` (
  `id`          INTEGER PRIMARY KEY AUTOINCREMENT,
  `analysis_id` INTEGER NOT NULL,
  `genre_id`    INTEGER NOT NULL,
  `score`       REAL NOT NULL,
  FOREIGN KEY(`analysis_id`) REFERENCES `analysis_result`(`id`),
  FOREIGN KEY(`genre_id`)    REFERENCES `genre`(`id`)
);

CREATE TABLE `analysis_result_propaganda` (
  `id`           INTEGER PRIMARY KEY AUTOINCREMENT,
  `technique_id` INTEGER,
  `analysis_id`  INTEGER,
  `score`        REAL NOT NULL,
  FOREIGN KEY(`analysis_id`)   REFERENCES `analysis_result`(`id`),
  FOREIGN KEY(`technique_id`)  REFERENCES `propaganda_technique`(`id`)
);

INSERT INTO 
  `genre` (id, name)
VALUES
  (1, "Технології"),
  (2, "Світ"),
  (3, "Політика"),
  (4, "Суспільство"),
  (5, "Економіка"),
  (6, "Війна"),
  (7, "Реклама"),
  (8, "Авто"),
  (9, "Наука"),
  (10, "Спорт"),
  (11, "Культура"),
  (12, "Здоров'я"),
  (13, "Кримінал"),
  (14, "Курйози"),
  (15, "Кулінарія"),
  (16, "Сад-город");

INSERT INTO
  `propaganda_technique` (id, name)
VALUES
  (1, "Нагнітання страху"),
	(2, "Навіювання сумнівів"),
	(3, "Розмахування прапором"),
	(4, "Навантажена мова"),
	(5, "Демонізація ворога"),
	(6, "Наклеп"),
	(7, "Лайливе ім’я"),
	(8, "Слова чеснот"),
	(9, "Теорія змови"),
	(10, "Надмірне спрощення"),
	(11, "Не виявлено");

INSERT INTO admin (id, handle) VALUES (1, 'maksymko');

INSERT INTO
  category (id, name)
VALUES
  (1, 'propaganda'),
  (2, 'fake_ml'),
  (3, 'fake_db'),
  (4, 'clickbait'),
  (5, 'genres');
