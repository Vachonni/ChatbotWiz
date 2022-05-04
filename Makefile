run: test parse_raw find_topics clf_base clf_HP evaluate


test:
	pytest

parse_raw:
	python -m src.data.parse

find_topics:
	python -m src.modelling.topic_modelling

clf_base:
	python -m src.modelling.clf_base

clf_HP:
	python -m src.modelling.clf_HP_Search

evaluate:
	python -m src.modelling.evaluate





