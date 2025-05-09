MODEL=n/a

char-model:
	python3 src/characters.py

char-plot:
	python3 src/characters-plot.py

clean:
	rm -fr models/*
