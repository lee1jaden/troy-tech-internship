MODEL=n/a

char-model:
	python3 src/characters.py

char-plot:
	python3 src/characters-plot.py

char-model-cnn:
	python3 src/characters-cnn.py

char-plot-cnn:
	python3 src/characters-cnn-plot.py

digits-model:
	python3 src/digits.py

digits-plot:
	python3 src/digits-plot.py

clean-models:
	rm -fr models/* 
	
clean:
	rm -fr src/__pycache__
