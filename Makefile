DATA=---SpecifyADataset---

plot: examples/$(DATA).png

build: models/$(DATA).keras

examples/digits.png: src/digits-plot.py models/digits.keras
	python3 src/digits-plot.py

models/digits.keras: src/digits.py
	python3 src/digits.py

examples/letters.png: src/letters-plot.py models/letters.keras
	python3 src/letters-plot.py

models/letters.keras: src/letters.py
	python3 src/letters.py

examples/letters-cnn.png: src/letters-cnn-plot.py models/letters-cnn.keras
	python3 src/letters-cnn-plot.py

models/letters-cnn.keras: src/letters-cnn.py
	python3 src/letters-cnn.py

examples/characters.png: src/characters-plot.py models/characters.keras
	python3 src/characters-plot.py

models/characters.keras: src/characters.py
	python3 src/characters.py

clean-examples:
	rm -fr examples/*

clean-models:
	rm -fr models/* 
	
clean:
	rm -fr src/__pycache__
