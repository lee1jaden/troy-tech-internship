MODEL=n/a

letters-model:
	python3 src/letters.py

letters-plot:
	python3 src/letters-plot.py

letters-model-cnn:
	python3 src/letters-cnn.py

letters-plot-cnn:
	python3 src/letters-cnn-plot.py

digits-model:
	python3 src/digits.py

digits-plot:
	python3 src/digits-plot.py

clean-models:
	rm -fr models/* 
	
clean:
	rm -fr src/__pycache__
