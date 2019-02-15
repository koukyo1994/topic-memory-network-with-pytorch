TAG := "exp"
INPUT := tmn_data.txt
OUTPUT := tmn_out

process:
	make -C data TAG=${TAG} INPUT=${INPUT} OUTPUT=${OUTPUT}

inspect:
	python -m unittest discover test/
