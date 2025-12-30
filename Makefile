SHELL := /bin/bash

.PHONY: lab1, lab3

lab1:
	set -a; \
	. .env; \
	. $@/.env; \
	set +a; \
	python -m $@.main \
		--input "$$PATH_DATA/inputs/lena.png" \
		--runs 10 \
		--outdir "$$PATH_DATA/outputs"

# lab3:
# 	set -a; \
# 	. .env; \
# 	. $@/.env; \
# 	set +a; \
# 	python -m $@.main \
# 		--input "$$PATH_DATA/inputs/lena.png" \
# 		--runs 10 \
# 		--outdir "$$PATH_DATA/outputs"