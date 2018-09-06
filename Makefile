all: taxi

.PHONY: taxi
taxi: venv
	. venv/bin/activate; cd taxi; python3 test.py

.PHONY: pacman_random
pacman_random: venv
	. venv/bin/activate; cd pacman_random; python3 test.py

venv: requirements.txt
	virtualenv -p python3 venv
	. venv/bin/activate; pip3 install -r requirements.txt

clean:
	rm -rf venv

