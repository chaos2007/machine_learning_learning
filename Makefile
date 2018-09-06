all: taxi

.PHONY: taxi
taxi: venv
	. venv/bin/activate; cd taxi; python3 test.py

.PHONY: pacman_random
pacman_random: venv
	. venv/bin/activate; cd pacman_random; python3 test.py

.PHONY: pacman_brute_force_tensors
pacman_brute_force_tensors: venv
	. venv/bin/activate; cd pacman_brute_force_tensors; python3 test.py

.PHONY: pacman_naive_q_learning
pacman_naive_q_learning: venv
	. venv/bin/activate; cd pacman_naive_q_learning; python3 test.py

venv: requirements.txt
	virtualenv -p python3 venv
	. venv/bin/activate; pip3 install -r requirements.txt

clean:
	rm -rf venv

