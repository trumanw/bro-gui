.PHONY: clean install reinstall
clean:
	pip uninstall -y brogui
	find . -type d -name __pycache__ -exec rm -r {} \+
	rm -rf dist build
install:
	python setup.py install
reinstall: clean
	python setup.py install
run: clean
	streamlit run brogui/gui.py