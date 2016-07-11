PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests

# Compilation...

CYTHONSRC= $(wildcard polylearn/*.pyx)
CSRC= $(CYTHONSRC:.pyx=.cpp)

inplace:
	$(PYTHON) setup.py build_ext -i

all: cython inplace

cython: $(CSRC)

clean:
	rm -f polylearn/*.c polylearn/*.cpp polylearn/*.html
	rm -f `find polylearn -name "*.pyc"`
	rm -f `find polylearn -name "*.so"`

%.cpp: %.pyx
	$(CYTHON) --cplus $<

# Tests...
#
test-code: inplace
	$(NOSETESTS) -s polylearn

test-coverage:
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=polylearn polylearn

