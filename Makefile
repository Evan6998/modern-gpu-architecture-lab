PYTHON ?= python
A ?= a1
OP ?= transpose
IMPL ?= student
SUITE ?= smoke

.PHONY: help doctor cpu check dist-smoke grade bench
help:
	@echo "make doctor | cpu | check | dist-smoke | grade A=a1 | bench A=a1 OP=transpose IMPL=student SUITE=full"
doctor:
	$(PYTHON) -m mgpu.doctor --output results/environment.json
cpu:
	$(PYTHON) -m pytest -q -m 'not gpu and not distributed'
check:
	$(PYTHON) -m compileall -q mgpu assignments scripts

grade:
	$(PYTHON) -m mgpu.grade $(A)
bench:
	$(PYTHON) -m mgpu.bench $(A) --op $(OP) --impl $(IMPL) --suite $(SUITE)
dist-smoke:
	$(PYTHON) -m torch.distributed.run --standalone --nproc_per_node=2 -m assignments.a6_moe.run --device cpu --impl reference --smoke --output results/a6-cpu-smoke
