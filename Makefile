# Inside the venv `python` exists; a bare macOS shell often only has python3.
PYTHON ?= $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)
A ?= a1
# Empty OP lets mgpu.bench pick the assignment's first operation.
OP ?=
IMPL ?= student
SUITE ?= smoke

.PHONY: help doctor cpu check dist-smoke grade bench
help:
	@echo "make doctor | cpu | check | dist-smoke | grade A=a1 | bench A=a1 [OP=transpose] IMPL=student SUITE=full"
doctor:
	$(PYTHON) -m mgpu.doctor --output results/environment.json
cpu:
	$(PYTHON) -m pytest -q -m 'not gpu and not distributed'
check:
	$(PYTHON) -m compileall -q mgpu assignments scripts


grade:
	$(PYTHON) -m mgpu.grade $(A)
bench:
	$(PYTHON) -m mgpu.bench $(A) $(if $(OP),--op $(OP)) --impl $(IMPL) --suite $(SUITE)
# Loopback rendezvous: torchrun otherwise advertises the host FQDN, and a name
# that does not resolve (common on laptops) hangs the launch for minutes.
dist-smoke:
	$(PYTHON) -m torch.distributed.run --standalone --local_addr=127.0.0.1 --nproc_per_node=2 -m assignments.a6_moe.run --device cpu --impl reference --smoke --output results/a6-cpu-smoke
