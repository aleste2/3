export GOPATH=/home/vrf/Code/
# Use the default go compiler
GO_BUILDFLAGS=-compiler gc
# Or uncomment the line below to use the gccgo compiler, which may
# or may not be faster than gc and which may or may not compile...
# GO_BUILDFLAGS=-compiler gccgo -gccgoflags '-static-libgcc -O4 -Ofast -march=native'

# Multiple gpus
# export CUDA_CC=30 32 35 37 50 52 53 60 61 62 70 72 75 80 86
export CUDA_CC=86
CGO_CFLAGS_ALLOW='(-fno-schedule-insns|-malign-double|-ffast-math)'


.PHONY: all cudakernels clean realclean checktests runtests hooks


all: cudakernels hooks
	go install -v $(GO_BUILDFLAGS) github.com/mumax/3/...

cudakernels:
	cd cuda && $(MAKE)

doc:
	cd doc && $(MAKE)

test: all
	go test -vet=off -i github.com/mumax/3/...
	go test -vet=off $(PKGS)  github.com/mumax/3/...
	cd test && ./run.bash

hooks: .git/hooks/post-commit .git/hooks/pre-commit

.git/hooks/post-commit: post-commit
	ln -sf $(CURDIR)/$< $@

.git/hooks/pre-commit: pre-commit
	ln -sf $(CURDIR)/$< $@

clean:
	rm -frv $(GOPATH)/pkg/*/github.com/mumax/3/*
	rm -frv $(GOPATH)/bin/mumax3*
	cd cuda && $(MAKE) clean

realclean: clean
	cd cuda && ${MAKE} realclean
