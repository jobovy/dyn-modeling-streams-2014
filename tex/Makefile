#FIGS= 	$(subst ps,eps,$(shell ls -d ../dffigs_g_230/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc230_an_margvrvt_staeckel_singles_model_data_g_*.ps)) \

#.SECONDARY:$(FIGS)

APJDIR= ApJ_v1

all: ms.pdf

apj:
	cp ms.tex $(APJDIR)

%.pdf: %.ps
	ps2pdf -dMaxSubsetPct=100 -dCompatibilityLevel=1.2 -dSubsetFonts=true -dEmbedAllFonts=true $<

%.ps: %.dvi
	dvips -t letter $< -o

ms.dvi: ms.tex #$(PAPERFIGS)
	latex $<
	latex $<
	latex $<
	- bash -c " ( grep Rerun $*.log && latex $< ) || echo noRerun "
	- bash -c " ( grep Rerun $*.log && latex $< ) || echo noRerun "

#hzszq_%.eps: hzszq_%.ps Makefile
#	cp $< tmp.ps
#	echo "1,\$$s/%%BoundingBox: 125 215 486 576/%%BoundingBox: 134 252 466 527/g" > edcmd
#	echo "w" >> edcmd
#	ed $< < edcmd
#	cp $< $@
#	cp tmp.ps $< 
#	rm edcmd tmp.ps

%.dvi: %.tex
	latex $<
	latex $<
	latex $<
	- bash -c " ( grep Rerun $*.log && latex $< ) || echo noRerun "
	- bash -c " ( grep Rerun $*.log && latex $< ) || echo noRerun "

.PHONY: clean spotless apj

clean:
	$(RM) *.log *.aux *.dvi

spotless: clean
	$(RM) ms.ps
	$(RM) ms.pdf
