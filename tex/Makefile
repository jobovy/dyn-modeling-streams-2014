FIGS= gd1_evol_lvlos.eps \
	gd1_evol_lpmll.eps \
	gd1_evol_lpmbb.eps \
	gd1_evol_lb.eps \
	gd1_evol_ld.eps \
	track_aa.eps \
	track_xz.eps \
	track_ld.eps

.SECONDARY:$(FIGS)

APJDIR= ApJ_v1

all: ms.pdf

apj:
	cp ms.tex $(APJDIR)

%.pdf: %.ps
	ps2pdf -dMaxSubsetPct=100 -dCompatibilityLevel=1.2 -dSubsetFonts=true -dEmbedAllFonts=true $<

%.ps: %.dvi
	dvips -t letter $< -o

ms.dvi: ms.tex $(FIGS)
	latex $<
	latex $<
	latex $<
	- bash -c " ( grep Rerun $*.log && latex $< ) || echo noRerun "
	- bash -c " ( grep Rerun $*.log && latex $< ) || echo noRerun "

gd1_evol_l%.eps: gd1_evol_l%.ps Makefile
	cp $< tmp.ps
	echo "1,\$$s/%%BoundingBox: 8 263 603 522/%%BoundingBox: 28 255 547 500/g" > edcmd
	echo "w" >> edcmd
	ed $< < edcmd
	cp $< $@
	cp tmp.ps $< 
	rm edcmd tmp.ps

track_ld.eps: track_ld.ps Makefile
	cp $< tmp.ps
	echo "1,\$$s/%%BoundingBox: 8 263 603 522/%%BoundingBox: 28 255 547 500/g" > edcmd
	echo "w" >> edcmd
	ed $< < edcmd
	cp $< $@
	cp tmp.ps $< 
	rm edcmd tmp.ps

track_%.eps: track_%.ps Makefile
	cp $< tmp.ps
	echo "1,\$$s/%%BoundingBox: 8 262 603 522/%%BoundingBox: 28 255 547 500/g" > edcmd
	echo "w" >> edcmd
	ed $< < edcmd
	cp $< $@
	cp tmp.ps $< 
	rm edcmd tmp.ps

%.dvi: %.tex

.PHONY: clean spotless apj

clean:
	$(RM) *.log *.aux *.dvi

spotless: clean
	$(RM) ms.ps
	$(RM) ms.pdf
