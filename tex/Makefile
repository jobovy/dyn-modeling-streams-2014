FIGS= gd1_evol_lvlos.eps \
	gd1_evol_lpmll.eps \
	gd1_evol_lpmbb.eps \
	gd1_evol_lb.eps \
	gd1_evol_ld.eps \
	track_aa.eps \
	track_xz.eps \
	track_ld.eps \
	gd1_evol_xz.eps \
	gd1_evol_aadohist.eps \
	gd1_evol_2Gya_xz.eps \
	gd1_evol_2Gya_xz_sim.eps \
	gd1_pdf_xz_0.eps \
	gd1_pdf_xz_1.eps \
	gd1_pdf_xz_2.eps \
	aAI_jr.eps \
	aAI_araz.eps

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

aAI_%.eps: aAI_%.ps Makefile
	cp $< tmp.ps
	echo "1,\$$s/%%BoundingBox: 89 215 522 576/%%BoundingBox: 97 218 479 548/g" > edcmd
	echo "w" >> edcmd
	ed $< < edcmd
	cp $< $@
	cp tmp.ps $< 
	rm edcmd tmp.ps

gd1_pdf_xz_%.eps: gd1_pdf_xz_%.ps Makefile
	cp $< tmp.ps
	echo "1,\$$s/%%BoundingBox: 125 215 486 576/%%BoundingBox: 124 218 457 548/g" > edcmd
	echo "w" >> edcmd
	ed $< < edcmd
	cp $< $@
	cp tmp.ps $< 
	rm edcmd tmp.ps

gd1_evol_2Gya_%.eps: gd1_evol_2Gya_%.ps Makefile
	cp $< tmp.ps
	echo "1,\$$s/%%BoundingBox: 125 215 486 576/%%BoundingBox: 123 216 454 549/g" > edcmd
	echo "w" >> edcmd
	ed $< < edcmd
	cp $< $@
	cp tmp.ps $< 
	rm edcmd tmp.ps

gd1_evol_xz.eps: gd1_evol_xz.ps Makefile
	cp $< tmp.ps
	echo "1,\$$s/%%BoundingBox: 125 215 486 576/%%BoundingBox: 130 217 458 545/g" > edcmd
	echo "w" >> edcmd
	ed $< < edcmd
	cp $< $@
	cp tmp.ps $< 
	rm edcmd tmp.ps

gd1_evol_aadohist.eps: gd1_evol_aadohist.ps Makefile
	cp $< tmp.ps
	echo "1,\$$s/%%BoundingBox: 125 215 486 576/%%BoundingBox: 145 211 460 552/g" > edcmd
	echo "w" >> edcmd
	ed $< < edcmd
	cp $< $@
	cp tmp.ps $< 
	rm edcmd tmp.ps

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
