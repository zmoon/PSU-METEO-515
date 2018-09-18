#!/bin/bash
# --------------------------------------------------------------------------------
#
# Convert IPython notebook to html and then html to pdf
# 
# description of pdf options here: http://wkhtmltopdf.org/usage/wkhtmltopdf.txt
# 
#
# Zach Moon
# --------------------------------------------------------------------------------

zoomfactor=3.0  # to make it not super wide and small

outdir=./nbconverted

for fname in hw2p1.ipynb hw2p2.ipynb; do

    fbasename=${fname%.ipynb}

    #> first use nbconvert to convert to html
    jupyter nbconvert --to html $fname 
    jupyter nbconvert --to pdf $fname  # standard article PDF using latex

    #> use wkhtmltopdf to convert the html to pdf
    /usr/local/bin/wkhtmltopdf --page-size Letter --zoom $zoomfactor \
        ${fbasename}.html ${fbasename}.nbstyle.pdf

    #> move to out dir
    mv ${fbasename}.html $outdir
    mv ${fbasename}.pdf $outdir
    mv ${fbasename}.nbstyle.pdf $outdir

done

