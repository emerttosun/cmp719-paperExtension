Overleaf upload steps:

1. Open the official ICLR 2025 template as a NEW Overleaf project:
   https://www.overleaf.com/latex/templates/template-for-iclr-2025-conference-submission/gqzkdyycxtvt

2. From this folder, upload these 3 files into the project root
   (overwrite when prompted):
     - main.tex
     - references.bib
     - gate_means_combined.png

3. Keep the template's existing files untouched:
     - iclr2025_conference.sty
     - iclr2025_conference.bst
     - math_commands.tex
     - fancyhdr.sty (if present)

4. Set Compiler to pdfLaTeX (Menu -> Compiler -> pdfLaTeX).

5. Click Recompile. If references don't render, click Recompile a second
   time so BibTeX picks up references.bib.

Expected output: 4-5 page PDF. If it overflows, the easiest cut is the
"Identity initialisation" paragraph in Section 3.
