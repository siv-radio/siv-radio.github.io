# Theme Modification

The basic theme  
https://github.com/adityatelange/hugo-PaperMod/tree/6e10faefc871d8ff3c96136f6b287ac7984bf40d  
hugo-PaperMod, 2025.09.28, 6e10fae (master)

## RSS Feed Modification
`./layout/_default/rss.xml`  
Type: file modified.  
Reason: exclude "License" and "About" pages from the RSS feed.  
Date: 2025.10.03

The line  
```html
{{- if and (ne .Layout `search`) (ne .Layout `archives`) }}
```
changed to  
```html
{{- if and (ne .Layout `search`) (ne .Layout `archives`) (ne .Layout `regular_page`) }}
```
in order to exclude non-post pages from the RSS feed.


## External Text Inclusion
`./layouts/shortcodes/include.html`  
Type: file added.  
Reason: add the ability to include another textual file content into a page markdown file.  
Date: 2025.10.01

References  
"Hugo: include/embed another file with a shortcode", 2022.03.29.  
https://roneo.org/en/hugo-include-another-file-with-a-shortcode/


## Math Rendering
`./layouts/_default/_markup/render-passthrough.html`  
Type: file added.

`./layouts/_default/baseof.html`  
Type: file modified.  
The following code snippet has been added  
```html
<head>
    {{- partial "head.html" . }}
    {{ $noop := .WordCount }}
    {{ if .Page.Store.Get "hasMath" }}
        <link href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css" rel="stylesheet">
    {{ end }}
</head>
```

`./hugo.yaml`  
Type: file modified.  
The following lines has been added  
```yaml
markup:
  goldmark:
    extensions:
      passthrough:
        delimiters:
          block:
          - - "$$"
            - "$$"
          inline:
          - - "$`"
            - "`$"
        enable: true
```

Reason: render mathematical expressions written in the LaTeX at the building time.  
Date: 2025.10.01

References  
1. https://gohugo.io/functions/transform/tomath/
2. https://gohugo.io/render-hooks/passthrough/
3. https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions
4. https://discourse.gohugo.io/t/how-to-use-the-transform-tomath-function/52152
5. The default Hugo theme can be found in  
https://github.com/gohugoio/hugo/blob/master/create/skeletons/theme/

## "License" and "About" Pages
`./layouts/_default/separate_page.html`  
Type: file added.  
Reason: a template for "License" and "About" pages.  
Date: 2025.10.01

Based on  
1. https://github.com/adityatelange/hugo-PaperMod/blob/master/layouts/_default/terms.html
2. https://github.com/adityatelange/hugo-PaperMod/blob/master/layouts/_default/single.html

## "Next" and "Prev" Post Interchange
`./i18n/en.yaml`  
Type: file modified.  
Reason: previous and next page labels are interchanged under post pages.  
Date: 2025.10.05

Usual page arrangement  
```text
<< < 1 2 3 4 5 ... N > >>
next 1 2 3 4 5 ... N prev
```

The following lines has been modified  
```yaml
- id: prev_page
  translation: "Next"

- id: next_page
  translation: "Prev"
```
