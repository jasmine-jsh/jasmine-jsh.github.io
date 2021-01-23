---
layout: post
title:  "How to Put Math Equations"
date:   2020-09-01 09:16:11 +0900
image:  02.jpg
tags:   Jekyll MathJax
---

I struggled with putting math equations on the post. Finally, found the way! 


Go to [MathJax][MathJax]. To include MathJax, Copy and paste the script as below

```
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```


to `default.html` file in `_layouts` dicrectory.


Now you can write math equations and physics formulas by using [LaTex][LaTex] syntax in <strong>$$ $$ $$ $$</strong> even though how complex they are.

$$f(x)=x^2$$

$$E=mc^2$$


[MathJax]: https://www.mathjax.org/#gettingstarted
[LaTex]: https://en.wikibooks.org/wiki/LaTeX/Mathematics
