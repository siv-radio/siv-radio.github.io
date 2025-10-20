---
author: "Author One"
title: "Test Post"
date: "2025-09-25"
description: "test description"
summary: "test summary"
tags: ["tag 0", "tag 1", "tag 2", "tag 3"]
math: true
ShowBreadCrumbs: false
draft: true
---

[hidden words]: #

This is an inline $`a^*=x-b^*`$ equation.

These are block equations:

$$a^*=x-b^*$$

$$a^*=x-b^*$$

![](somepic.png)  
Picture 1. \<picture-name\>.

[collapsible](https://github.com/adityatelange/hugo-PaperMod/discussions/658 "\"popdown rendering in markdown for posts (like the table of contents)\", #658, by cs-mshah, 2021.12.03") section  
{{< collapse summary="**`somefile.py`**" >}}
```python
{{< include "somefile.py" >}}
```
{{< /collapse >}}

alternative formating (HTML):  
<b><code>somefile.py</code></b>

include:  
```python
{{% include "somefile.py" %}}
```
