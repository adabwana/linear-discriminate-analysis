(ns index)

;# Assignment 2: Linear Discriminant Analysis (LDA)
;
;Instructions:
;
;* Generate a suitable dataset that contains three classes.
;* Scale, normalize, and/or encode your data appropriately.
;* Implement and apply Univariate LDA.
;* Make sure to use 5-fold cross-validation.
;* Modify the "README.md" file to include the following sections<br>
;;&nbsp;&nbsp;&nbsp;&nbsp;* **Summary**: A one-paragraph summary of the algorithm that was implemented including any pertinent or useful information. If mathematics are appropriate, include those as well.<br>
;;&nbsp;&nbsp;&nbsp;&nbsp;* **Reflection**: One paragraph describing useful takeaways from the week, things that surprised you, or things that caused you inordinate difficulty.
;* Make sure that your README file is formatted properly and is visually appealing. It should be free of grammatical errors, punctuation errors, capitalization issues, etc.
;
;What I did:
;
;* Generated dataset based on three groupings with overlaps.
;* Visualized data with histograms and scatter plot.
;* Implemented a univariate LDA from scratch.
;* Implemented a multivariate LDA with Clojure's built-in functionality (still tuning my Clojure workflow). Resampled with bootstrapping repeated `b = 30` times.<br>
;;&nbsp;&nbsp;&nbsp;&nbsp; * Tested a scale and normalized model pipeline versus raw data. Both models performed equivalently.
;* R interop to visualize the multivariate model with ggplots2.
;* Rendered document with Clay, pushed to GitHub, and had GitHub Actions deploy the static rendering to a webpage.