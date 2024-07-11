# Dissertation

Code to support my MSc dissertation, looking at robustness of AI explanations.

Code is all in Python. Dependencies are managed using Conda. You can import dependencies using the environment.yml file: `conda env create --file environment.yml`

Folders are organised as follows:

- `notebooks`: the classification and regression models I am working with to generate explanations.
- `package`: the explanation robustness toolbox package published to PyPI.

Final PDF doc here: https://github.com/pidg3/hw-dissertation/blob/master/210823%20MP%20Thesis%20Final.pdf

Abstract copied below for ease of reference:

_One challenge with machine learning is the models used are often ‘black boxes’, which cannot be meaningfully interpreted by humans. A number of ‘post-hoc’ explainable AI techniques can help provide a better understanding of why certain predictions were made, without requiring access to model internals._

_This project investigated metrics that can be applied to these post-hoc explanations, in order to build confidence in their robustness. In the context of explanations, robustness means ensuring small input perturbations do not cause major changes in the explanation, as well as assessing its degree of faithfulness to the underlying model._

_These two robustness characteristics were assessed by a novel implementation of expla- nation ‘sensitivity’ and ‘infidelity’ metrics. This implementation was published as an open-source Python package, with the results successfully tested and validated._

_The metrics were used to provide useful information on explanations of a series of ‘real- world’ models, whose underlying robustness is uncertain. The metrics showed a clear ability to detect overfitted models, as well as some evidence of detecting adversarial models (where the explanations are deliberately manipulated)._

_My overall aim in this project was to add to the body of work examining methods of building trust in explanations of machine learning models - and, by implication, help understand when such explanations should not be trusted. I feel the project has met this aim, and the open-source software published as a result should prove useful. Further work could provide a better understanding of the subtleties in applying the metrics in practical situations._
