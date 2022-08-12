# Data  <a class="anchor" id="data-overview"></a>
These data have been provided by the Georgia State University and the Learning Agency Lab for the Kaggle competition, “Feedback Prize – Predicting Effective Arguments”, They contain argumentative essays written by U.S. students in grades 6-12. Each essay has been annotated with discourse elements  commonly found in argumentative writing:
- Lead - an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis
- Position - an opinion or conclusion on the main question
- Claim - a claim that supports the position
- Counterclaim - a claim that refutes another claim or gives an opposing reason to the position
- Rebuttal - a claim that refutes a counterclaim
- Evidence - ideas or examples that support claims, counterclaims, or rebuttals.
- Concluding Statement - a concluding statement that restates the claims

The text classification task is to predict the quality of each discourse within each essay as one of:
- Ineffective
- Adequate
- Effective

The dataset is comprised of training data and example test data.

**Training Data**
The training dataset consists of a .csv file and a directory containing the full-text essays. Note, that parts of the full-text essays are not annotated and are not part of the training.csv file. Annotated discourse elements for each essay, including:
- discourse_id - ID code for discourse element
- essay_id - ID code for essay response. This ID code corresponds to the name of the full-text file in the train/ folder.
- discourse_text - Text of discourse element.
- discourse_type - Class label of discourse element.
- discourse_type_num - Enumerated class label of discourse element.
- discourse_effectiveness - Quality rating of discourse element, the target.

**Example Test Data**
The example test data consists of:
- test/ - A folder containing an example essay from the test set. The actual test set comprises about 3,000 essays in a format similar to the training set essays.
- test.csv - Annotations for the test set essays, containing all of the fields of train.csv except the target, discourse_effectiveness.
- sample_submission.csv - A sample submission file in the correct format.