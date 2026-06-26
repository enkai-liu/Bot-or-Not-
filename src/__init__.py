"""Bot-or-Not: social-media bot detection pipeline.

Modules
-------
parser   : load raw dataset JSON + bot label files into User objects.
features : turn a user's posts and metadata into a numeric feature vector.
dataset  : build the full user-level feature table from every dataset.
model    : train, cross-validate and evaluate the classifier.
export   : write metrics and predictions to JSON for the web showcase.
"""

__all__ = ["parser", "features", "dataset", "model", "export"]
