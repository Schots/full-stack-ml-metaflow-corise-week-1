from metaflow import (
    FlowSpec,
    step,
    Parameter,
    IncludeFile,
    card,
    current
)
from metaflow.cards import Table, Markdown, Artifact

def labeling_function(row):
    """
    Label a provided row based on the "rating" column value.
    
    Parameters:
    - row (pd.Series): A row from a DataFrame with a "rating" key.
    
    Returns:
    - int: 1 if rating is 4 or 5 (indicating a positive review), otherwise 0.
    """
    return 1 if row["rating"] in [4, 5] else 0

class GoodFirstModelNLPFlow(FlowSpec):
    split_size = Parameter("split-sz", default=0.2)
    seed = Parameter("random_seed",default=42)
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # Load and preprocess the dataset
        df = pd.read_csv(io.StringIO(self.data))
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        self.df = pd.DataFrame({"label": labels, **_has_review_df})

        # Split data for training and validation
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, 
                                                     test_size=self.split_size,
                                                     random_state=self.seed)
        self.next(self.first_nlp_model)

    @step
    def first_nlp_model(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import MaxAbsScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import accuracy_score, roc_auc_score

        ## Model Pipeline Components
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        scaler = MaxAbsScaler()
        estimator = LogisticRegression(max_iter=1_000)
        first_good_model = make_pipeline(vectorizer,scaler,estimator)

        # Train and score the first good model
        first_good_model.fit(self.traindf.drop("label",axis=1),self.traindf["label"])
        self.preds = first_good_model.predict(self.valdf["label"])
        self.base_acc = accuracy_score(self.valdf["label"],self.preds)
        self.base_rocauc = roc_auc_score(self.valdf["label"],self.preds)

        self.next(self.end)

    @card(type="corise")
    @step
    def end(self):
        print(f"Baseline Accuracy: {self.base_acc:0.2f} - Baseline ROCAUC: {self.base_rocauc:0.2f}")
        # Display model results using cards
        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))
        current.card.append(Markdown("## ROC AUC"))
        current.card.append(Artifact(self.base_rocauc))

        # Show false positives
        current.card.append(Markdown("## Examples of False Positives"))
        fp_mask = (self.preds == 1) & (self.valdf["label"] == 0)
        false_positives = self.valdf[fp_mask]
        if not false_positives.empty: 
            current.card.append(Table(false_positives.values.tolist()))
            
        # Show false negatives
        current.card.append(Markdown("## Examples of False Negatives"))
        fn_mask = (self.preds == 0) & (self.valdf["label"] == 1)
        false_negatives = self.valdf[fn_mask]
        if not false_negatives.empty: 
            current.card.append(Table(false_negatives.values.tolist()))

if __name__ == "__main__":
    GoodFirstModelNLPFlow()
