from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
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
    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter("split-sz", default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        # Step-level dependencies are loaded within a Step, instead of loading them
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({"label": labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.lr)

    @step
    def lr(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.pipeline import make_pipeline

        # Tokenization
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        
        #Estimator
        estimator = LogisticRegression(max_iter=1_000)

        #Model
        lr_model = make_pipeline(vectorizer,estimator)
        lr_model.fit(self.traindf['review'], self.traindf['label'])
        self.preds = lr_model.predict(self.valdf['review'])
        self.probas = lr_model.predict_proba(self.valdf['review'])[:,1]
        
        # Metrics
        self.lr_acc = accuracy_score(self.valdf['label'], self.preds)
        self.lr_rocauc = roc_auc_score(self.valdf['label'], self.probas)
        
        self.next(self.end)

    @card(type="corise")
    @step
    def end(self):
        print(f"Baseline Accuracy: {self.lr_acc:0.2f} - Baseline ROCAUC: {self.lr_rocauc:0.2f}")
        # Display model results using cards
        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.lr_acc))
        current.card.append(Markdown("## ROC AUC"))
        current.card.append(Artifact(self.lr_rocauc))

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
