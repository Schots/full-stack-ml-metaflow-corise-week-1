from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact

# TODO move your labeling function from earlier in the notebook here
def labeling_function(row):
    """
    Labels the provided row based on the "rating" column value.
    
    If the rating is 4 or 5, the function returns 1, indicating a positive review.
    Otherwise, it returns 0, indicating a non-positive review.
    
    Parameters:
    - row (pd.Series): A row from a DataFrame, expected to have a "rating" key.
    
    Returns:
    - int: 1 if the rating is 4 or 5, otherwise 0.
    """
    return 1 if row["rating"] in [4, 5] else 0


class BaselineNLPFlow(FlowSpec):
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

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import roc_auc_score

        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        dummy_model = DummyClassifier()
        dummy_model.fit(self.traindf.drop("label",axis=1),self.traindf["label"])
        self.preds = dummy_model.predict(self.valdf["label"])
        self.base_acc = accuracy_score(self.valdf["label"],self.preds)
        self.base_rocauc = roc_auc_score(self.valdf["label"],self.preds)

        self.next(self.end)

    @card(
        type="corise"
    )  # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        msg = "Baseline Accuracy: {}\nBaseline AUC: {}"
        print(msg.format(round(self.base_acc, 3), round(self.base_rocauc, 3)))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        current.card.append(Markdown("## Examples of False Positives"))
        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0.
        # TODO: display the false_positives dataframe using metaflow.cards
        # Documentation: https://docs.metaflow.org/api/cards#table
        fp_mask = (self.preds == 1) & (self.valdf["label"] == 0)
        false_positives = self.valdf[fp_mask]
        if not false_positives.empty: 
            current.card.append(Table(false_positives.values.tolist()))
            
        current.card.append(Markdown("## Examples of False Negatives"))
        # TODO: compute the false positive predictions where the baseline is 0 and the valdf label is 1.
        # TODO: display the false_negatives dataframe using metaflow.cards
        fn_mask = (self.preds == 0) & (self.valdf["label"] == 1)
        false_negatives = self.valdf[fn_mask]
        if not false_negatives.empty: 
            current.card.append(Table(false_negatives.values.tolist()))
        
if __name__ == "__main__":
    BaselineNLPFlow()
