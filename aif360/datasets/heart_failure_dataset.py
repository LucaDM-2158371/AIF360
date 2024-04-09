import os
import pandas as pd

from aif360.datasets import StandardDataset

default_mappings = {
    'protected_attribute_maps': [{1.0: 'Old', 0.0: 'Young'}],
}

def custom_preprocessing(df):
    # Add a new binary 'age_over_55' column
    df['age_over_55'] = df['age'].apply(lambda x: 1 if x > 55 else 0)
    return df

class HeartFailureDataset(StandardDataset):
    def __init__(self, label_name='DEATH_EVENT', favorable_classes=[1],
                 protected_attribute_names=['age'],
                 privileged_classes=[lambda x: x > 55],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings):
        
        # get filepath from dropoutchance dataset
        filepath = "aif360/data/raw/heart_failure/heart_failure_clinical_records_dataset.csv"
        
        column_names = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time','DEATH_EVENT']

        try:
            df = pd.read_csv(filepath, header=None, names=column_names,
                             na_values=na_values)
            # Convert the 'age' column to numeric
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print("\n\thttps://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(os.path.abspath(__file__), 'data', 'raw', 'heart_failure', 'heart_failure_clinical_records_dataset.csv'))))
            import sys
            sys.exit(1)

        super(HeartFailureDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
   

# test the class
def test():
    dataset = HeartFailureDataset()
    print('FUll DATASET\n',dataset)
    print('FEATURES\n',dataset.features)
    print('LABELS\n',dataset.labels)
    print(dataset.protected_attribute_names)
    print(dataset.protected_attributes)
    print(dataset.instance_weights)
    print(dataset.metadata)

if __name__ == "__main__":
    test()