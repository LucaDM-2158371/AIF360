import os
import pandas as pd

from aif360.datasets import StandardDataset

class DropoutChanceDataset(StandardDataset):
    def __init__(self, label_name='Target', favorable_classes=[1],
                 protected_attribute_names=['Gender', 'Age_at_enrollment'],
                 privileged_classes=[[1], lambda x: x < 24],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=None):
        
        # get filepath from dropoutchance dataset
        filepath = "aif360/data/raw/dropout/data.csv"
        
        column_names = ['Marital_status', 
                        'Application_mode',
                        'Application_order',
                        'Course',
                        'Daytime/evening_attendance',
                        'Previous_qualification',
                        'Previous_qualification(grade)',
                        'Nationality',
                        "Mother's_qualification",
                        "Father's_qualification",
                        "Mother's_occupation",
                        "Father's_occupation",
                        'Admission_grade',
                        'Displaced',
                        'Educational_special_needs',
                        'Debtor',
                        'Tuition_fees_up_to_date',
                        'Gender',
                        'Scholarship_holder',
                        'Age_at_enrollment',
                        'International',
                        'Curricular_units_1st_sem(credited)',
                        'Curricular_units_1st_sem(enrolled)',
                        'Curricular_units_1st_sem(evaluations)',
                        'Curricular_units_1st_sem(approved)',
                        'Curricular_units_1st_sem(grade)',
                        'Curricular_units_1st_sem(without_evaluations)',
                        'Curricular_units_2nd_sem(credited)',
                        'Curricular_units_2nd_sem(enrolled)',
                        'Curricular_units_2nd_sem(evaluations)',
                        'Curricular_units_2nd_sem(approved)',
                        'Curricular_units_2nd_sem(grade)',
                        'Curricular_units_2nd_sem(without_evaluations)',
                        'Unemployment_rate',
                        'Inflation_rate',
                        'GDP',
                        'Target']

        try:
            df = pd.read_csv(filepath, header=None, names=column_names, sep=';', na_values=na_values)
            df['Target'] = df['Target'].map({'Dropout': 0, 'Graduate': 1, 'Enrolled': 1})
            
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print("\n\thttps://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(os.path.abspath(__file__), 'data', 'raw', 'dropoutchance', 'dropoutchance.csv'))))
            import sys
            sys.exit(1)

        super(DropoutChanceDataset, self).__init__(df=df, label_name=label_name,
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
    dataset = DropoutChanceDataset()
    print(dataset)
    print(dataset.features)
    print(dataset.labels)
    print(dataset.protected_attribute_names)
    print(dataset.protected_attributes)
    print(dataset.instance_weights)
    print(dataset.metadata)

if __name__ == "__main__":
    test()